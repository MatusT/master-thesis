use master_thesis::camera::*;
use master_thesis::framework;
use master_thesis::frustrum_culler::*;
use master_thesis::pipelines::SphereBillboardsPipeline;
use master_thesis::postprocess::*;
use master_thesis::pvs::*;
use master_thesis::ssao;
use master_thesis::structure::*;

use bytemuck::cast_slice;
use futures::task::LocalSpawn;
use nalgebra_glm::*;
use rand::distributions::Distribution;
use wgpu::util::*;
use wgpu::*;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

struct ApplicationState {
    pub draw_lod: bool,
    pub draw_occluded: bool,
    pub animating: bool,
    pub animating_reveal: f32,

    pub ssao_settings: [ssao::Settings; 2],
    pub ssao_modifying: usize,
    pub ssao_parameter: u32,

    pub fog_modifying: bool,
    pub fog_distance: f32,

    pub render_mode: u32,
}

pub struct Application {
    width: u32,
    height: u32,

    state: ApplicationState,
    start_time: Instant,

    depth_texture: TextureView,
    multisampled_texture: TextureView,
    normals_texture: TextureView,
    instance_texture: TextureView,
    output_texture: TextureView,

    camera: RotationCamera,

    billboards_pipeline: SphereBillboardsPipeline,

    pvs_module: Rc<StructurePvsModule>,

    structures: Vec<Rc<RefCell<Structure>>>,
    structures_bgs: Vec<Vec<BindGroup>>,
    structures_pvs: Vec<StructurePvsField>,

    /// (Structure Index in `structure` array, Translation, Rotation)
    structures_transforms: Vec<(usize, Mat4, Mat4)>,

    structures_transforms_gpu: Buffer,
    structures_transforms_bg: BindGroup,

    ssao_module: ssao::SsaoModule,
    ssao_finals: [TextureView; 2],

    output_pipeline: RenderPipeline,
    output_bind_group: BindGroup,

    postprocess_module: PostProcessModule,
}

impl framework::ApplicationStructure for Application {
    fn required_features() -> wgpu::Features {
        wgpu::Features::PUSH_CONSTANTS
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_push_constant_size: 32,
            ..wgpu::Limits::default()
        }
    }

    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let width = sc_desc.width;
        let height = sc_desc.height;
        let sample_count = 1;

        // Shared bind group layouts
        let camera_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Camera bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::all(),
                    ty: BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: Some(CameraUbo::size()),
                    },
                    count: None,
                }],
            });

        let per_molecule_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Molecule bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStage::all(),
                        ty: BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStage::all(),
                        ty: BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let per_structure_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Structure bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::VERTEX,
                    ty: BindingType::UniformBuffer {
                        dynamic: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Camera
        let mut camera = RotationCamera::new(
            &device,
            &camera_bind_group_layout,
            &reversed_infinite_perspective_rh_zo(
                sc_desc.width as f32 / sc_desc.height as f32,
                0.785398163,
                0.1,
            ),
            1400.0,
            100.0,
        );
        camera.set_yaw(1.4100000000000015);
        camera.set_pitch(1.5207963267948965);

        // Data
        let args: Vec<String> = std::env::args().collect();

        let mut structures = Vec::new();
        let mut structures_bgs = Vec::new();
        for path in args[1..].iter() {
            let (structure, structure_bgs) =
                Structure::from_ron_with_bgs(&device, path, &per_molecule_bind_group_layout);

            structures.push(Rc::new(RefCell::new(structure)));
            structures_bgs.push(structure_bgs)
        }

        let mut colors: std::collections::HashMap<String, Vec<f32>> =
            ron::de::from_str(&std::fs::read_to_string("colors.ron").unwrap())
                .expect("Unable to load colors.");

        for structure in structures.iter_mut() {
            for molecule in structure.borrow_mut().molecules_mut() {
                if let Some(color) = colors.get_mut(molecule.name()) {
                    for channel in color.iter_mut() {
                        if *channel > 1.0 {
                            *channel = *channel / 255.0;
                        }
                    }
                    molecule.set_color(&vec3(color[0], color[1], color[2]));
                }
            }
        }

        // Pipelines
        let billboards_pipeline = SphereBillboardsPipeline::new_normals(
            &device,
            &camera_bind_group_layout,
            &per_molecule_bind_group_layout,
            &per_structure_bind_group_layout,
            sample_count,
        );

        // Default framebuffer
        let depth_texture = device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let multisampled_texture = device
            .create_texture(&TextureDescriptor {
                size: Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: sc_desc.format,
                usage: TextureUsage::OUTPUT_ATTACHMENT,
                label: None,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let normals_texture = device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsage::OUTPUT_ATTACHMENT
                    | TextureUsage::SAMPLED
                    | TextureUsage::STORAGE,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let instance_texture = device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Uint,
                usage: TextureUsage::OUTPUT_ATTACHMENT
                    | TextureUsage::SAMPLED
                    | TextureUsage::STORAGE,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let output_texture = device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::OUTPUT_ATTACHMENT
                    | TextureUsage::SAMPLED
                    | TextureUsage::STORAGE,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let pvs_module = Rc::new(StructurePvsModule::new(
            &device,
            &camera_bind_group_layout,
            &per_molecule_bind_group_layout,
        ));
        let structures_pvs = structures
            .iter()
            .map(|structure| {
                pvs_module.pvs_field(
                    &device,
                    &camera_bind_group_layout,
                    structure.clone(),
                    15,
                    256,
                )
            })
            .collect();

        let n = 7;
        let n3 = n * n * n;

        let mut structures_transforms: Vec<(usize, Mat4, Mat4)> = Vec::new();
        let structure_rand = rand_distr::Uniform::from(0..structures.len());

        structures_transforms.push((0, Mat4::identity(), Mat4::identity()));
        for i in 1..n3 {
            let structure_id = structure_rand.sample(&mut rand::thread_rng());
            let [x, y, z]: [f32; 3] = rand_distr::UnitBall.sample(&mut rand::thread_rng());

            let rotation: Mat4 = rotation((i as f32).to_radians(), &vec3(0.0, 1.0, 0.0));

            let radius = structures[structure_id].borrow_mut().bounding_radius();
            let position = vec3(
                x * radius * 1.75 * n as f32,
                y * radius * 1.75 * n as f32,
                z * radius * 1.75 * n as f32,
            );
            let translation = translation(&position);

            structures_transforms.push((structure_id, translation, rotation));
        }

        let mut indices_to_delete = Vec::new();
        for i in 0..structures_transforms.len() {
            for j in (i + 1)..structures_transforms.len() {
                let r1 = structures[structures_transforms[i].0]
                    .borrow()
                    .bounding_radius();
                let r2 = structures[structures_transforms[j].0]
                    .borrow()
                    .bounding_radius();

                let pos1 = structures_transforms[i].1.column(3).xyz();
                let pos2 = structures_transforms[j].1.column(3).xyz();

                let dist = distance(&pos1, &pos2);
                let dist = dist - r1 - r2;

                if dist < 0.0 {
                    indices_to_delete.push(j);
                }
            }
        }

        let structures_transforms_old = structures_transforms;
        let mut structures_transforms = Vec::new();

        for (i, s) in structures_transforms_old.iter().enumerate() {
            if !indices_to_delete.contains(&i) {
                structures_transforms.push(*s);
            }
        }

        println!("Amount of structures: {}", structures_transforms.len());

        let structures_transforms_gpu = {
            let mut raw: Vec<f32> = Vec::new();
            for (_, translation, rotation) in structures_transforms.iter() {
                let transform: Mat4 = (*translation) * (*rotation);
                raw.extend_from_slice(transform.as_slice());
                raw.extend_from_slice(&[0.0; 48]);
            }
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: cast_slice(&raw),
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            })
        };

        let structures_transforms_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &per_structure_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &structures_transforms_gpu,
                    offset: 0,
                    size: std::num::NonZeroU64::new(256),
                },
            }],
        });

        let ssao_module = ssao::SsaoModule::new(&device, width, height);
        let ssao_finals = [
            device
                .create_texture(&TextureDescriptor {
                    label: Some("Final results"),
                    size: Extent3d {
                        width,
                        height,
                        depth: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Float,
                    usage: TextureUsage::OUTPUT_ATTACHMENT
                        | TextureUsage::STORAGE
                        | TextureUsage::SAMPLED,
                })
                .create_view(&TextureViewDescriptor::default()),
            device
                .create_texture(&TextureDescriptor {
                    label: Some("Final results"),
                    size: Extent3d {
                        width,
                        height,
                        depth: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Float,
                    usage: TextureUsage::OUTPUT_ATTACHMENT
                        | TextureUsage::STORAGE
                        | TextureUsage::SAMPLED,
                })
                .create_view(&TextureViewDescriptor::default()),
        ];

        let postprocess_module = PostProcessModule::new(device, width, height);

        let output_vs = device.create_shader_module(include_spirv!("passthrough.vert.spv"));
        let output_fs = device.create_shader_module(include_spirv!("passthrough.frag.spv"));

        let output_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStage::all(),
                        ty: BindingType::Sampler { comparison: false },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStage::all(),
                        ty: BindingType::SampledTexture {
                            dimension: TextureViewDimension::D2,
                            component_type: TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStage::all(),
                        ty: BindingType::SampledTexture {
                            dimension: TextureViewDimension::D2,
                            component_type: TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStage::all(),
                        ty: BindingType::SampledTexture {
                            dimension: TextureViewDimension::D2,
                            component_type: TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStage::all(),
                        ty: BindingType::SampledTexture {
                            dimension: TextureViewDimension::D2,
                            component_type: TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // BindGroupLayoutEntry {
                    //     binding: 5,
                    //     visibility: ShaderStage::all(),
                    //     ty: BindingType::SampledTexture {
                    //         dimension: TextureViewDimension::D2,
                    //         component_type: TextureComponentType::Uint,
                    //         multisampled: false,
                    //     },
                    //     count: None,
                    // },
                ],
            });

        let output_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&output_bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStage::FRAGMENT,
                range: 0..24,
            }],
        });

        let output_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&output_pipeline_layout),
            vertex_stage: ProgrammableStageDescriptor {
                module: &output_vs,
                entry_point: "main",
            },
            fragment_stage: Some(ProgrammableStageDescriptor {
                module: &output_fs,
                entry_point: "main",
            }),
            rasterization_state: Some(RasterizationStateDescriptor {
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
                polygon_mode: PolygonMode::Fill,
            }),
            primitive_topology: PrimitiveTopology::TriangleList,
            color_states: &[ColorStateDescriptor {
                format: sc_desc.format,
                color_blend: BlendDescriptor::REPLACE,
                alpha_blend: BlendDescriptor::REPLACE,
                write_mask: ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: VertexStateDescriptor {
                index_format: IndexFormat::Uint16,
                vertex_buffers: &[],
            },
            sample_count,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let linear_clamp_sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            ..Default::default()
        });

        let output_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &output_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&linear_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&output_texture),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&ssao_finals[0]),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&ssao_finals[1]),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&depth_texture),
                },
                // BindGroupEntry {
                //     binding: 5,
                //     resource: BindingResource::TextureView(&instance_texture),
                // },
            ],
        });

        let state = ApplicationState {
            draw_lod: true,
            draw_occluded: false,
            animating: false,
            animating_reveal: structures[0].borrow().bounding_radius(),

            ssao_settings: [
                ssao::Settings {
                    radius: 1000.0,
                    projection: camera.ubo().projection,
                    shadowMultiplier: 1.0,
                    shadowPower: 1.0,
                    horizonAngleThreshold: 0.0,
                    sharpness: 0.0,
                    detailShadowStrength: 0.0,
                    blurPassCount: 1,
                    x: 1.0,
                    ..Default::default()
                },
                ssao::Settings {
                    radius: 100.0,
                    projection: camera.ubo().projection,
                    shadowMultiplier: 1.0,
                    shadowPower: 1.0,
                    horizonAngleThreshold: 0.0,
                    sharpness: 0.0,
                    detailShadowStrength: 0.0,
                    blurPassCount: 1,
                    x: 1.0,
                    ..Default::default()
                },
            ],
            ssao_modifying: 0,
            ssao_parameter: 0,

            fog_modifying: false,
            fog_distance: 50000.0,

            render_mode: 0,
        };

        let start_time = Instant::now();        

        Self {
            width,
            height,

            state,
            start_time,

            depth_texture,
            multisampled_texture,
            normals_texture,
            instance_texture,
            output_texture,

            camera,

            billboards_pipeline,

            pvs_module: pvs_module.clone(),

            structures,
            structures_bgs,
            structures_pvs,
            structures_transforms,
            structures_transforms_gpu,
            structures_transforms_bg,

            ssao_module,
            ssao_finals,

            output_pipeline,
            output_bind_group,

            postprocess_module,
        }
    }

    fn resize(
        &mut self,
        sc_desc: &wgpu::SwapChainDescriptor,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        //
    }

    fn window_event(&mut self, event: winit::event::WindowEvent) {
        use winit::event::ElementState;
        use winit::event::VirtualKeyCode;
        use winit::event::WindowEvent::*;

        self.camera.window_event(&event);

        let mut changed = false;
        match event {
            KeyboardInput { input, .. } => {
                if input.state == ElementState::Pressed {
                    if let Some(keycode) = input.virtual_keycode {
                        match keycode {
                            VirtualKeyCode::L => {
                                self.state.draw_lod = !self.state.draw_lod;
                            }
                            VirtualKeyCode::O => {
                                self.state.draw_occluded = !self.state.draw_occluded;
                            }
                            VirtualKeyCode::C => {
                                let mut colors: std::collections::HashMap<String, Vec<f32>> =
                                    ron::de::from_str(
                                        &std::fs::read_to_string("colors.ron").unwrap(),
                                    )
                                    .expect("Unable to load colors.");

                                for structure in self.structures.iter_mut() {
                                    for molecule in structure.borrow_mut().molecules_mut() {
                                        if let Some(color) = colors.get_mut(molecule.name()) {
                                            for channel in color.iter_mut() {
                                                if *channel > 1.0 {
                                                    *channel = *channel / 255.0;
                                                }
                                            }
                                            molecule.set_color(&vec3(color[0], color[1], color[2]));
                                        }
                                    }
                                }
                            }
                            VirtualKeyCode::F => {
                                self.state.fog_modifying = !self.state.fog_modifying;
                                changed = true;
                            }
                            VirtualKeyCode::Key1 => {
                                self.state.ssao_modifying = 0;
                                changed = true;
                                self.state.fog_modifying = false;
                            }
                            VirtualKeyCode::Key2 => {
                                self.state.ssao_modifying = 1;
                                changed = true;
                                self.state.fog_modifying = false;
                            }
                            VirtualKeyCode::Add => {
                                if self.state.fog_modifying {
                                    self.state.fog_distance += 100.0;
                                } else {
                                    self.state.ssao_settings[self.state.ssao_modifying]
                                        .add(self.state.ssao_parameter);
                                }
                                changed = true;
                            }
                            VirtualKeyCode::Subtract => {
                                if self.state.fog_modifying {
                                    self.state.fog_distance -= 100.0;
                                } else {
                                    self.state.ssao_settings[self.state.ssao_modifying]
                                        .sub(self.state.ssao_parameter);
                                }
                                changed = true;
                            }
                            VirtualKeyCode::Up => {
                                self.state.ssao_parameter -=
                                    if self.state.ssao_parameter == 0 { 0 } else { 1 };
                                changed = true;
                                self.state.fog_modifying = false;
                            }
                            VirtualKeyCode::Down => {
                                self.state.ssao_parameter +=
                                    if self.state.ssao_parameter == 6 { 0 } else { 1 };
                                changed = true;
                                self.state.fog_modifying = false;
                            }
                            VirtualKeyCode::A => {
                                self.state.animating = !self.state.animating;
                            }
                            VirtualKeyCode::S => {
                                self.state.render_mode = (self.state.render_mode + 1) % 3;
                            }
                            // Set focus
                            VirtualKeyCode::N => {
                                let addsub = 10.0f32.powf(
                                    -self.postprocess_module.options.focus.log10().abs().ceil(),
                                );
                                self.postprocess_module.options.focus += addsub;
                                println!("Focus: {}", self.postprocess_module.options.focus);
                            }
                            VirtualKeyCode::M => {
                                let addsub = 10.0f32.powf(
                                    -self.postprocess_module.options.dof.log10().abs().ceil(),
                                );
                                self.postprocess_module.options.dof += addsub;
                                println!("Dof: {}", self.postprocess_module.options.dof);
                            }
                            VirtualKeyCode::Space => {

                            }
                            _ => {}
                        };
                    }
                }
            }
            _ => {}
        };

        if changed {
            println!("");
            println!(
                "Ssao settings {}",
                if self.state.ssao_modifying == 0 {
                    "Far"
                } else {
                    "Near"
                }
            );
            println!(
                "[{}] Shadow multiplier: {}",
                if self.state.ssao_parameter != 0 {
                    " "
                } else {
                    "*"
                },
                self.state.ssao_settings[self.state.ssao_modifying].shadowMultiplier,
            );
            println!(
                "[{}] Shadow power: {}",
                if self.state.ssao_parameter != 1 {
                    " "
                } else {
                    "*"
                },
                self.state.ssao_settings[self.state.ssao_modifying].shadowPower,
            );
            println!(
                "[{}] Horizon angle threshold: {}",
                if self.state.ssao_parameter != 2 {
                    " "
                } else {
                    "*"
                },
                self.state.ssao_settings[self.state.ssao_modifying].horizonAngleThreshold,
            );
            println!(
                "[{}] Sharpness: {}",
                if self.state.ssao_parameter != 3 {
                    " "
                } else {
                    "*"
                },
                self.state.ssao_settings[self.state.ssao_modifying].sharpness,
            );
            println!(
                "[{}] Detail shadow strength: {}",
                if self.state.ssao_parameter != 4 {
                    " "
                } else {
                    "*"
                },
                self.state.ssao_settings[self.state.ssao_modifying].detailShadowStrength,
            );
            println!(
                "[{}] Radius: {}",
                if self.state.ssao_parameter != 5 {
                    " "
                } else {
                    "*"
                },
                self.state.ssao_settings[self.state.ssao_modifying].radius,
            );
            println!(
                "[{}] Pow blend: {}",
                if self.state.ssao_parameter != 6 {
                    " "
                } else {
                    "*"
                },
                self.state.ssao_settings[self.state.ssao_modifying].x,
            );

            println!(
                "[{}] Fog: {}",
                if self.state.fog_modifying { "*" } else { " " },
                self.state.fog_distance,
            );
        }
    }

    fn device_event(&mut self, event: winit::event::DeviceEvent) {
        self.camera.device_event(&event);
    }

    fn render(
        &mut self,
        frame: &wgpu::SwapChainTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spawner: &impl futures::task::LocalSpawn,
    ) {
        let time = Instant::now().duration_since(self.start_time);
        let time = time.as_secs_f32() + time.subsec_millis() as f32;

        // Rotate the structure
        for r in self.structures_transforms[1..].iter_mut() {
            r.2 = rotation(0.2f32.to_radians(), &vec3(0.0, 1.0, 0.0)) * (r.2);
        }

        //================== DATA UPLOAD
        if self.state.animating {
            let sub = self.structures[0].borrow().bounding_radius() / 100.0;
            if self.camera.distance() <= self.structures[0].borrow().bounding_radius() * 2.0 {
                self.state.animating_reveal -= if self.state.animating_reveal <= 0.0 { 0.0 } else { sub }; 
                println!("{}", self.state.animating_reveal);
            } else {
                self.state.animating_reveal = self.structures[0].borrow().bounding_radius();
                self.camera.set_distance(self.camera.distance() - 100.0);
            }
        }
        self.camera.update_gpu(queue);

        let culler = FrustrumCuller::from_matrix(self.camera.ubo().projection_view);

        let mut structures_transforms_f32: Vec<f32> = Vec::new();
        for (_, translation, rotation) in self.structures_transforms.iter() {
            let transform: Mat4 = translation * rotation;
            structures_transforms_f32.extend_from_slice(transform.as_slice());
            structures_transforms_f32.extend_from_slice(&[0.0; 48]);
        }

        queue.write_buffer(
            &self.structures_transforms_gpu,
            0,
            cast_slice(&structures_transforms_f32),
        );

        //================== DATA UPLOAD
        let mut computed_count = 0;
        for i in 0..self.structures_transforms.len() {
            if computed_count > 4 {
                continue;
            }

            let eye = self.camera.eye();
            let eye = vec3(eye.x as f32, eye.y as f32, eye.z as f32);

            let rotation = self.structures_transforms[i].2.fixed_slice::<U3, U3>(0, 0);
            let position = self.structures_transforms[i].1.column(3).xyz();

            let direction = rotation.try_inverse().unwrap() * normalize(&(eye - position));

            // let start = Instant::now();
            if futures::executor::block_on(
                self.structures_pvs[self.structures_transforms[i].0].compute_from_eye(
                    device,
                    queue,
                    vec3(direction.x as f64, direction.y as f64, direction.z as f64),
                ),
            ) {
                computed_count += 1;
                // println!("Occlusion time: {}", start.elapsed().as_millis());
            }
        }

        //================== RENDER MOLECULES
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[
                    RenderPassColorAttachmentDescriptor {
                        attachment: &self.output_texture,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::BLACK),
                            store: true,
                        },
                    },
                    RenderPassColorAttachmentDescriptor {
                        attachment: &self.instance_texture,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::BLACK),
                            store: true,
                        },
                    },
                    RenderPassColorAttachmentDescriptor {
                        attachment: &self.normals_texture,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::BLACK),
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            rpass.set_pipeline(&self.billboards_pipeline.pipeline);
            rpass.set_push_constants(ShaderStage::VERTEX, 0, cast_slice(&[time]));
            rpass.set_bind_group(0, &self.camera.bind_group(), &[]);

            for i in 0..self.structures_transforms.len() {
                let structure_id = self.structures_transforms[i].0;
                let structure = self.structures[structure_id].borrow();
                let structure_pvs = &self.structures_pvs[structure_id];

                let rotation = self.structures_transforms[i].2.fixed_slice::<U3, U3>(0, 0);
                let position = self.structures_transforms[i].1.column(3).xyz();

                if !culler.test_sphere(vec4(
                    position.x,
                    position.y,
                    position.z,
                    structure.bounding_radius(),
                )) {
                    continue;
                }

                let eye = self.camera.eye();
                let eye = vec3(eye.x as f32, eye.y as f32, eye.z as f32);

                let rotation = self.structures_transforms[i].2.fixed_slice::<U3, U3>(0, 0);
                let position = self.structures_transforms[i].1.column(3).xyz();

                let direction = eye - position;
                let direction_norm_rot = rotation.try_inverse().unwrap() * normalize(&direction);

                let distance = (direction.magnitude() - 2.0 * structure.bounding_radius()).max(1.0);

                rpass.set_bind_group(2, &self.structures_transforms_bg, &[(i * 256) as u32]);
                rpass.set_push_constants(
                    ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                    4,
                    cast_slice(&[(i + 1) as u32]),
                );
                if i != 0 {
                    rpass.set_push_constants(
                        ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                        8,
                        cast_slice(&[structure.bounding_radius()]),
                    );
                } else {
                    rpass.set_push_constants(
                        ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                        8,
                        cast_slice(&[self.state.animating_reveal]),
                    );
                }

                let draw_occluded = self.state.draw_occluded || (distance < structure.bounding_radius() * 2.0 && i == 0);
                // let draw_occluded =
                //     self.state.draw_occluded || distance < structure.bounding_radius() * 2.0;
                let draw_lod = self.state.draw_lod;

                // For each molecule type
                for molecule_id in 0..structure.molecules().len() {
                    let dont_cull = ["A", "G", "U", "C", "P", "NTD", "CTD"];
                    let molecule_name = structure.molecules()[molecule_id].name();
                    if i == 0 {
                        if dont_cull.contains(&molecule_name) {
                            rpass.set_push_constants(
                                ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                                8,
                                cast_slice(&[structure.bounding_radius()]),
                            );
                        } else {
                            rpass.set_push_constants(
                                ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                                8,
                                cast_slice(&[self.state.animating_reveal]),
                            );
                        }
                    }

                    // Bind its data
                    rpass.set_bind_group(1, &self.structures_bgs[structure_id][molecule_id], &[]);

                    // Set its colors
                    let color: [f32; 3] = structure.molecules()[molecule_id].color().into();
                    rpass.set_push_constants(ShaderStage::FRAGMENT, 16, cast_slice(&color));

                    // Find LOD
                    let lods = structure.molecules()[molecule_id].lods();
                    let (start, end) = if draw_lod {
                        let mut start = 0;
                        let mut end = 0;

                        for i in 0..lods.len() {
                            if (i == lods.len() - 1)
                                || (distance > lods[i].0 && distance < lods[i + 1].0)
                            {
                                start = lods[i].1.start;
                                end = lods[i].1.end;

                                break;
                            }
                        }

                        (start, end)
                    } else {
                        let start = lods[0].1.start;
                        let end = lods[0].1.end;

                        (start, end)
                    };

                    // IF !draw_occluded && PVS is available -> iterate only over visible parts
                    if !draw_occluded {
                        if let Some(pvs) = structure_pvs.get_from_eye(direction_norm_rot) {
                            for range in pvs.visible[molecule_id].iter() {
                                rpass.draw(start..end, range.0..range.1);
                            }
                            continue;
                        }
                    }

                    rpass.draw(start..end, 0..structure.transforms()[molecule_id].1 as u32);
                }
            }
        }
        queue.submit(Some(encoder.finish()));

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        self.ssao_module.compute(
            device,
            queue,
            &mut encoder,
            &self.state.ssao_settings[0],
            &self.depth_texture,
            // None,
            Some(&self.normals_texture),
            &self.ssao_finals[0],
        );
        queue.submit(Some(encoder.finish()));

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        self.ssao_module.compute(
            device,
            queue,
            &mut encoder,
            &self.state.ssao_settings[1],
            &self.depth_texture,
            // None,
            Some(&self.normals_texture),
            &self.ssao_finals[1],
        );
        queue.submit(Some(encoder.finish()));

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        self.postprocess_module.compute(
            device,
            &mut encoder,
            &self.output_texture,
            &self.depth_texture,
            &self.instance_texture,
            time,
        );
        queue.submit(Some(encoder.finish()));

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let depth_unpack_mul = -self.camera.ubo().projection[(2, 3)];
            let mut depth_unpack_add = -self.camera.ubo().projection[(2, 2)];
            if depth_unpack_mul * depth_unpack_add < 0.0 {
                depth_unpack_add = -depth_unpack_add;
            }

            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.output_pipeline);
            rpass.set_push_constants(
                ShaderStage::FRAGMENT,
                0,
                cast_slice(&[depth_unpack_mul, depth_unpack_add, self.state.fog_distance]),
            );
            rpass.set_push_constants(
                ShaderStage::FRAGMENT,
                12,
                cast_slice(&[self.state.ssao_settings[0].x]),
            );
            rpass.set_push_constants(
                ShaderStage::FRAGMENT,
                16,
                cast_slice(&[self.state.ssao_settings[1].x]),
            );
            rpass.set_bind_group(0, &self.output_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

fn main() {
    framework::run::<Application>("Occlusion");
}
