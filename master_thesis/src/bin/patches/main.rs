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

    structure: Rc<RefCell<Structure>>,
    structure_bgs: Vec<BindGroup>,
    structure_pvs: StructurePvsField,

    /// (Structure Index in `structure` array, Translation, Rotation)
    structure_transforms: (usize, Mat4, Mat4),

    structure_transforms_gpu: Buffer,
    structure_transforms_bg: BindGroup,

    ssao_module: ssao::SsaoModule,
    ssao_finals: [TextureView; 2],

    output_pipeline: RenderPipeline,
    output_bind_group: BindGroup,

    postprocess_module: PostProcessModule,
}

impl framework::ApplicationStructure for Application {
    fn required_features() -> wgpu::Features {
        Features::PUSH_CONSTANTS
            | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | Features::TIMESTAMP_QUERY
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
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
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
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStage::all(),
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
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
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
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
        let path = &args[1];
        let (structure, structure_bgs) =
            Structure::from_ron_with_bgs(&device, path, &per_molecule_bind_group_layout);
        let structure = Rc::new(RefCell::new(structure));

        let mut colors: std::collections::HashMap<String, Vec<f32>> =
            ron::de::from_str(&std::fs::read_to_string("colors.ron").unwrap())
                .expect("Unable to load colors.");

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

        {
            let mut total_molecules = 0usize;
            let mut total_atoms = 0usize;
            let s = structure.borrow();
            for molecule_index in 0..s.molecules().len() {
                let molecule = &s.molecules()[molecule_index];
                let molecule_atoms = molecule.lods()[0].1.end;

                let num_molecules = s.transforms()[molecule_index].1;
                total_atoms += molecule_atoms as usize * num_molecules as usize;
                total_molecules += num_molecules;
            }

            println!("Total molecules: {}", total_molecules);
            println!("Total atoms: {}", total_atoms);
        }

        // Pipelines
        let billboards_pipeline = SphereBillboardsPipeline::new_debug(
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
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let multisampled_texture = device
            .create_texture(&TextureDescriptor {
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: sc_desc.format,
                usage: TextureUsage::RENDER_ATTACHMENT,
                label: None,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let normals_texture = device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsage::RENDER_ATTACHMENT
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
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Uint,
                usage: TextureUsage::RENDER_ATTACHMENT
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
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::RENDER_ATTACHMENT
                    | TextureUsage::SAMPLED
                    | TextureUsage::STORAGE,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let pvs_module = Rc::new(StructurePvsModule::new(
            &device,
            &camera_bind_group_layout,
            &per_molecule_bind_group_layout,
        ));

        let structure_pvs = pvs_module.pvs_field(
            &device,
            &camera_bind_group_layout,
            structure.clone(),
            5,
            32,
        );

        let mut structure_transforms: (usize, Mat4, Mat4) = (0, Mat4::identity(), Mat4::identity());

        let structure_transforms_gpu = {
            let mut raw: Vec<f32> = Vec::new();

            let transform: Mat4 = (structure_transforms.1) * (structure_transforms.2);
            raw.extend_from_slice(transform.as_slice());
            raw.extend_from_slice(&[0.0; 48]);

            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: cast_slice(&raw),
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            })
        };

        let structure_transforms_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &per_structure_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &structure_transforms_gpu,
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
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Float,
                    usage: TextureUsage::RENDER_ATTACHMENT
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
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Float,
                    usage: TextureUsage::RENDER_ATTACHMENT
                        | TextureUsage::STORAGE
                        | TextureUsage::SAMPLED,
                })
                .create_view(&TextureViewDescriptor::default()),
        ];
        let postprocess_module = PostProcessModule::new(device, width, height);

        let output_vs = device.create_shader_module(&include_spirv!("passthrough.vert.spv"));
        let output_fs = device.create_shader_module(&include_spirv!("passthrough.frag.spv"));

        let output_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Output"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::all(),
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let output_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&output_bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStage::FRAGMENT,
                range: 0..16,
            }],
        });

        let output_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Main output"),
            layout: Some(&output_pipeline_layout),
            vertex: VertexState {
                module: &output_vs,
                entry_point: "main",
                buffers: &[],
            },
            primitive: PrimitiveState {
                cull_mode: Some(Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            fragment: Some(FragmentState {
                module: &output_fs,
                entry_point: "main",
                targets: &[
                    // Output color
                    ColorTargetState {
                        format: sc_desc.format,
                        write_mask: ColorWrite::ALL,
                        blend: None,
                    },
                ],
            }),
            multisample: wgpu::MultisampleState::default(),
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
            label: Some("Output bind group"),
            layout: &output_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&postprocess_module.temporary_textures[1]),
            }],
        });

        let state = ApplicationState {
            draw_lod: false,
            draw_occluded: true,
            animating: false,
            animating_reveal: structure.borrow().bounding_radius(),

            ssao_settings: [
                ssao::Settings {
                    radius: 1200.0,
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
                    radius: 50.0,
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
            fog_distance: 24000.0,

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

            structure,
            structure_bgs,
            structure_pvs,
            structure_transforms,
            structure_transforms_gpu,
            structure_transforms_bg,

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

                                for molecule in self.structure.borrow_mut().molecules_mut() {
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
                            VirtualKeyCode::Plus => {
                                if self.state.fog_modifying {
                                    self.state.fog_distance += 100.0;
                                } else {
                                    self.state.ssao_settings[self.state.ssao_modifying]
                                        .add(self.state.ssao_parameter);
                                }
                                changed = true;
                            }
                            VirtualKeyCode::Minus => {
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
                            VirtualKeyCode::Space => {}
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
        // for r in self.structure_transforms {
        //     r.2 = rotation(0.2f32.to_radians(), &vec3(0.0, 1.0, 0.0)) * (r.2);
        // }

        // //================== DATA UPLOAD
        if self.state.animating {
            let sub = self.structure.borrow().bounding_radius() / 100.0;
            if self.camera.distance() <= self.structure.borrow().bounding_radius() * 2.0 {
                self.state.animating_reveal -= if self.state.animating_reveal <= 0.0 {
                    0.0
                } else {
                    sub
                };
                println!("{}", self.state.animating_reveal);
            } else {
                self.state.animating_reveal = self.structure.borrow().bounding_radius();
                self.camera.set_distance(self.camera.distance() - 100.0);
            }
        }
        self.camera.update_gpu(queue);

        let mut structure_transforms_f32: Vec<f32> = Vec::new();
        let transform: Mat4 = self.structure_transforms.1 * self.structure_transforms.2;
        structure_transforms_f32.extend_from_slice(transform.as_slice());
        structure_transforms_f32.extend_from_slice(&[0.0; 48]);

        queue.write_buffer(
            &self.structure_transforms_gpu,
            0,
            cast_slice(&structure_transforms_f32),
        );

        //================== DATA UPLOAD
        let eye = self.camera.eye();
        let eye = vec3(eye.x as f32, eye.y as f32, eye.z as f32);

        let rotation = self.structure_transforms.2.fixed_slice::<U3, U3>(0, 0);
        let position = self.structure_transforms.1.column(3).xyz();

        let direction = rotation.try_inverse().unwrap() * normalize(&(eye - position));

        if futures::executor::block_on(self.structure_pvs.compute_from_eye(
            device,
            queue,
            vec3(direction.x as f64, direction.y as f64, direction.z as f64),
        )) {}

        if futures::executor::block_on(self.structure_pvs.compute_from_eye(
            device,
            queue,
            -vec3(direction.x as f64, direction.y as f64, direction.z as f64),
        )) {}

        // Calculate molecules
        {
            let structure_id = 0;
            let structure = self.structure.borrow();
            let structure_pvs = &self.structure_pvs;

            let mut total_molecules: u32 = 0;
            let mut visible_molecules: u32 = 0;
            for molecule_id in 0..structure.molecules().len() {
                total_molecules += structure.transforms()[molecule_id].1 as u32;
                
                let view = structure_pvs.get_from_eye(vec3(direction.x, direction.y, direction.z)).unwrap();

                for range in view.visible[molecule_id].iter() {
                    visible_molecules += range.1 - range.0;
                }
            }

            println!("Total molecules: {}/{} = {}%", visible_molecules, total_molecules, (visible_molecules as f32 / total_molecules as f32) * 100.0);
        }

        //================== RENDER MOLECULES
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render molecules"),
                color_attachments: &[
                    RenderPassColorAttachmentDescriptor {
                        attachment: &self.output_texture,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::WHITE),
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

            let structure_id = self.structure_transforms.0;
            let structure = self.structure.borrow();
            let structure_pvs = &self.structure_pvs;

            let rotation = self.structure_transforms.2.fixed_slice::<U3, U3>(0, 0);
            let position = self.structure_transforms.1.column(3).xyz();

            let eye = self.camera.eye();
            let eye = vec3(eye.x as f32, eye.y as f32, eye.z as f32);

            let rotation = self.structure_transforms.2.fixed_slice::<U3, U3>(0, 0);
            let position = self.structure_transforms.1.column(3).xyz();

            let direction = eye - position;
            let direction_norm_rot = rotation.try_inverse().unwrap() * normalize(&direction);

            let distance = (direction.magnitude() - 2.0 * structure.bounding_radius()).max(1.0);

            rpass.set_bind_group(2, &self.structure_transforms_bg, &[(0 * 256) as u32]);
            rpass.set_push_constants(
                ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                4,
                cast_slice(&[(0 + 1) as u32]),
            );

            rpass.set_push_constants(
                ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                8,
                cast_slice(&[self.state.animating_reveal]),
            );

            let draw_occluded = self.state.draw_occluded;
            let draw_lod = self.state.draw_lod;

            // For each molecule type
            for molecule_id in 0..structure.molecules().len() {
                // Bind its data
                rpass.set_bind_group(1, &self.structure_bgs[molecule_id], &[]);

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
        self.postprocess_module.options.ssao_pow =
            [self.state.ssao_settings[0].x, self.state.ssao_settings[1].x];
        self.postprocess_module.options.fog = self.state.fog_distance;
        self.postprocess_module.compute(
            &mut self.camera,
            device,
            &mut encoder,
            &self.output_texture,
            &self.depth_texture,
            [&self.ssao_finals[0], &self.ssao_finals[1]],
            &self.instance_texture,
            time,
        );
        queue.submit(Some(encoder.finish()));

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::WHITE),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.output_pipeline);
            let resolution: [f32; 2] = [self.width as f32, self.height as f32];
            rpass.set_push_constants(ShaderStage::FRAGMENT, 0, cast_slice(&resolution));
            rpass.set_bind_group(0, &self.output_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

fn main() {
    framework::run::<Application>("Occlusion");
}
