use master_thesis::camera::*;
use master_thesis::pipelines::SphereBillboardsPipeline;
use master_thesis::pvs::*;
use master_thesis::structure::*;
use master_thesis::ApplicationEvent;
use master_thesis::ssao;

use bytemuck::cast_slice;
use nalgebra_glm::*;
use wgpu::*;
use wgpu::util::*;

use std::rc::Rc;

struct ApplicationState {
    pub draw_lod: bool,
    pub draw_occluded: bool,
}

pub struct Application {
    width: u32,
    height: u32,

    device: Device,
    queue: Queue,

    state: ApplicationState,

    depth_texture: TextureView,
    multisampled_texture: TextureView,
    normals_texture: TextureView,
    output_texture: TextureView,

    camera: RotationCamera,

    billboards_pipeline: SphereBillboardsPipeline,

    pvs_module: Rc<StructurePvsModule>,

    covid: Rc<Structure>,
    covid_pvs: StructurePvsField,
    covid_transforms: Vec<Mat4>,
    covid_transforms_gpu: Buffer,
    covid_transforms_bgs: Vec<BindGroup>,

    ssao_module: ssao::SsaoModule,
    ssao_settings: ssao::Settings,

    output_pipeline: RenderPipeline,
    output_bind_group: BindGroup,
}

impl Application {
    pub fn new(
        width: u32,
        height: u32,
        device: Device,
        queue: Queue,
        swapchain_format: TextureFormat,
        sample_count: u32,
    ) -> Self {
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
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Camera
        let mut camera = RotationCamera::new(
            &device,
            &camera_bind_group_layout,
            &reversed_infinite_perspective_rh_zo(width as f32 / height as f32, 0.785398163, 0.1),
            6.0 * 1500.0,
            100.0,
        );

        // Data
        let args: Vec<String> = std::env::args().collect();

        let covid = Rc::new(Structure::from_ron(
            &device,
            &args[1],
            &per_molecule_bind_group_layout,
        ));

        // Pipelines
        let billboards_pipeline = SphereBillboardsPipeline::new(
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
                format: swapchain_format,
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
                usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED | TextureUsage::STORAGE,
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
                format: TextureFormat::Rgba32Float,
                usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED | TextureUsage::STORAGE,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let pvs_module = Rc::new(StructurePvsModule::new(
            &device,
            &camera_bind_group_layout,
            &per_molecule_bind_group_layout,
        ));
        let covid_pvs =
            pvs_module.pvs_field(&device, &camera_bind_group_layout, covid.clone(), 5, 128);

        let mut covid_transforms = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    // vec![rotation(90.0f32.to_radians(), &vec3(0.0, 1.0, 0.0)), translation(&vec3(2.0 * covid.bounding_radius(), 0.0, 0.0))];
                    let x = x as f32 * 2.0 * covid.bounding_radius();
                    let y = y as f32 * 2.0 * covid.bounding_radius();
                    let z = z as f32 * 2.0 * covid.bounding_radius();
                    covid_transforms.push(translation(&vec3(x, y, z)));
                }
            }
        }
        let covid_transforms_gpu = {
            let mut raw: Vec<f32> = Vec::new();
            for transform in &covid_transforms {
                raw.extend_from_slice(transform.as_slice());
                raw.extend_from_slice(&[0.0; 48]);
            }
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: cast_slice(&raw), usage: BufferUsage::UNIFORM })
        };
        let mut covid_transforms_bgs = Vec::new();
        for i in 0..covid_transforms.len() {
            let i = i as u64;
            covid_transforms_bgs.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &per_structure_bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(
                        covid_transforms_gpu.slice(i * 256..(i + 1) * 256),
                    ),
                }],
            }));
        }

        let ssao_module = ssao::SsaoModule::new(&device, width, height);
        let mut ssao_settings = ssao::Settings::default();
        ssao_settings.radius = 10.0;
        ssao_settings.projection = camera.ubo().projection;

        let output_vs = device.create_shader_module(include_spirv!("passthrough.vert.spv"));
        let output_fs = device.create_shader_module(include_spirv!("passthrough.frag.spv"));

        let output_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::all(),
                    ty: BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2,
                        format: TextureFormat::R32Float,
                        readonly: true,
                    },
                    count: None,
                }
            ],
        });

        let output_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                &output_bind_group_layout,
            ],
            push_constant_ranges: &[],
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
            }),
            primitive_topology: PrimitiveTopology::TriangleList,
            color_states: &[ColorStateDescriptor {
                format: swapchain_format,
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

        let output_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &output_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&ssao_module.final_result),
                },
            ],
        }); 

        let state = ApplicationState {
            draw_lod: true,
            draw_occluded: false,
        };

        Self {
            width,
            height,

            device,
            queue,

            state,

            depth_texture,
            multisampled_texture,
            normals_texture,
            output_texture,

            camera,

            billboards_pipeline,

            pvs_module: pvs_module.clone(),

            covid: covid.clone(),
            covid_pvs,
            covid_transforms,
            covid_transforms_gpu,
            covid_transforms_bgs,

            ssao_module,
            ssao_settings,

            output_pipeline,
            output_bind_group,
        }
    }

    pub fn resize(&mut self, _: u32, _: u32) {
        //
    }

    pub fn update<'a>(&mut self, event: &ApplicationEvent<'a>) {
        use winit::event::ElementState;
        use winit::event::VirtualKeyCode;
        use winit::event::WindowEvent::*;

        self.camera.update(event);

        match event {
            ApplicationEvent::WindowEvent(event) => {
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
                                    VirtualKeyCode::Add => {
                                        self.ssao_settings.radius += 1.0;
                                        println!("Radius: {}", self.ssao_settings.radius);
                                    }
                                    VirtualKeyCode::Minus => {
                                        self.ssao_settings.radius -= 1.0;
                                        println!("Radius: {}", self.ssao_settings.radius);
                                    }
                                    _ => {}
                                };
                            }
                        }
                    }
                    _ => {}
                };
            }
            _ => {}
        }
    }

    pub fn render(&mut self, frame: &TextureView) {
        //================== CAMERA DATA UPLOAD
        self.camera.update_gpu(&self.queue);

        for i in 0..self.covid_transforms.len() {
            let rotation = self.covid_transforms[i].fixed_slice::<U3, U3>(0, 0);
            let position = self.covid_transforms[i].column(3).xyz();
            let direction = normalize(&(self.camera.eye() - position));

            self.covid_pvs
                .compute_from_eye(&self.device, &self.queue, direction);
        }

        //================== RENDER MOLECULES
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[RenderPassColorAttachmentDescriptor {
                    attachment: &self.normals_texture,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::TRANSPARENT),
                        store: true,
                    },
                }],
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
            rpass.set_bind_group(0, &self.camera.bind_group(), &[]);

            for i in 0..self.covid_transforms.len() {
                rpass.set_bind_group(2, &self.covid_transforms_bgs[i], &[]);

                let rotation = self.covid_transforms[i].fixed_slice::<U3, U3>(0, 0);
                let position = self.covid_transforms[i].column(3).xyz();
                let direction = self.camera.eye() - position;

                if self.state.draw_lod {
                    if self.state.draw_occluded {
                        self.covid.draw_lod(&mut rpass, direction.magnitude());
                    } else {
                        self.covid_pvs
                            .draw_lod(&mut rpass, direction, direction.magnitude());
                    }
                } else {
                    if self.state.draw_occluded {
                        self.covid.draw(&mut rpass);
                    } else {
                        self.covid_pvs.draw(&mut rpass, direction);
                    }
                }
            }
        }

        self.ssao_module.draw(&self.device, &self.queue, &mut encoder, &self.ssao_settings, &self.depth_texture, &self.normals_texture);

        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[RenderPassColorAttachmentDescriptor {
                    attachment: &frame,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.output_pipeline);
            rpass.set_bind_group(0, &self.output_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
