use master_thesis::camera::*;
use master_thesis::framework;
use master_thesis::frustrum_culler::*;
use master_thesis::pipelines::SphereBillboardsPipeline;
use master_thesis::pvs::*;
use master_thesis::ssao;
use master_thesis::structure::*;

use bytemuck::cast_slice;
use futures::task::LocalSpawn;
use nalgebra_glm::*;
use rand::distributions::Distribution;
use wgpu::util::*;
use wgpu::*;

use std::rc::Rc;
use std::time::{Duration, Instant};

struct ApplicationState {
    pub draw_lod: bool,
    pub draw_occluded: bool,
    pub animating: bool,
}

pub struct Application {
    width: u32,
    height: u32,

    state: ApplicationState,
    start_time: Instant,

    depth_texture: TextureView,
    multisampled_texture: TextureView,
    normals_texture: TextureView,
    output_texture: TextureView,

    camera: RotationCamera,

    billboards_pipeline: SphereBillboardsPipeline,

    pvs_module: Rc<StructurePvsModule>,

    covid: Rc<Structure>,
    covid_pvs: StructurePvsField,
    covid_rotations: Vec<Mat4>,
    covid_translations: Vec<Mat4>,
    covid_transforms_gpu: Buffer,
    covid_transforms_bgs: Vec<BindGroup>,

    ssao_module: ssao::SsaoModule,
    ssao_finals: [TextureView; 2],

    output_pipeline: RenderPipeline,
    output_bind_group: BindGroup,
}

impl framework::ApplicationStructure for Application {
    fn required_features() -> wgpu::Features {
        wgpu::Features::PUSH_CONSTANTS
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_push_constant_size: 4,
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
            &reversed_infinite_perspective_rh_zo(
                sc_desc.width as f32 / sc_desc.height as f32,
                0.785398163,
                0.1,
            ),
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
        let covid_pvs =
            pvs_module.pvs_field(&device, &camera_bind_group_layout, covid.clone(), 5, 96);

        let n = 10;
        let p = 0.25;
        let n3 = n * n * n;
        let rand_distr = rand_distr::Binomial::new(1, p).unwrap();
        let mut covid_rotations = Vec::new();
        let mut covid_translations = Vec::new();

        let mut count = 0;
        for i in 0..n3 {
            if count == (n3 as f64 * p) as i32 {
                break;
            }

            let sample = rand_distr.sample(&mut rand::thread_rng());
            if sample == 0 {
                continue;
            }

            covid_rotations.push(rotation((count as f32).to_radians(), &vec3(0.0, 1.0, 0.0)));

            let position = vec3(
                ((i % n) - (n / 2)) as f32 * covid.bounding_radius() * 2.0,
                (((i / n) % n) - (n / 2)) as f32 * covid.bounding_radius() * 2.0,
                ((i / (n * n)) - (n / 2)) as f32 * covid.bounding_radius() * 2.0,
            );
            covid_translations.push(translation(&position));

            count += 1;
        }
        println!("Amount of structures: {}", covid_translations.len());

        let covid_transforms_gpu = {
            let mut raw: Vec<f32> = Vec::new();
            for (translation, rotation) in covid_translations.iter().zip(covid_rotations.iter()) {
                let transform: Mat4 = translation * rotation;
                raw.extend_from_slice(transform.as_slice());
                raw.extend_from_slice(&[0.0; 48]);
            }
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: cast_slice(&raw),
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            })
        };

        let mut covid_transforms_bgs = Vec::new();
        for i in 0..covid_translations.len() {
            let i = i as u64;
            covid_transforms_bgs.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &per_structure_bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &covid_transforms_gpu,
                        offset: i * 256,
                        size: std::num::NonZeroU64::new(256),
                    },
                }],
            }));
        }

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
        // let mut ssao_settings = ssao::Settings::default();
        // ssao_settings.radius = covid.bounding_radius() * 2.0;
        // ssao_settings.projection = camera.ubo().projection;
        // ssao_settings.horizonAngleThreshold = 0.05;
        // ssao_settings.blurPassCount = 8;
        // ssao_settings.sharpness = 0.05;

        let output_vs = device.create_shader_module(include_spirv!("passthrough.vert.spv"));
        let output_fs = device.create_shader_module(include_spirv!("passthrough.frag.spv"));

        let output_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStage::all(),
                        ty: BindingType::StorageTexture {
                            dimension: TextureViewDimension::D2,
                            format: TextureFormat::R32Float,
                            readonly: true,
                        },
                        count: None,
                    },
                ],
            });

        let output_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&output_bind_group_layout],
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

        let output_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &output_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&ssao_finals[0]),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&ssao_finals[1]),
                },
            ],
        });

        let state = ApplicationState {
            draw_lod: true,
            draw_occluded: false,
            animating: false,
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
            output_texture,

            camera,

            billboards_pipeline,

            pvs_module: pvs_module.clone(),

            covid: covid.clone(),
            covid_pvs,
            covid_rotations,
            covid_translations,
            covid_transforms_gpu,
            covid_transforms_bgs,

            ssao_module,
            ssao_finals,

            output_pipeline,
            output_bind_group,
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
                            // VirtualKeyCode::Add => {
                            //     self.ssao_settings.radius += 1.0;
                            //     println!("Radius: {}", self.ssao_settings.radius);
                            // }
                            // VirtualKeyCode::Minus => {
                            //     self.ssao_settings.radius -= 1.0;
                            //     println!("Radius: {}", self.ssao_settings.radius);
                            // }
                            VirtualKeyCode::A => {
                                self.state.animating = !self.state.animating;
                            }
                            _ => {}
                        };
                    }
                }
            }
            _ => {}
        };
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
        let time: u32 = unsafe { std::mem::transmute(time) };

        // Rotate the structure
        for r in self.covid_rotations.iter_mut() {
            *r = rotation(0.2f32.to_radians(), &vec3(0.0, 1.0, 0.0)) * (*r);
        }

        //================== DATA UPLOAD
        if self.state.animating {
            self.camera.set_distance(self.camera.distance() - 30.0);
        }
        self.camera.update_gpu(queue);

        let culler = FrustrumCuller::from_matrix(self.camera.ubo().projection_view);

        let mut covid_transforms_f32: Vec<f32> = Vec::new();
        for (translation, rotation) in self
            .covid_translations
            .iter()
            .zip(self.covid_rotations.iter())
        {
            let transform: Mat4 = translation * rotation;
            covid_transforms_f32.extend_from_slice(transform.as_slice());
            covid_transforms_f32.extend_from_slice(&[0.0; 48]);
        }

        queue.write_buffer(
            &self.covid_transforms_gpu,
            0,
            cast_slice(&covid_transforms_f32),
        );

        for i in 0..self.covid_rotations.len() {
            let rotation = self.covid_rotations[i].fixed_slice::<U3, U3>(0, 0);
            let position = self.covid_translations[i].column(3).xyz();
            let direction =
                rotation.try_inverse().unwrap() * normalize(&(self.camera.eye() - position));

            futures::executor::block_on(self.covid_pvs.compute_from_eye(device, queue, direction));
        }

        //================== RENDER MOLECULES
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[
                //     RenderPassColorAttachmentDescriptor {
                //     attachment: &self.normals_texture,
                //     resolve_target: None,
                //     ops: Operations {
                //         load: LoadOp::Clear(Color::TRANSPARENT),
                //         store: true,
                //     },
                // }
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
            rpass.set_push_constants(ShaderStage::VERTEX, 0, &[time]);
            rpass.set_bind_group(0, &self.camera.bind_group(), &[]);

            let mut culled = 0;
            for i in 0..self.covid_translations.len() {
                let rotation = self.covid_rotations[i].fixed_slice::<U3, U3>(0, 0);
                let position = self.covid_translations[i].column(3).xyz();

                if !culler.test_sphere(vec4(
                    position.x,
                    position.y,
                    position.z,
                    self.covid.bounding_radius(),
                )) {
                    culled += 1;
                    continue;
                }

                let direction = self.camera.eye() - position;
                let direction_norm_rot = rotation.try_inverse().unwrap() * normalize(&direction);

                rpass.set_bind_group(2, &self.covid_transforms_bgs[i], &[]);
                if self.state.draw_lod {
                    if self.state.draw_occluded
                        || direction.magnitude() < self.covid.bounding_radius() * 1.5
                    {
                        self.covid.draw_lod(&mut rpass, direction.magnitude());
                    } else {
                        self.covid_pvs.draw_lod(
                            &mut rpass,
                            direction_norm_rot,
                            direction.magnitude(),
                        );
                    }
                } else {
                    if self.state.draw_occluded
                        || direction.magnitude() < self.covid.bounding_radius() * 1.5
                    {
                        self.covid.draw(&mut rpass);
                    } else {
                        self.covid_pvs.draw(&mut rpass, direction_norm_rot);
                    }
                }
            }

            println!("Culled {}", culled);
        }

        queue.submit(Some(encoder.finish()));
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        let mut ssao_settings = ssao::Settings::default();
        ssao_settings.radius = self.covid.bounding_radius() * 2.0;
        ssao_settings.projection = self.camera.ubo().projection;
        ssao_settings.horizonAngleThreshold = 0.05;
        ssao_settings.blurPassCount = 8;
        ssao_settings.sharpness = 0.05;

        self.ssao_module.compute(
            device,
            queue,
            &mut encoder,
            &ssao_settings,
            &self.depth_texture,
            None, // Some(&self.normals_texture),
            &self.ssao_finals[0],
        );

        queue.submit(Some(encoder.finish()));
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        ssao_settings.radius = 16.0;
        ssao_settings.horizonAngleThreshold = 0.00;
        ssao_settings.blurPassCount = 8;
        ssao_settings.sharpness = 0.05;
        self.ssao_module.compute(
            device,
            queue,
            &mut encoder,
            &ssao_settings,
            &self.depth_texture,
            None, // Some(&self.normals_texture),
            &self.ssao_finals[1],
        );

        queue.submit(Some(encoder.finish()));
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
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
            rpass.set_bind_group(0, &self.output_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

fn main() {
    framework::run::<Application>("Occlusion");
}
