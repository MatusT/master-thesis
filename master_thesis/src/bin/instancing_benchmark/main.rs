use master_thesis::camera::*;
use master_thesis::framework;

use bytemuck::cast_slice;
use futures::task::LocalSpawn;
use nalgebra_glm::*;
use rand::distributions::Distribution;
use wgpu::util::*;
use wgpu::*;

use std::rc::Rc;

pub struct Application {
    width: u32,
    height: u32,

    depth_texture: TextureView,

    camera: RotationCamera,

    camera_bgl: BindGroupLayout,
    manual_instanced_bgl: BindGroupLayout,

    instanced_pipeline: RenderPipeline,
    manual_instanced_pipeline: RenderPipeline,

    atom_positions: Buffer,
    atom_positions_instanced: Buffer,
    atom_positions_len: u32,
    molecules_positions: Buffer,
    molecules_positions_len: u32,

    instanced: bool,
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

        // Camera
        let camera_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Camera bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStage::all(),
                ty: BindingType::UniformBuffer {
                    has_dynamic_offset: false,
                    min_binding_size: Some(CameraUbo::size()),
                },
                count: None,
            }],
        });

        let mut camera = RotationCamera::new(
            &device,
            &camera_bgl,
            &reversed_infinite_perspective_rh_zo(
                sc_desc.width as f32 / sc_desc.height as f32,
                0.785398163,
                0.1,
            ),
            200.0,
            10.0,
        );

        // Pipelines
        let vs_instanced =
            device.create_shader_module(&include_spirv!("pipelines/spheres_instanced.vert.spv"));
        let vs_manual_instanced = device.create_shader_module(&include_spirv!(
            "pipelines/spheres_manual_instanced.vert.spv"
        ));
        let fs = device.create_shader_module(&include_spirv!("pipelines/spheres.frag.spv"));

        let instanced_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });

        let instanced_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&instanced_pipeline_layout),
            vertex_stage: ProgrammableStageDescriptor {
                module: &vs_instanced,
                entry_point: "main",
            },
            fragment_stage: Some(ProgrammableStageDescriptor {
                module: &fs,
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
            depth_stencil_state: Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: StencilStateDescriptor::default(),
            }),
            vertex_state: VertexStateDescriptor {
                index_format: Some(IndexFormat::Uint32),
                vertex_buffers: &[
                    VertexBufferDescriptor {
                        stride: 16,
                        step_mode: InputStepMode::Instance,
                        attributes: &vertex_attr_array![0 => Float4],
                    },
                    VertexBufferDescriptor {
                        stride: 16,
                        step_mode: InputStepMode::Vertex,
                        attributes: &vertex_attr_array![1 => Float4],
                    },
                ],
            },
            sample_count,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let manual_instanced_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Positions bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::VERTEX,
                    ty: BindingType::UniformBuffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::VERTEX,
                    ty: BindingType::StorageBuffer {
                        has_dynamic_offset: false,
                        access: StorageTextureAccess::ReadOnly,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let manual_instanced_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&camera_bgl, &manual_instanced_bgl],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStage::VERTEX,
                    range: 0..4,
                }],
            });

        let manual_instanced_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&manual_instanced_pipeline_layout),
            vertex_stage: ProgrammableStageDescriptor {
                module: &vs_manual_instanced,
                entry_point: "main",
            },
            fragment_stage: Some(ProgrammableStageDescriptor {
                module: &fs,
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
            depth_stencil_state: Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: StencilStateDescriptor::default(),
            }),
            vertex_state: VertexStateDescriptor {
                index_format: Some(IndexFormat::Uint32),
                vertex_buffers: &[],
            },
            sample_count,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

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
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Data
        let atom_positions = vec![0.0, 0.0, 0.0, 0.0];
        let atom_positions_len = (atom_positions.len() / 4) as u32;

        let atom_positions_instanced =
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let atom_positions_instanced = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&atom_positions_instanced),
            usage: BufferUsage::VERTEX,
        });
        let atom_positions = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&atom_positions),
            usage: BufferUsage::UNIFORM | BufferUsage::VERTEX,
        });

        let n = 200;
        let mut molecules_positions: Vec<f32> = Vec::new();
        for x in -n..n {
            for y in -n..n {
                for z in -n..n {
                    molecules_positions.push(x as f32 * 4.0);
                    molecules_positions.push(y as f32 * 4.0);
                    molecules_positions.push(z as f32 * 4.0);
                    molecules_positions.push(1.0);
                }
            }
        }
        let molecules_positions_len = (molecules_positions.len() / 4) as u32;

        let molecules_positions = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&molecules_positions),
            usage: BufferUsage::STORAGE | BufferUsage::VERTEX,
        });

        Self {
            width,
            height,

            depth_texture,

            camera,

            camera_bgl,
            manual_instanced_bgl,

            instanced_pipeline,
            manual_instanced_pipeline,

            atom_positions,
            atom_positions_instanced,
            atom_positions_len,
            molecules_positions,
            molecules_positions_len,

            instanced: false,
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
                            VirtualKeyCode::I => {
                                self.instanced = !self.instanced;
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
        //================== DATA UPLOAD
        self.camera.update_gpu(queue);

        //================== RENDER MOLECULES
        let positions_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.manual_instanced_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &self.atom_positions,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer {
                        buffer: &self.molecules_positions,
                        offset: 0,
                        size: None,
                    },
                },
            ],
        });

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
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            if self.instanced {
                rpass.set_pipeline(&self.instanced_pipeline);
                rpass.set_bind_group(0, &self.camera.bind_group(), &[]);
                rpass.set_vertex_buffer(0, self.molecules_positions.slice(..));
                rpass.set_vertex_buffer(1, self.atom_positions.slice(..));
                rpass.draw(
                    0..self.atom_positions_len * 3,
                    0..self.molecules_positions_len as u32,
                );
            } else {
                rpass.set_pipeline(&self.manual_instanced_pipeline);
                rpass.set_bind_group(0, &self.camera.bind_group(), &[]);
                rpass.set_bind_group(1, &positions_bind_group, &[]);
                rpass.set_push_constants(ShaderStage::VERTEX, 0, &[self.atom_positions_len]);
                rpass.draw(
                    0..(self.molecules_positions_len * self.atom_positions_len) * 3,
                    0..1,
                );
            }
        }

        queue.submit(Some(encoder.finish()));
    }
}

fn main() {
    framework::run::<Application>("Occlusion");
}
