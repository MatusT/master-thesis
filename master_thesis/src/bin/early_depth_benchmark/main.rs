use master_thesis::camera::*;
use master_thesis::framework;

use bytemuck::cast_slice;
use futures::task::LocalSpawn;
use nalgebra_glm::*;
use wgpu::util::*;
use wgpu::*;

use std::collections::HashMap;

pub struct Application {
    width: u32,
    height: u32,

    depth_texture: TextureView,

    camera: RotationCamera,
    camera_bgl: BindGroupLayout,

    pipelines: Vec<RenderPipeline>,

    atoms_bgl: BindGroupLayout,
    atom_positions: Buffer,
    atom_positions_len: u32,

    early: bool,
    greater: bool,
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
                    dynamic: false,
                    min_binding_size: Some(CameraUbo::size()),
                },
                count: None,
            }],
        });

        let atoms_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Camera bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStage::all(),
                ty: BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: None,
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
            5.0,
            1.0,
        );

        // Pipelines
        let spheres_vs = device.create_shader_module(include_spirv!("pipelines/spheres.vert.spv"));
        let spheres_less_fs =
            device.create_shader_module(include_spirv!("pipelines/spheres_less.frag.spv"));
        let spheres_greater_fs =
            device.create_shader_module(include_spirv!("pipelines/spheres_greater.frag.spv"));
        let spheres_early_less_fs =
            device.create_shader_module(include_spirv!("pipelines/spheres_early_less.frag.spv"));
        let spheres_early_greater_fs =
            device.create_shader_module(include_spirv!("pipelines/spheres_early_greater.frag.spv"));

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_bgl, &atoms_bgl],
            push_constant_ranges: &[],
        });

        let mut pipelines = Vec::new();
        for fragment in [
            (&spheres_less_fs, CompareFunction::Greater),
            (&spheres_greater_fs, CompareFunction::Less),
            (&spheres_early_less_fs, CompareFunction::Greater),
            (&spheres_early_greater_fs, CompareFunction::Less),
        ]
        .iter()
        {
            pipelines.push(device.create_render_pipeline(&RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex_stage: ProgrammableStageDescriptor {
                    module: &spheres_vs,
                    entry_point: "main",
                },
                fragment_stage: Some(ProgrammableStageDescriptor {
                    module: fragment.0,
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
                depth_stencil_state: Some(DepthStencilStateDescriptor {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: fragment.1,
                    stencil: StencilStateDescriptor::default(),
                }),
                vertex_state: VertexStateDescriptor {
                    index_format: IndexFormat::Uint16,
                    vertex_buffers: &[VertexBufferDescriptor {
                        stride: 12,
                        step_mode: InputStepMode::Instance,
                        attributes: &vertex_attr_array![0 => Float3],
                    }],
                },
                sample_count,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            }));
        }

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

        // Data
        let atom_positions: Vec<f32> = vec![
            0.5, 0.5, 0.5, 1.0,
            0.0, 0.0, 0.0, 1.0,
            0.5, 0.5, -0.5, 1.0,
            -0.5, 0.5, 0.5, 1.0,
            0.5, -0.5, 0.5, 1.0,
        ];
        let atom_positions_len = (atom_positions.len() / 4) as u32;

        let atom_positions = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&atom_positions),
            usage: BufferUsage::STORAGE,
        });

        Self {
            width,
            height,

            depth_texture,

            camera,
            camera_bgl,

            atoms_bgl,

            pipelines,

            atom_positions,
            atom_positions_len,

            early: false,
            greater: false,
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
                            VirtualKeyCode::E => {
                                self.early = !self.early;
                                changed = true;
                            }
                            VirtualKeyCode::G => {
                                self.greater = !self.greater;
                                changed = true;
                            }
                            _ => {}
                        };
                    }
                }
            }
            _ => {}
        };

        if changed {
            if self.greater {
                self.camera.set_projection(&infinite_perspective_rh_zo(
                    self.width as f32 / self.height as f32,
                    0.785398163,
                    0.1,
                ));
            } else {
                self.camera.set_projection(&reversed_infinite_perspective_rh_zo(
                    self.width as f32 / self.height as f32,
                    0.785398163,
                    0.1,
                ));
            }

            let pipeline_index = 2 * (self.early as usize) + self.greater as usize;
            println!("Early: {:?} | Greater: {:?} | Index: {}", self.early, self.greater, pipeline_index);
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
        //================== DATA UPLOAD
        self.camera.update_gpu(queue);

        let atoms_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.atoms_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &self.atom_positions,
                        offset: 0,
                        size: None,
                    },
                },
            ],
        });

        //================== RENDER MOLECULES
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
                        load: LoadOp::Clear(if self.greater { 1.0 } else { 0.0 }),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            let pipeline_index = 2 * (self.early as usize) + self.greater as usize;
            rpass.set_pipeline(&self.pipelines[pipeline_index]);
            rpass.set_bind_group(0, &self.camera.bind_group(), &[]);
            rpass.set_bind_group(1, &atoms_bg, &[]);
            rpass.draw(
                0..self.atom_positions_len * 3,
                0..1,
            );
        }

        queue.submit(Some(encoder.finish()));
    }
}

fn main() {
    framework::run::<Application>("Occlusion");
}
