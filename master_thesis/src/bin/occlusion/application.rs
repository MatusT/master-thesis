use master_thesis::camera::*;
use master_thesis::pipelines::SphereBillboardsPipeline;
use master_thesis::pvs::*;
use master_thesis::structure::*;
use master_thesis::ApplicationEvent;

use bytemuck::cast_slice;
use nalgebra_glm::*;
use wgpu::*;

use std::borrow::Cow::Borrowed;
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

    camera: RotationCamera,

    billboards_pipeline: SphereBillboardsPipeline,

    pvs_module: Rc<StructurePvsModule>,

    covid: Rc<Structure>,
    covid_pvs: StructurePvsField,
    covid_transforms: Vec<Mat4>,
    covid_transforms_gpu: Buffer,
    covid_transforms_bgs: Vec<BindGroup>,
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
                label: Some(Borrowed("Camera bind group layout")),
                entries: Borrowed(&[BindGroupLayoutEntry::new(
                    0,
                    ShaderStage::all(),
                    BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: Some(CameraUbo::size()),
                    },
                )]),
            });

        let per_molecule_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(Borrowed("Molecule bind group layout")),
                entries: Borrowed(&[
                    BindGroupLayoutEntry::new(
                        0,
                        ShaderStage::all(),
                        BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: None,
                        },
                    ),
                    BindGroupLayoutEntry::new(
                        1,
                        ShaderStage::all(),
                        BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: None,
                        },
                    ),
                ]),
            });
        let per_structure_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(Borrowed("Structure bind group layout")),
                entries: Borrowed(&[BindGroupLayoutEntry::new(
                    0,
                    ShaderStage::VERTEX,
                    BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                )]),
            });

        // Camera
        let camera = RotationCamera::new(
            &device,
            &camera_bind_group_layout,
            &reversed_infinite_perspective_rh_zo(width as f32 / height as f32, 0.785398163, 0.1),
            1500.0,
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
                usage: TextureUsage::OUTPUT_ATTACHMENT,
            })
            .create_default_view();

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
            .create_default_view();

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
            device.create_buffer_with_data(cast_slice(&raw), BufferUsage::UNIFORM)
        };
        let mut covid_transforms_bgs = Vec::new();
        for i in 0..covid_transforms.len() {
            let i = i as u64;
            covid_transforms_bgs.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &per_structure_bind_group_layout,
                entries: Borrowed(&[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(
                        covid_transforms_gpu.slice(i * 256..(i + 1) * 256),
                    ),
                }]),
            }));
        }

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

            camera,

            billboards_pipeline,

            pvs_module: pvs_module.clone(),

            covid: covid.clone(),
            covid_pvs,
            covid_transforms,
            covid_transforms_gpu,
            covid_transforms_bgs,
        }
    }

    pub fn resize(&mut self, _: u32, _: u32) {
        //
    }

    pub fn update<'a>(&mut self, event: &ApplicationEvent<'a>) {
        use winit::event::VirtualKeyCode;
        use winit::event::WindowEvent::*;
        use winit::event::ElementState;

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
                color_attachments: Borrowed(&[RenderPassColorAttachmentDescriptor {
                    attachment: &frame,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::WHITE),
                        store: true,
                    },
                }]),
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

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
