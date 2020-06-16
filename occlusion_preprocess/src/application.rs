use crate::camera::*;
use crate::pipelines::{SphereBillboardsDepthPipeline, SphereBillboardsPipeline};
use crate::ApplicationEvent;

use std::convert::TryInto;
use bytemuck::*;
use rpdb::*;
use wgpu::*;
use futures::executor::ThreadPool;
pub struct Application {
    width: u32,
    height: u32,

    device: Device,
    queue: Queue,

    depth_texture: TextureView,
    multisampled_texture: TextureView,

    camera: RotationCamera,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,

    billboards_pipeline: SphereBillboardsPipeline,
    billboards_depth_pipeline: SphereBillboardsDepthPipeline,
    billboards_depth_write_pipeline: SphereBillboardsDepthPipeline,

    /// Array of molecules. Each element contains GPU buffer of atoms of a molecule.
    molecules: Vec<Buffer>,
    molecules_len: Vec<u32>,

    /// Array of structure's molecules. Each element contains GPU buffer of model matrices of one type of molecule.
    pub structure: Vec<Buffer>,
    pub structure_len: Vec<u32>,
    pub structure_bind_groups: Vec<BindGroup>,

    /// Occlusion data
    pub structure_visible: Vec<Buffer>,
    pub structure_visible_staging: Vec<Buffer>,
    pub structure_visible_cpu: Vec<Vec<i32>>,
    pub structure_visible_bind_groups: Vec<BindGroup>,

    pub recalculate: bool,

    threadpool: ThreadPool,
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
        let camera_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Camera bind group layout"),
                bindings: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                    ty: BindingType::UniformBuffer { dynamic: false },
                    ..Default::default()
                }],
            });

        let per_molecule_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Molecule bind group layout"),
                bindings: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStage::VERTEX,
                        ty: BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                        },
                        ..Default::default()
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStage::VERTEX,
                        ty: BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                        },
                        ..Default::default()
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStage::FRAGMENT,
                        ty: BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                        },
                        ..Default::default()
                    },
                ],
            });

        // Pipelines
        let billboards_pipeline =
            SphereBillboardsPipeline::new(&device, &camera_bind_group_layout, sample_count);
        let billboards_depth_pipeline = SphereBillboardsDepthPipeline::new(
            &device,
            &camera_bind_group_layout,
            &per_molecule_bind_group_layout,
            sample_count,
            false,
        );
        let billboards_depth_write_pipeline = SphereBillboardsDepthPipeline::new(
            &device,
            &camera_bind_group_layout,
            &per_molecule_bind_group_layout,
            sample_count,
            true,
        );

        // Default framebuffer
        let depth_texture = device.create_texture(&TextureDescriptor {
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
        });
        let depth_texture = depth_texture.create_default_view();

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

        // Data
        let args: Vec<String> = std::env::args().collect();

        let structure_file = structure::Structure::from_ron(&args[1]);

        let mut molecules = Vec::new();
        let mut molecules_len = Vec::new();
        let mut structure = Vec::new();
        let mut structure_len = Vec::new();
        let mut structure_bind_groups = Vec::new();

        let mut structure_visible = Vec::new();
        let mut structure_visible_staging = Vec::new();
        let mut structure_visible_cpu = Vec::new();
        let mut structure_visible_bind_groups = Vec::new();

        for (molecule_name, molecule_model_matrices) in structure_file.molecules {
            let molecule = molecule::Molecule::from_ron(
                std::path::Path::new(&args[1]).with_file_name(molecule_name + ".ron"),
            );
            let atoms = {
                let atoms = molecule.lods()[0].atoms();
                let mut atoms_flat: Vec<f32> = Vec::new();
                for atom in atoms {
                    atoms_flat.extend_from_slice(atom.into());
                }

                atoms_flat
            };
            molecules
                .push(device.create_buffer_with_data(cast_slice(&atoms), BufferUsage::STORAGE));
            molecules_len.push((atoms.len() / 4) as u32 * 3);

            let molecule_model_matrices = {
                let mut matrices_flat: Vec<f32> = Vec::new();
                for molecule_model_matrix in molecule_model_matrices {
                    for i in 0..16 {
                        matrices_flat.push(molecule_model_matrix[i]);
                    }
                }

                matrices_flat
            };
            structure.push(device.create_buffer_with_data(
                cast_slice(&molecule_model_matrices),
                BufferUsage::STORAGE,
            ));
            structure_len.push(molecule_model_matrices.len() as u32 / 16);
            structure_bind_groups.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &billboards_pipeline.per_molecule_bind_group_layout,
                bindings: &[
                    Binding {
                        binding: 0,
                        resource: BindingResource::Buffer(molecules.last().unwrap().slice(0..!0)),
                    },
                    Binding {
                        binding: 1,
                        resource: BindingResource::Buffer(structure.last().unwrap().slice(0..!0)),
                    },
                ],
            }));

            structure_visible.push(device.create_buffer_with_data(
                cast_slice(&vec![0i32; molecule_model_matrices.len() / 16]),
                BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            ));
            structure_visible_staging.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (molecule_model_matrices.len() / 16) as u64
                    * std::mem::size_of::<i32>() as u64,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }));
            structure_visible_cpu.push(vec![0i32; molecule_model_matrices.len() / 16]);
            structure_visible_bind_groups.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &per_molecule_bind_group_layout,
                bindings: &[
                    Binding {
                        binding: 0,
                        resource: BindingResource::Buffer(molecules.last().unwrap().slice(0..!0)),
                    },
                    Binding {
                        binding: 1,
                        resource: BindingResource::Buffer(structure.last().unwrap().slice(0..!0)),
                    },
                    Binding {
                        binding: 2,
                        resource: BindingResource::Buffer(
                            structure_visible.last().unwrap().slice(0..!0),
                        ),
                    },
                ],
            }));
        }

        // Camera
        let mut camera = RotationCamera::new(width as f32 / height as f32, 0.785398163, 0.1);
        let camera_buffer = device.create_buffer_with_data(
            cast_slice(&[camera.ubo()]),
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        );
        camera.set_distance(3000.0);
        camera.set_speed(100.0);

        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &camera_bind_group_layout,
            bindings: &[Binding {
                binding: 0,
                resource: BindingResource::Buffer(camera_buffer.slice(0..!0)),
            }],
        });

        let threadpool = ThreadPool::new().unwrap();

        Self {
            width,
            height,

            device,
            queue,

            depth_texture,
            multisampled_texture,

            camera,
            camera_buffer,
            camera_bind_group,

            billboards_pipeline,
            billboards_depth_pipeline,
            billboards_depth_write_pipeline,

            molecules,
            molecules_len,

            structure,
            structure_len,
            structure_bind_groups,

            structure_visible,
            structure_visible_staging,
            structure_visible_cpu,
            structure_visible_bind_groups,

            recalculate: true,
            threadpool,
        }
    }

    pub fn resize(&mut self, _: u32, _: u32) {
        //
    }

    pub fn update<'a>(&mut self, event: &ApplicationEvent<'a>) {
        self.camera.update(event);
    }

    pub fn render(&mut self, frame: &TextureView) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
            let size = std::mem::size_of::<CameraUbo>();
            let camera_buffer = self
                .device
                .create_buffer_with_data(cast_slice(&[self.camera.ubo()]), BufferUsage::COPY_SRC);

            encoder.copy_buffer_to_buffer(
                &camera_buffer,
                0,
                &self.camera_buffer,
                0,
                size as BufferAddress,
            );
        }

        if self.recalculate {
            {
                let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                        attachment: &self.depth_texture,
                        depth_load_op: LoadOp::Clear,
                        depth_store_op: StoreOp::Store,
                        stencil_load_op: LoadOp::Clear,
                        stencil_store_op: StoreOp::Store,
                        clear_depth: 0.0,
                        clear_stencil: 0,
                        stencil_read_only: true,
                        depth_read_only: false,
                    }),
                });

                rpass.set_pipeline(&self.billboards_depth_pipeline.pipeline);
                rpass.set_bind_group(0, &self.camera_bind_group, &[]);

                for molecule_id in 0..self.molecules.len() {
                    rpass.set_bind_group(1, &self.structure_visible_bind_groups[molecule_id], &[]);

                    rpass.draw(
                        0..self.molecules_len[molecule_id],
                        0..self.structure_len[molecule_id],
                    );
                }
            }

            {
                let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                        attachment: &self.depth_texture,
                        depth_load_op: LoadOp::Load,
                        depth_store_op: StoreOp::Store,
                        stencil_load_op: LoadOp::Clear,
                        stencil_store_op: StoreOp::Store,
                        clear_depth: 0.0,
                        clear_stencil: 0,
                        stencil_read_only: true,
                        depth_read_only: true,
                    }),
                });

                rpass.set_pipeline(&self.billboards_depth_write_pipeline.pipeline);
                rpass.set_bind_group(0, &self.camera_bind_group, &[]);

                for molecule_id in 0..self.molecules.len() {
                    rpass.set_bind_group(1, &self.structure_visible_bind_groups[molecule_id], &[]);

                    rpass.draw(
                        0..self.molecules_len[molecule_id],
                        0..self.structure_len[molecule_id],
                    );
                }
            }

            for molecule_id in 0..self.molecules.len() {
                encoder.copy_buffer_to_buffer(
                    &self.structure_visible[molecule_id],
                    0,
                    &self.structure_visible_staging[molecule_id],
                    0,
                    (self.structure_len[molecule_id] * std::mem::size_of::<i32>() as u32)
                        as BufferAddress,
                );
            }

            self.recalculate = false;
        }

        /*
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[RenderPassColorAttachmentDescriptor {
                    attachment: &frame,
                    resolve_target: None,
                    // attachment: &self.multisampled_texture,
                    // resolve_target: Some(&frame),
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: Color::WHITE,
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture,
                    depth_load_op: LoadOp::Clear,
                    depth_store_op: StoreOp::Store,
                    stencil_load_op: LoadOp::Clear,
                    stencil_store_op: StoreOp::Store,
                    clear_depth: 0.0,
                    clear_stencil: 0,
                    stencil_read_only: true,
                    depth_read_only: false,
                }),
            });

            rpass.set_pipeline(&self.billboards_pipeline.pipeline);
            rpass.set_bind_group(0, &self.camera_bind_group, &[]);

            for (molecule_id, molecule) in self.molecules.iter().enumerate() {
                rpass.set_bind_group(1, &self.structure_bind_groups[molecule_id], &[]);

                rpass.draw(
                    0..self.molecules_len[molecule_id],
                    0..self.structure_len[molecule_id],
                );
            }
        }
        */

        self.queue.submit(Some(encoder.finish()));

        for molecule_id in 0..1 {
            let buffer_slice = self.structure_visible_staging[molecule_id].slice(0..4);
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

            // Poll the device in a blocking manner so that our future resolves.
            // In an actual application, `device.poll(...)` should
            // be called in an event loop or on another thread.
            self.device.poll(Maintain::Wait);
            
            let visible: Vec<i32> = if let Ok(()) = futures::executor::block_on(buffer_future) {
                let data = buffer_slice.get_mapped_range();
                let result = data
                    .chunks_exact(4)
                    .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();

                // With the current interface, we have to make sure all mapped views are
                // dropped before we unmap the buffer.
                drop(data);
                self.structure_visible_staging[molecule_id].unmap();

                result
            } else {
                panic!("failed to run on gpu!")
            };
            
            self.structure_visible_staging[molecule_id].unmap();
        }

        println!("Frame done");
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}