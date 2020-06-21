use crate::biological_structure::*;
use crate::camera::*;
use crate::hilbert;
use crate::pipelines::{LinesPipeline, SphereBillboardsDepthPipeline, SphereBillboardsPipeline};
use crate::ApplicationEvent;

use bytemuck::*;
use rpdb::*;
use std::convert::TryInto;
use wgpu::*;

pub fn list_to_ranges(list: &[u32]) -> Vec<(u32, u32)> {
    let mut ranges = Vec::new();

    let mut start = 0;
    for (index, value) in list.iter().enumerate() {
        if (index == list.len() - 1) || (*value != list[index + 1] - 1) {
            ranges.push((list[start], *value + 1));
            start = index + 1;

            if start >= list.len() - 1 {
                break;
            }
        }
    }

    ranges
}

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
    lines_pipeline: LinesPipeline,

    /// Array of molecules. Each element contains GPU buffer of atoms of a molecule.
    molecules: Vec<Buffer>,
    molecules_len: Vec<u32>,
    molecules_lods: Vec<Vec<(f32, std::ops::Range<u32>)>>,

    /// Array of structure's molecules. Each element contains GPU buffer of model matrices of one type of molecule.
    pub structure: Vec<Buffer>,
    pub structure_len: Vec<u32>,
    pub structure_bind_groups: Vec<BindGroup>,

    /// Occlusion data
    pub structure_visible: Vec<Buffer>,
    pub structure_visible_staging: Vec<Buffer>,
    pub structure_visible_cpu: Vec<Vec<u32>>,
    pub structure_visible_bind_groups: Vec<BindGroup>,
    pub structure_result: Vec<Vec<(u32, u32)>>,

    pub recalculate: bool,

    projected_lines: Buffer,
    projected_lines_len: u32,
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
                bindings: &[BindGroupLayoutEntry::new(
                    0,
                    ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                    BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                )],
            });

        let per_molecule_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Molecule bind group layout"),
                bindings: &[
                    BindGroupLayoutEntry::new(
                        0,
                        ShaderStage::VERTEX,
                        BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: None,
                        },
                    ),
                    BindGroupLayoutEntry::new(
                        1,
                        ShaderStage::VERTEX,
                        BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: None,
                        },
                    ),
                    BindGroupLayoutEntry::new(
                        2,
                        ShaderStage::FRAGMENT,
                        BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                            min_binding_size: None,
                        },
                    ),
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
        let lines_pipeline = LinesPipeline::new(&device, &camera_bind_group_layout, sample_count);

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
        let mut molecules_lods = Vec::new();

        let mut structure = Vec::new();
        let mut structure_len = Vec::new();
        let mut structure_bind_groups = Vec::new();

        let mut structure_visible = Vec::new();
        let mut structure_visible_staging = Vec::new();
        let mut structure_visible_cpu = Vec::new();
        let mut structure_visible_bind_groups = Vec::new();

        let mut projected_lines = Vec::new();
        // let mut hilbert_lines = Vec::new();

        for (molecule_name, molecule_model_matrices) in structure_file.molecules {
            let molecule = molecule::Molecule::from_ron(
                std::path::Path::new(&args[1]).with_file_name(molecule_name + ".ron"),
            );
            let mut molecule_lods: Vec<(f32, std::ops::Range<u32>)> = Vec::new();
            let mut atoms = Vec::new();
            let mut sum = 0u32;
            for lod in molecule.lods() {
                for atom in lod.atoms() {
                    atoms.extend_from_slice(&[atom.x, atom.y, atom.z, atom.w]);
                }
                molecule_lods.push((
                    lod.breakpoint(),
                    sum * 3..(sum + lod.atoms().len() as u32) * 3,
                ));
                sum += lod.atoms().len() as u32;
            }

            molecules
                .push(device.create_buffer_with_data(cast_slice(&atoms), BufferUsage::STORAGE));
            molecules_len.push((atoms.len() / 4) as u32 * 3);
            molecules_lods.push(molecule_lods);

            // Hilbert lines
            let hilbert = hilbert::sort_by_hilbert(&molecule_model_matrices);
            for d in hilbert.1[2].iter() {
                for m in d {
                    let v = m.column(3).xyz();
                    let xy = hilbert::intersect_inside_no(&v);
                    // let face = hilbert::vector_cube_face(&xy);
                    // let distance = length(&v);
                    // let xy = distance * xy;

                    // if face == hilbert::CubeFace::Back {
                    projected_lines.extend_from_slice(&[xy[0], xy[1], 0.0]);
                    projected_lines.push(v[0]);
                    projected_lines.push(v[1]);
                    projected_lines.push(v[2]);
                    // }
                }
            }

            let molecule_model_matrices = hilbert.0;
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
            structure_visible_cpu.push(vec![0; molecule_model_matrices.len() / 16]);
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
        let projected_lines_len = (projected_lines.len() / 6) as u32;
        let projected_lines =
            device.create_buffer_with_data(cast_slice(&projected_lines), BufferUsage::VERTEX);

        let structure_result = vec![Vec::new(); structure_len.len()];

        // Camera
        let mut camera = RotationCamera::new(width as f32 / height as f32, 0.785398163, 0.1);
        let camera_buffer = device.create_buffer_with_data(
            cast_slice(&[camera.ubo()]),
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        );
        camera.set_distance(2200.0);
        camera.set_speed(100.0);

        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &camera_bind_group_layout,
            bindings: &[Binding {
                binding: 0,
                resource: BindingResource::Buffer(camera_buffer.slice(0..!0)),
            }],
        });

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
            lines_pipeline,

            molecules,
            molecules_len,
            molecules_lods,

            structure,
            structure_len,
            structure_bind_groups,

            structure_visible,
            structure_visible_staging,
            structure_visible_cpu,
            structure_visible_bind_groups,
            structure_result,

            recalculate: true,

            projected_lines,
            projected_lines_len,
        }
    }

    pub fn resize(&mut self, _: u32, _: u32) {
        //
    }

    pub fn update<'a>(&mut self, event: &ApplicationEvent<'a>) {
        match event {
            ApplicationEvent::WindowEvent(event) => match event {
                winit::event::WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(keycode) = input.virtual_keycode {
                        if keycode == winit::event::VirtualKeyCode::Space {
                            self.recalculate = true;
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
        self.camera.update(event);
    }

    pub fn render(&mut self, frame: &TextureView) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        //================== CAMERA DATA UPLOAD
        {
            // self.camera.yaw += 1.0;
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

        //================== OCCLUSION
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
                        stencil_load_op: LoadOp::Load,
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
        }

        //================== RENDER MOLECULES
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
                    stencil_load_op: LoadOp::Load,
                    stencil_store_op: StoreOp::Store,
                    clear_depth: 0.0,
                    clear_stencil: 0,
                    stencil_read_only: true,
                    depth_read_only: false,
                }),
            });

            rpass.set_pipeline(&self.billboards_pipeline.pipeline);
            rpass.set_bind_group(0, &self.camera_bind_group, &[]);

            for molecule_id in 0..self.molecules.len() {
                rpass.set_bind_group(1, &self.structure_bind_groups[molecule_id], &[]);

                for range in self.structure_result[molecule_id].iter() {
                    rpass.draw(0..self.molecules_len[molecule_id], range.0..range.1);
                }
            }

            // //================== LINES
            // rpass.set_pipeline(&self.lines_pipeline.pipeline);
            // rpass.set_bind_group(0, &self.camera_bind_group, &[]);
            // rpass.set_vertex_buffer(0, self.projected_lines.slice(0..!0));
            // rpass.draw(0..self.projected_lines_len, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));

        //================== OCCLUSION DOWNLOAD
        if self.recalculate {
            for molecule_id in 0..self.molecules.len() {
                let buffer_slice = self.structure_visible_staging[molecule_id].slice(..);
                let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
                self.device.poll(Maintain::Wait);

                self.structure_visible_cpu[molecule_id] =
                    if let Ok(()) = futures::executor::block_on(buffer_future) {
                        let data = buffer_slice.get_mapped_range();
                        let result = data
                            .chunks_exact(4)
                            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                            .collect();

                        drop(data);
                        self.structure_visible_staging[molecule_id].unmap();

                        result
                    } else {
                        panic!("failed to run on gpu!")
                    };

                self.structure_visible_staging[molecule_id].unmap();

                let mut tmp = self.structure_visible_cpu[molecule_id]
                    .iter()
                    .enumerate()
                    .filter_map(|e| if *e.1 == 1 { Some(e.0 as u32) } else { None })
                    .collect::<Vec<u32>>();
                tmp.sort();
                self.structure_result[molecule_id] = list_to_ranges(&tmp);
            }

            self.structure_result =
                reduce_visible(&self.molecules_len, self.structure_result.clone(), 1024);

            for molecule_id in 0..self.molecules.len() {
                println!(
                    "Molecule #{} count {}/{}",
                    molecule_id,
                    self.structure_result[molecule_id]
                        .iter()
                        .map(|range| (range.1 - range.0) as usize)
                        .sum::<usize>(),
                    self.structure_len[molecule_id]
                );
            }

            self.recalculate = false;
        }

        println!("{}", self.camera.distance());
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
