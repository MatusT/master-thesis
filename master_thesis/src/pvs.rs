///!
///! Vec<Atoms>
///!      ^
///!      | has multiple
///! Vec<Molecules>
///!      ^
///!      | has multiple
///! Vec<Structure>
use bytemuck::cast_slice;
use nalgebra_glm::*;
use wgpu::util::*;
use wgpu::*;

use std::cell::RefCell;
use std::convert::TryInto;
use std::mem::size_of;
use std::rc::Rc;

use crate::camera::*;
use crate::pipelines::SphereBillboardsDepthPipeline;
use crate::structure::*;
use crate::*;

pub struct StructurePvsModule {
    ///
    depth: TextureView,

    ///
    pipeline: SphereBillboardsDepthPipeline,

    ///
    pipeline_write: SphereBillboardsDepthPipeline,

    bind_group_layout: BindGroupLayout,
}

impl StructurePvsModule {
    pub fn new(
        device: &Device,
        camera_bind_group_layout: &BindGroupLayout,
        per_molecule_bind_group_layout: &BindGroupLayout,
    ) -> Self {
        let depth = device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 512,
                    height: 512,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::RENDER_ATTACHMENT,
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let per_visibility_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Molecule visibility bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline = SphereBillboardsDepthPipeline::new(
            &device,
            camera_bind_group_layout,
            per_molecule_bind_group_layout,
            None,
            1,
            false,
        );

        let pipeline_write = SphereBillboardsDepthPipeline::new(
            &device,
            camera_bind_group_layout,
            per_molecule_bind_group_layout,
            Some(&per_visibility_bind_group_layout),
            1,
            true,
        );

        Self {
            depth,
            pipeline,
            pipeline_write,
            bind_group_layout: per_visibility_bind_group_layout,
        }
    }

    ///
    pub fn pvs_field(
        self: &Rc<StructurePvsModule>,
        device: &Device,
        camera_bind_group_layout: &BindGroupLayout,
        structure: Rc<RefCell<Structure>>,
        step: u32,
        ranges_limit: usize,
    ) -> StructurePvsField {
        let views_per_circle = 360 / step;
        let sets = vec![None; (views_per_circle * views_per_circle) as usize];

        let r = structure.borrow().bounding_radius();
        let projection = ortho_rh_zo(-r, r, -r, r, 0.0, 2.0 * r);
        let camera = RotationCamera::new(device, camera_bind_group_layout, &projection, -r, 0.0);

        let mut visible = Vec::new();
        let mut visible_staging = Vec::new();
        let mut visible_bind_groups = Vec::new();

        let mut max_size = 0;
        for i in 0..structure.borrow().molecules().len() {
            max_size = max_size.max(structure.borrow().transforms()[i].1);
            visible.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: cast_slice(&vec![0i32; structure.borrow().transforms()[i].1]),
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                }),
            );
            visible_staging.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (structure.borrow().transforms()[i].1 * size_of::<i32>()) as u64,
                usage: BufferUsage::MAP_READ | BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }));
            visible_bind_groups.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &visible.last().unwrap(),
                        offset: 0,
                        size: None,
                    },
                }],
            }));
        }

        let zero_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: cast_slice(&vec![0i32; max_size]),
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        });

        StructurePvsField {
            camera,

            module: Rc::clone(&self),
            structure: Rc::clone(&structure),

            sets,

            step,
            ranges_limit,

            visible,
            visible_staging,
            visible_bind_groups,

            zero_buffer,
        }
    }
}

/// `BiologicalStructure`'s field of all potentially visible sets at certain
pub struct StructurePvsField {
    ///
    camera: RotationCamera,

    ///
    module: Rc<StructurePvsModule>,

    ///
    structure: Rc<RefCell<Structure>>,

    ///
    pub sets: Vec<Option<StructurePvs>>,

    /// Step of distribution of discretized views. In degrees.
    step: u32,

    /// Upper bound of ranges to generate.
    ranges_limit: usize,

    ///
    visible: Vec<Buffer>,

    ///
    visible_staging: Vec<Buffer>,

    ///
    visible_bind_groups: Vec<BindGroup>,

    ///
    zero_buffer: Buffer,
}

impl StructurePvsField {
    pub fn spherical_to_snapped(&self, spherical_coords: TVec2<f64>) -> TVec2<u32> {
        // Convert spherical coordinates to degrees and modulo them into the range of 0-360.
        let snapped_coords =
            spherical_coords.apply_into(|e| ((e.to_degrees().round() % 360.0) + 360.0) % 360.0);

        // Snap spherical coordinates to the closest view given by step size.
        let snapped_coords = TVec2::new(
            (((snapped_coords[0] as u32 + self.step - 1) / self.step) * self.step) % 360,
            (((snapped_coords[1] as u32 + self.step - 1) / self.step) * self.step) % 360,
        );

        snapped_coords
    }

    pub fn spherical_to_index(&self, spherical_coords: TVec2<f64>) -> usize {
        self.snapped_to_index(self.spherical_to_snapped(spherical_coords))
    }

    pub fn snapped_to_spherical(&self, snapped_coords: TVec2<u32>) -> TVec2<f64> {
        // Conver to floating point radians.
        let spherical_coords = vec2(
            (snapped_coords.x as f64).to_radians(),
            (snapped_coords.y as f64).to_radians(),
        );

        spherical_coords
    }

    pub fn snapped_to_index(&self, snapped_coords: TVec2<u32>) -> usize {
        debug_assert!(snapped_coords.x < 360);
        debug_assert!(snapped_coords.y < 360);
        debug_assert!(snapped_coords.x % self.step == 0);
        debug_assert!(snapped_coords.y % self.step == 0);

        let steps = 360 / self.step;

        ((snapped_coords.x / self.step) * steps + (snapped_coords.y / self.step)) as usize
    }

    pub fn index_to_snapped(&self, index: usize) -> TVec2<u32> {
        let index = index as u32;
        let steps = 360 / self.step;

        vec2((index / steps) * self.step, (index % steps) * self.step)
    }

    pub fn index_to_spherical(&self, index: usize) -> TVec2<f64> {
        self.snapped_to_spherical(self.index_to_snapped(index))
    }

    pub async fn compute(&mut self, device: &Device, queue: &Queue, index: usize) -> bool {
        if self.sets[index].is_some() {
            return false;
        }

        {
            let structure = self.structure.borrow();
            let mut visible = Vec::new();

            // Configure camera
            let spherical_coords = self.index_to_spherical(index);
            self.camera.set_yaw(spherical_coords.x);
            self.camera.set_pitch(spherical_coords.y);

            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });

            self.camera.update_gpu(queue);

            // Clear the visibility data from the device buffers
            for molecule_id in 0..structure.molecules().len() {
                encoder.copy_buffer_to_buffer(
                    &self.zero_buffer,
                    0,
                    &self.visible[molecule_id],
                    0,
                    (structure.transforms()[molecule_id].1 * size_of::<i32>()) as BufferAddress,
                );
            }

            // Draw the depth buffer
            {
                let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("PVS computation: first depth pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                        attachment: &self.module.depth,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(0.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                });

                rpass.push_debug_group("Draw depth buffer");
                rpass.set_pipeline(&self.module.pipeline.pipeline);
                rpass.set_bind_group(0, self.camera.bind_group(), &[]);

                for molecule_id in 0..structure.molecules().len() {
                    rpass.set_bind_group(1, &structure.bind_groups()[molecule_id], &[]);

                    // rpass.draw(
                    //     structure.molecules()[molecule_id].lods().last().unwrap().1.start
                    //         ..structure.molecules()[molecule_id].lods().last().unwrap().1.end,
                    //     0..structure.transforms()[molecule_id].1 as u32,
                    // );

                    rpass.draw(
                        structure.molecules()[molecule_id].lods()[0].1.start
                            ..structure.molecules()[molecule_id].lods()[0].1.end,
                        0..structure.transforms()[molecule_id].1 as u32,
                    );
                }
                rpass.pop_debug_group();
            }

            // Draw a second time without writing to a depth buffer but writing visibility
            {
                let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("PVS computation: second depth pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                        attachment: &self.module.depth,
                        depth_ops: Some(Operations {
                            load: LoadOp::Load,
                            store: false,
                        }),
                        stencil_ops: None,
                    }),
                });

                rpass.push_debug_group("Draw write fragment depth buffer");
                rpass.set_pipeline(&self.module.pipeline_write.pipeline);
                rpass.set_bind_group(0, self.camera.bind_group(), &[]);

                for molecule_id in 0..structure.molecules().len() {
                    // rpass.set_push_constants(ShaderStage::VERTEX, 0, cast_slice(&[structure.molecules()[molecule_id].bounding_radius()]));
                    rpass.set_bind_group(1, &structure.bind_groups()[molecule_id], &[]);
                    rpass.set_bind_group(2, &self.visible_bind_groups[molecule_id], &[]);

                    // let lod = structure.molecules()[molecule_id].lods().last().unwrap();
                    let lod = &structure.molecules()[molecule_id].lods()[0];
                    rpass.draw(
                        lod.1.start..lod.1.end,
                        // 0..3,
                        0..structure.transforms()[molecule_id].1 as u32,
                    );
                }
                rpass.pop_debug_group();
            }

            // Download the visibility data from the device buffer to the staging one
            for molecule_id in 0..structure.molecules().len() {
                encoder.copy_buffer_to_buffer(
                    &self.visible[molecule_id],
                    0,
                    &self.visible_staging[molecule_id],
                    0,
                    (structure.transforms()[molecule_id].1 * size_of::<i32>()) as BufferAddress,
                );
            }

            queue.submit(Some(encoder.finish()));

            // Download the visibility data from the staging buffer to CPU
            for molecule_id in 0..structure.molecules().len() {
                let buffer_slice = self.visible_staging[molecule_id].slice(..);
                let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
                device.poll(Maintain::Wait);

                let visible_cpu: Vec<u32> = if let Ok(()) = buffer_future.await {
                    let data = buffer_slice.get_mapped_range();
                    let result = data
                        .chunks_exact(4)
                        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                        .collect();
                    drop(data);

                    result
                } else {
                    panic!("failed to run on gpu!")
                };

                self.visible_staging[molecule_id].unmap();

                let mut tmp = visible_cpu
                    .iter()
                    .enumerate()
                    .filter_map(|e| if *e.1 == 1 { Some(e.0 as u32) } else { None })
                    .collect::<Vec<u32>>();
                tmp.sort();

                visible.push(list_to_ranges(&tmp));
            }

            self.sets[index] = Some(StructurePvs { visible });
        }
        self.reduce(index);

        return true;
    }

    pub async fn compute_from_eye(
        &mut self,
        device: &Device,
        queue: &Queue,
        eye: TVec3<f64>,
    ) -> bool {
        let index = self.spherical_to_index(cartesian_to_spherical(&vec3(eye.x, eye.y, eye.z)));

        self.compute(device, queue, index).await
    }

    /// Computes potentially visible sets from all possible polar coordinates given by `step`.
    pub async fn compute_all(&mut self, device: &Device, queue: &Queue) {
        for index in 0..self.sets.len() {
            self.compute(device, queue, index).await;
        }
    }

    /// Returns potentially visible set from the given viewpoint, given by spherical coordinates in degrees of `step` multiple.
    pub fn get(&self, index: usize) -> Option<&StructurePvs> {
        self.sets[index].as_ref()
    }

    /// Returns potentially visible set from the given viewpoint.
    ///
    /// # Arguments
    ///
    /// * `eye` - vector **to** viewing point. Must be in the local coordinate system of the structure. If the structure is rotated, so must be the vector.
    ///
    pub fn get_from_eye(&self, eye: Vec3) -> Option<&StructurePvs> {
        let index = self.spherical_to_index(cartesian_to_spherical(&vec3(
            eye.x as f64,
            eye.y as f64,
            eye.z as f64,
        )));

        self.get(index)
    }

    pub fn reduce(&mut self, index: usize) {
        let structure = self.structure.borrow();
        let pvs = self.sets[index].as_mut().unwrap();

        let mut gaps = vec![Vec::new(); pvs.visible.len()];

        // Find the gaps
        for (molecule_index, ranges) in pvs.visible.iter().enumerate() {
            'ranges: for range_index in 1..ranges.len() {
                let distance = ranges[range_index].0 - ranges[range_index - 1].1;

                // If we find a gap
                if distance > 0 && structure.molecules()[molecule_index].name() != "S" {
                    // Check that it doesn't cross faces
                    // - potentially large range
                    for face in structure.transforms_sides().unwrap()[molecule_index].iter() {
                        if *face >= ranges[range_index - 1].1 && *face <= ranges[range_index].0 {
                            continue 'ranges;
                        }
                    }
                    gaps[molecule_index].push((ranges[range_index - 1].1, ranges[range_index].0));
                }
            }

            // Sort the gaps by their length in decreasing order
            gaps[molecule_index].sort_by(|a, b| {
                let a_distance = a.1 - a.0;
                let b_distance = b.1 - b.1;

                b_distance.cmp(&a_distance)
            });
        }

        // Compute how many ranges we have in total across all molecule types
        let mut ranges_num: usize = pvs.visible.iter().map(|v| v.len()).sum();

        // While we are not under the imposed limit
        while ranges_num > self.ranges_limit {
            // Run through gaps of each molecule and find the gap with smallest cost (distance * number of atoms)
            let mut min_gap = None;
            let mut min_cost = u32::MAX;
            let mut min_index = 0;
            for (molecule_index, gaps) in gaps.iter().enumerate() {
                if let Some(gap) = gaps.last() {
                    let atoms_count = (structure.molecules()[molecule_index].lods()[0].1.end
                        - structure.molecules()[molecule_index].lods()[0].1.start)
                        / 3;
                    let distance = gap.1 - gap.0;

                    let cost = distance * atoms_count;

                    if cost < min_cost {
                        min_gap = Some(gap);
                        min_cost = cost;
                        min_index = molecule_index;
                    }
                }
            }

            // Add the found smallest gap to the list of ranges
            if let Some(gap) = min_gap {
                pvs.visible[min_index].push(*gap);
                gaps[min_index].pop();
                ranges_num -= 1;
            } else {
                break;
            }
        }

        // To reduce the list, the ranges are sorted and merged together
        for i in 0..self.visible.len() {
            pvs.visible[i].sort_by(|a, b| a.0.cmp(&b.0));
            pvs.visible[i] = compress_ranges(pvs.visible[i].clone(), 0);
        }
    }

    pub fn ranges_limit(&self) -> usize {
        self.ranges_limit
    }
}
/// One potentially visible set of field of them.
#[derive(Clone)]
pub struct StructurePvs {
    /// Ranges
    pub visible: Vec<Vec<(u32, u32)>>,
}

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

pub fn compress_ranges(list: Vec<(u32, u32)>, threshold: u32) -> Vec<(u32, u32)> {
    if list.is_empty() {
        return Vec::new();
    }

    let mut new_list = Vec::new();

    let mut start = list[0].0;
    for (index, range) in list.iter().enumerate() {
        if index >= list.len() - 1 {
            new_list.push((start, range.1));
            break;
        }

        if list[index + 1].0 - range.1 > threshold {
            new_list.push((start, range.1));
            start = list[index + 1].0;
        }
    }

    new_list
}
