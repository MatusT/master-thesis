///!
///! Vec<Atoms>
///!      ^
///!      | has multiple
///! Vec<Molecules>
///!      ^
///!      | has multiple
///! Vec<Structure>
use bytemuck::cast_slice;
use nalgebra_glm::{distance, length, normalize, ortho_rh_zo, vec2, vec3, TVec2, Vec2, Vec3};
use rpdb;
use rpdb::BoundingBox;
use rpdb::FromRon;
use wgpu::*;

use std::convert::TryInto;
use std::mem::size_of;
use std::rc::Rc;

use crate::camera::*;
use crate::hilbert;
use crate::pipelines::SphereBillboardsDepthPipeline;

/// GPU represantion of a molecule for visualization.
pub struct Molecule {
    /// Buffer containing atoms/spheres of molecule and Its level of detail representations.
    atoms: Buffer,

    /// View ranges into `atoms` buffer Level of detail of molecule calculated using k-means algorithm.
    /// Level 0 is the original molecule. Tuple contains breakpoint of distance from the camera when to
    /// apply the LOD, and range view into the `atoms` buffer where the corresponding atoms/spheres of the LOD reside.
    lods: Vec<(f32, std::ops::Range<u32>)>,

    /// Bounding box encompassing the molecule. Includes radii of atoms.
    bounding_box: BoundingBox,

    ///
    bounding_radius: f32,
}

impl Molecule {
    pub fn from_ron<P: AsRef<std::path::Path>>(device: &Device, path: P) -> Self {
        let molecule = rpdb::molecule::Molecule::from_ron(path);

        let mut lods: Vec<(f32, std::ops::Range<u32>)> = Vec::new();
        let mut atoms = Vec::new();

        let mut sum = 0u32;
        for lod in molecule.lods() {
            for atom in lod.atoms() {
                atoms.extend_from_slice(&[atom.x, atom.y, atom.z, atom.w]);
            }
            lods.push((
                lod.breakpoint(),
                sum * 3..(sum + lod.atoms().len() as u32) * 3,
            ));
            sum += lod.atoms().len() as u32;
        }

        let atoms = device.create_buffer_with_data(cast_slice(&atoms), BufferUsage::STORAGE);

        let bounding_box = *molecule.bounding_box();
        let bounding_radius = distance(&bounding_box.max, &bounding_box.min) / 2.0;

        Self {
            atoms,
            lods,
            bounding_box,
            bounding_radius,
        }
    }

    pub fn atoms(&self) -> &Buffer {
        &self.atoms
    }

    pub fn lods(&self) -> &[(f32, std::ops::Range<u32>)] {
        &self.lods
    }

    pub fn bounding_box(&self) -> BoundingBox {
        self.bounding_box
    }
}

/// Biological structure, like SARS-Cov-19
pub struct Structure {
    /// Molecule of the biological structure.
    molecules: Vec<Molecule>,

    /// Transforms (Rotation, Translation) of molecules on a GPU.
    transforms: Vec<(Buffer, usize)>,

    /// For globular structures a split of translations of transformations into 6 faces of spherified cube.
    transforms_sides: Option<Vec<[u32; 6]>>,

    /// Bind groups for each molecule type containing reference to `molecules` and `transforms`.
    bind_groups: Vec<BindGroup>,

    /// Bounding box of the entire structure. Creates as an intersection of all bounding boxes of structure's molecules.
    bounding_box: BoundingBox,

    ///
    bounding_radius: f32,
}

impl Structure {
    pub fn from_ron<P: AsRef<std::path::Path>>(
        device: &Device,
        path: P,
        per_molecule_bind_group_layout: &BindGroupLayout,
    ) -> Self {
        let structure_file = rpdb::structure::Structure::from_ron(&path);

        let mut molecules = Vec::new();

        let mut transforms = Vec::new();
        let mut transforms_sides = Vec::new();

        let mut bind_groups = Vec::new();

        let mut bounding_radius: f32 = 0.0;

        for (molecule_name, molecule_model_matrices) in structure_file.molecules {
            let new_molecule =
                Molecule::from_ron(device, path.as_ref().with_file_name(molecule_name + ".ron"));

            let hilbert = hilbert::sort_by_hilbert(&molecule_model_matrices);
            let molecule_model_matrices = hilbert.0;
            let molecule_model_matrices_len = molecule_model_matrices.len();
            let molecule_model_matrices = {
                let mut matrices_flat: Vec<f32> = Vec::new();
                for molecule_model_matrix in molecule_model_matrices {
                    bounding_radius = bounding_radius.max(
                        length(&molecule_model_matrix.column(3).xyz())
                            + new_molecule.bounding_radius,
                    );
                    matrices_flat.append(&mut molecule_model_matrix.as_slice().to_owned());
                }

                matrices_flat
            };

            transforms.push((
                device.create_buffer_with_data(
                    cast_slice(&molecule_model_matrices),
                    BufferUsage::STORAGE,
                ),
                molecule_model_matrices_len,
            ));

            transforms_sides.push(hilbert.2);

            bind_groups.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &per_molecule_bind_group_layout,
                bindings: &[
                    Binding {
                        binding: 0,
                        resource: BindingResource::Buffer(new_molecule.atoms().slice(..)),
                    },
                    Binding {
                        binding: 1,
                        resource: BindingResource::Buffer(transforms.last().unwrap().0.slice(..)),
                    },
                ],
            }));

            molecules.push(new_molecule);
        }

        let bounding_box = BoundingBox {
            min: vec3(-bounding_radius, -bounding_radius, -bounding_radius),
            max: vec3(bounding_radius, bounding_radius, bounding_radius),
        };

        Self {
            molecules,
            transforms,
            transforms_sides: Some(transforms_sides),
            bind_groups,
            bounding_box,
            bounding_radius,
        }
    }

    pub fn molecules(&self) -> &[Molecule] {
        &self.molecules
    }

    pub fn transforms(&self) -> &[(Buffer, usize)] {
        &self.transforms
    }

    pub fn bind_groups(&self) -> &[BindGroup] {
        &self.bind_groups
    }

    pub fn bounding_box(&self) -> BoundingBox {
        self.bounding_box
    }

    pub fn bounding_radius(&self) -> f32 {
        self.bounding_radius
    }

    pub fn draw<'a>(&'a self, rpass: &mut RenderPass<'a>) {
        for molecule_id in 0..self.molecules().len() {
            rpass.set_bind_group(1, &self.bind_groups()[molecule_id], &[]);

            let start = self.molecules()[molecule_id].lods()[0].1.start;
            let end = self.molecules()[molecule_id].lods()[0].1.end;
            rpass.draw(start..end, 0..self.transforms()[molecule_id].1 as u32);
        }
    }

    pub fn draw_lod<'a>(&'a self, rpass: &mut RenderPass<'a>, distance: f32) {
        for molecule_id in 0..self.molecules().len() {
            rpass.set_bind_group(1, &self.bind_groups()[molecule_id], &[]);

            // Select Its LOD
            for i in 0..self.molecules()[molecule_id].lods().len() {
                if (i == self.molecules()[molecule_id].lods().len() - 1)
                    || (distance > self.molecules()[molecule_id].lods()[i].0
                        && distance < self.molecules()[molecule_id].lods()[i + 1].0)
                {
                    let start = self.molecules()[molecule_id].lods()[i].1.start;
                    let end = self.molecules()[molecule_id].lods()[i].1.end;
                    rpass.draw(start..end, 0..self.transforms()[molecule_id].1 as u32);

                    break;
                }
            }
        }
    }
}

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
                    width: 1024,
                    height: 1024,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::OUTPUT_ATTACHMENT,
            })
            .create_default_view();

        let per_visibility_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Molecule visibility bind group layout"),
                bindings: &[BindGroupLayoutEntry::new(
                    0,
                    ShaderStage::all(),
                    BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                        min_binding_size: None,
                    },
                )],
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
        structure: Rc<Structure>,
        step: u32,
        ranges_limit: usize,
    ) -> StructurePvsField {
        let views_per_circle = 360 / step;
        let sets = vec![None; (views_per_circle * views_per_circle) as usize];

        let r = structure.bounding_radius();
        let projection = ortho_rh_zo(-r, r, -r, r, r * 2.0, -r * 2.0);
        let camera = RotationCamera::new(device, camera_bind_group_layout, &projection, r, 0.0);

        let mut visible = Vec::new();
        let mut visible_staging = Vec::new();
        let mut visible_bind_groups = Vec::new();

        let mut max_size = 0;
        for i in 0..structure.molecules.len() {
            max_size = max_size.max(structure.transforms[i].1);
            visible.push(device.create_buffer_with_data(
                cast_slice(&vec![0i32; structure.transforms[i].1]),
                BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            ));
            visible_staging.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (structure.transforms[i].1 * size_of::<i32>()) as u64,
                usage: BufferUsage::MAP_READ | BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }));
            visible_bind_groups.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                bindings: &[Binding {
                    binding: 0,
                    resource: BindingResource::Buffer(visible.last().unwrap().slice(..)),
                }],
            }));
        }

        let zero_buffer = device.create_buffer_with_data(
            cast_slice(&vec![0i32; max_size]),
            BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        );

        StructurePvsField {
            module: Rc::clone(&self),
            structure: Rc::clone(&structure),
            camera,

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

/// Converts (X, Y, Z) normalized cartesian coordinates to (φ, θ)/(azimuth, latitude) spherical coordinate
pub fn cartesian_to_spherical(v: &Vec3) -> Vec2 {
    let v = normalize(&v);

    vec2(v[2].atan2(v[0]), v[1].atan())
}

/// Converts (φ, θ)/(azimuth, latitude) spherical coordinates to (X, Y, Z) normalized cartesian coordinates
pub fn spherical_to_cartesian(v: &Vec2) -> Vec3 {
    let yaw = v[0];
    let pitch = v[1];

    let x = yaw.cos() * pitch.cos();
    let y = pitch.sin();
    let z = yaw.sin() * pitch.cos();

    vec3(x, y, z)
}

/// `BiologicalStructure`'s field of all potentially visible sets at certain
pub struct StructurePvsField {
    ///
    module: Rc<StructurePvsModule>,

    ///
    structure: Rc<Structure>,

    ///
    camera: RotationCamera,

    ///
    sets: Vec<Option<StructurePvs>>,

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
    fn spherical_to_snapped(&self, spherical_coords: Vec2) -> TVec2<u32> {
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

    fn spherical_to_index(&self, spherical_coords: Vec2) -> usize {
        self.snapped_to_index(self.spherical_to_snapped(spherical_coords))
    }

    fn snapped_to_spherical(&self, snapped_coords: TVec2<u32>) -> Vec2 {
        // Conver to floating point radians.
        let spherical_coords = vec2(
            (snapped_coords.x as f32).to_radians(),
            (snapped_coords.y as f32).to_radians(),
        );

        spherical_coords
    }

    fn snapped_to_index(&self, snapped_coords: TVec2<u32>) -> usize {
        assert!(snapped_coords.x < 360);
        assert!(snapped_coords.y < 360);
        assert!(snapped_coords.x % self.step == 0);
        assert!(snapped_coords.y % self.step == 0);

        let steps = 360 / self.step;

        ((snapped_coords.x / self.step) * steps + (snapped_coords.y / self.step)) as usize
    }

    fn index_to_snapped(&self, index: usize) -> TVec2<u32> {
        let index = index as u32;
        let steps = 360 / self.step;

        vec2((index / steps) * self.step, (index % steps) * self.step)
    }

    fn index_to_spherical(&self, index: usize) -> Vec2 {
        self.snapped_to_spherical(self.index_to_snapped(index))
    }

    pub fn compute(&mut self, device: &Device, queue: &Queue, index: usize) {
        let mut visible = Vec::new();

        // Configure camera
        let spherical_coords = self.index_to_spherical(index);
        self.camera
            .set_yaw(spherical_coords.x as f32);
        self.camera
            .set_pitch(spherical_coords.y as f32);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        self.camera.update_gpu(queue);

        // Clear the visibility data from the device buffers
        for molecule_id in 0..self.structure.molecules.len() {
            encoder.copy_buffer_to_buffer(
                &self.zero_buffer,
                0,
                &self.visible[molecule_id],
                0,
                (self.structure.transforms[molecule_id].1 * size_of::<i32>()) as BufferAddress,
            );
        }

        // Draw the depth buffer
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.module.depth,
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

            rpass.set_pipeline(&self.module.pipeline.pipeline);
            rpass.set_bind_group(0, self.camera.bind_group(), &[]);

            for molecule_id in 0..self.structure.molecules.len() {
                rpass.set_bind_group(1, &self.structure.bind_groups[molecule_id], &[]);

                rpass.draw(
                    self.structure.molecules[molecule_id].lods[0].1.start
                        ..self.structure.molecules[molecule_id].lods[0].1.end,
                    0..self.structure.transforms[molecule_id].1 as u32,
                );
            }
        }

        // Draw a second time without writing to a depth buffer but writing visibility
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.module.depth,
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

            rpass.set_pipeline(&self.module.pipeline_write.pipeline);
            rpass.set_bind_group(0, self.camera.bind_group(), &[]);

            for molecule_id in 0..self.structure.molecules.len() {
                rpass.set_bind_group(1, &self.structure.bind_groups[molecule_id], &[]);
                rpass.set_bind_group(2, &self.visible_bind_groups[molecule_id], &[]);

                rpass.draw(
                    self.structure.molecules[molecule_id].lods[0].1.start
                        ..self.structure.molecules[molecule_id].lods[0].1.end,
                    0..self.structure.transforms[molecule_id].1 as u32,
                );
            }
        }

        // Download the visibility data from the device buffer to the staging one
        for molecule_id in 0..self.structure.molecules.len() {
            encoder.copy_buffer_to_buffer(
                &self.visible[molecule_id],
                0,
                &self.visible_staging[molecule_id],
                0,
                (self.structure.transforms[molecule_id].1 * size_of::<i32>()) as BufferAddress,
            );
        }

        queue.submit(Some(encoder.finish()));

        // Download the visibility data from the staging buffer to CPU
        for molecule_id in 0..self.structure.molecules.len() {
            let buffer_slice = self.visible_staging[molecule_id].slice(..);
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
            device.poll(Maintain::Wait);

            let visible_cpu: Vec<u32> = if let Ok(()) = futures::executor::block_on(buffer_future) {
                let data = buffer_slice.get_mapped_range();
                let result = data
                    .chunks_exact(4)
                    .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();

                drop(data);
                self.visible_staging[molecule_id].unmap();

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
        self.reduce(index);
    }

    /// Computes potentially visible sets from all possible polar coordinates given by `step`.
    pub fn compute_all(&mut self, device: &Device, queue: &Queue) {
        for index in 0..self.sets.len() {
            self.compute(device, queue, index);
        }
    }

    /// Returns potentially visible set from the given viewpoint, given by spherical coordinates in degrees of `step` multiple.
    pub fn get(&mut self, device: &Device, queue: &Queue, index: usize) -> &StructurePvs {
        self.compute(device, queue, index);

        // if self.sets[index].is_some() {
        //     return self.sets[index].as_ref().unwrap();
        // }

        self.sets[index].as_ref().unwrap()
    }

    /// Returns potentially visible set from the given viewpoint.
    ///
    /// # Arguments
    ///
    /// * `eye` - vector **to** viewing point. Must be in the local coordinate system of the structure. If the structure is rotated, so must be the vector.
    ///
    pub fn get_from_eye(&mut self, device: &Device, queue: &Queue, eye: Vec3) -> &StructurePvs {
        let index = self.spherical_to_index(cartesian_to_spherical(&eye));

        self.get(device, queue, index)
    }

    fn reduce(&mut self, index: usize) {
        let pvs = self.sets[index].as_mut().unwrap();

        let mut gaps = vec![Vec::new(); pvs.visible.len()];

        // Find the gaps
        for (molecule_index, ranges) in pvs.visible.iter().enumerate() {
            'ranges: for range_index in 1..ranges.len() {
                let distance = ranges[range_index].0 - ranges[range_index - 1].1;

                // If we find a gap
                if distance > 0 {
                    // Check that it doesn't cross faces
                    // - potentially large range
                    for face in
                        self.structure.transforms_sides.as_ref().unwrap()[molecule_index].iter()
                    {
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
                    let atoms_count = (self.structure.molecules[molecule_index].lods[0].1.end
                        - self.structure.molecules[molecule_index].lods[0].1.start)
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
                panic!("Somethin went wrong");
            }
        }

        // To reduce the list, the ranges are sorted and merged together
        for i in 0..self.visible.len() {
            pvs.visible[i].sort_by(|a, b| a.0.cmp(&b.0));
            pvs.visible[i] = compress_ranges(pvs.visible[i].clone(), 0);
        }
    }

    pub fn draw<'a>(
        &'a mut self,
        device: &Device,
        queue: &Queue,
        rpass: &mut RenderPass<'a>,
        eye: Vec3,
    ) {
        let index = self.spherical_to_index(cartesian_to_spherical(&eye));

        if self.sets[index].is_none() {
            self.get(device, queue, index);
        }

        let pvs = self.sets[index].as_ref().unwrap();

        for molecule_id in 0..self.structure.molecules().len() {
            rpass.set_bind_group(1, &self.structure.bind_groups()[molecule_id], &[]);

            for range in pvs.visible[molecule_id].iter() {
                let start = self.structure.molecules()[molecule_id].lods()[0].1.start;
                let end = self.structure.molecules()[molecule_id].lods()[0].1.end;
                rpass.draw(start..end, range.0..range.1);
            }
        }
    }
}
/// One potentially visible set of field of them.
#[derive(Clone)]
pub struct StructurePvs {
    /// Ranges
    visible: Vec<Vec<(u32, u32)>>,
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
