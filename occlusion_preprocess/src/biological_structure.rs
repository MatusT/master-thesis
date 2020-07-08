///
/// Vec<Atoms>
///      ^
///      | has multiple 
/// Vec<Molecules>
///      ^
///      | has multiple 
/// Vec<Structure>

use wgpu::*;
use nalgebra_glm::{vec3, Vec3, ortho_rh_zo, vec2, Vec2, normalize, TVec2, zero, distance};
use rpdb;
use rpdb::FromRon;
use rpdb::BoundingBox;
use bytemuck::cast_slice;

use std::mem::size_of;

use crate::camera::*;
use crate::pipelines::SphereBillboardsDepthPipeline;
use crate::hilbert;

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

        Self {
            atoms,
            lods,
            bounding_box: *molecule.bounding_box(),
        }
    }

    pub fn atoms(&self) -> &Buffer {
        &self.atoms
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

    ///
    bind_group_layout: BindGroupLayout,

    /// Bind groups for each molecule type containing reference to `molecules` and `transforms`.
    bind_groups: Vec<BindGroup>,

    /// Bounding box of the entire structure. Creates as an intersection of all bounding boxes of structure's molecules.
    bounding_box: BoundingBox,

    /// 
    bounding_radius: f32,
}

impl Structure {
    pub fn from_ron<P: AsRef<std::path::Path>>(device: &Device, path: P) -> Self {
        let structure_file = rpdb::structure::Structure::from_ron(&path);

        let mut molecules = Vec::new();

        let mut transforms = Vec::new();
        let mut transforms_sides = Vec::new();

        let mut bind_groups = Vec::new();

        let mut bounding_box = BoundingBox { min: zero(), max: zero() };

        let bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Molecule bind group layout"),
                bindings: &[
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
                ],
            });

        for (molecule_name, molecule_model_matrices) in structure_file.molecules {
            let new_molecule = Molecule::from_ron(device, path.as_ref().with_file_name(molecule_name + ".ron"));
            
            let hilbert = hilbert::sort_by_hilbert(&molecule_model_matrices);
            let molecule_model_matrices = hilbert.0;
            let molecule_model_matrices_len = molecule_model_matrices.len();
            let molecule_model_matrices = {
                let mut matrices_flat: Vec<f32> = Vec::new();
                for molecule_model_matrix in molecule_model_matrices {
                    matrices_flat.append(&mut molecule_model_matrix.as_slice().to_owned());
                }

                matrices_flat
            };

            transforms.push((device.create_buffer_with_data(
                cast_slice(&molecule_model_matrices),
                BufferUsage::STORAGE,
            ), molecule_model_matrices_len));

            transforms_sides.push(hilbert.2);

            bind_groups.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
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

            bounding_box.union(&new_molecule.bounding_box());

            molecules.push(new_molecule);
        }

        let bounding_radius = distance(&bounding_box.max, &bounding_box.min);

        Self {
            molecules,
            transforms,
            transforms_sides: Some(transforms_sides),
            bind_group_layout,
            bind_groups,
            bounding_box,
            bounding_radius,            
        }
    }

    pub fn bounding_box(&self) -> BoundingBox {
        self.bounding_box
    }

    pub fn bounding_radius(&self) -> f32 {
        self.bounding_radius
    }
}

pub struct StructurePvsModule<'a> {
    ///
    device: &'a Device, 

    ///
    depth: TextureView,
    
    ///
    pipeline: SphereBillboardsDepthPipeline,

    bind_group_layout: BindGroupLayout,
}

impl<'a>  StructurePvsModule<'a> {
    pub fn new(device: &'a Device, per_molecule_bind_group_layout: &BindGroupLayout) -> Self {
        let depth = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 512,
                height: 512,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsage::OUTPUT_ATTACHMENT,
        }).create_default_view();

        let camera_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Camera bind group layout"),
            bindings: &[BindGroupLayoutEntry::new(
                0,
                ShaderStage::all(),
                BindingType::UniformBuffer {
                    dynamic: false,
                    min_binding_size: Some(CameraUbo::size()),
                },
            )],
        });

        let per_visibility_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Molecule visibility bind group layout"),
                bindings: &[
                    BindGroupLayoutEntry::new(
                        0,
                        ShaderStage::all(),
                        BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                            min_binding_size: None,
                        },
                    ),
                ],
            });

        let pipeline = SphereBillboardsDepthPipeline::new(
            &device,
            &camera_bind_group_layout,
            &per_molecule_bind_group_layout,
            Some(&per_visibility_bind_group_layout),
            1,
            false,
        );

        Self {
            device,
            depth,
            pipeline,
            bind_group_layout: per_visibility_bind_group_layout,
        }
    }

    ///
    pub fn pvs_field(&'a self, structure: &'a Structure, step: u32, ranges_limit: usize) -> StructurePvsField {
        let views_per_circle = 360 / step;
        let sets = vec![None; (views_per_circle * views_per_circle) as usize];

        let r = structure.bounding_radius();
        let projection = ortho_rh_zo(-r, r, -r, r, 0.0, r * 2.0);
        let camera = RotationCamera::new(self.device, &projection, r * 2.0, 0.0);

        let visible = Vec::new();
        let visible_staging = Vec::new();
        let visible_bind_groups = Vec::new();

        for i in 0..structure.molecules.len() {
            visible.push(self.device.create_buffer_with_data(
                cast_slice(&vec![0i32; structure.transforms[i].1]),
                BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            ));
            visible_staging.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (structure.transforms[i].1
                    * size_of::<i32>()) as u64,
                usage: BufferUsage::MAP_READ | BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }));
            visible_bind_groups.push(self.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                bindings: &[
                    Binding {
                        binding: 0,
                        resource: BindingResource::Buffer(
                            visible.last().unwrap().slice(..),
                        ),
                    },
                ],
            }));
        }

        StructurePvsField {
            module: &self,
            structure,
            camera,

            sets,

            step,
            ranges_limit,

            visible,
            visible_staging,
            visible_bind_groups,
        }
    } 
}

// TODO: snap polar

/// Converts (X, Y, Z) normalized cartesian coordinates to (φ, θ)/(azimuth, latitude) spherical coordinate
pub fn cartesian_to_spherical(v: &Vec3) -> Vec2 {
    let v = normalize(&v);

    vec2(v[0].atan2(v[1]), v[2].acos())
}

/// Converts (φ, θ)/(azimuth, latitude) spherical coordinates to (X, Y, Z) normalized cartesian coordinates
pub fn spherical_to_cartesian(v: &Vec2) -> Vec3 {
    let x = v[0].sin() * v[1].cos();
    let y = v[0].sin() * v[1].sin();
    let z = v[0].cos();

    vec3(x, y, z)
}

/// `BiologicalStructure`'s field of all potentially visible sets at certain 
pub struct StructurePvsField<'a> {
    /// 
    module: &'a StructurePvsModule<'a>,

    ///
    structure: &'a Structure,

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
}

impl<'a> StructurePvsField<'a> {
    /// Computes potentially visible sets from all possible polar coordinates given by `step`.
    pub fn compute_all(&mut self) {

    }

    /// Returns potentially visible set from the given viewpoint.
    pub fn at_coordinates(&mut self, spherical_coords: TVec2<u32>) -> &StructurePvs {
        assert!(spherical_coords[0] % self.step == 0);
        assert!(spherical_coords[1] % self.step == 0);

        let steps = 360 / self.step;
        let index = (spherical_coords[0] * steps + spherical_coords[1]) as usize;

        match self.sets[index] {
            Some(structure) => &structure,
            None => {
                // TODO: Compute pvs
            }
        }
    }

    /// Returns potentially visible set from the given viewpoint.
    pub fn pvs_from_eye(&mut self, eye: Vec3) -> &StructurePvs {
        let spherical_coords = cartesian_to_spherical(&eye).apply_into(|e| e.to_degrees().round());

        // Snap spherical coordinates to the closest view given by step size.
        let spherical_coords = TVec2::new(
            ((spherical_coords[0] as u32 + self.step - 1) / self.step) * self.step,
            ((spherical_coords[0] as u32 + self.step - 1) / self.step) * self.step
        );

        self.at_coordinates(spherical_coords)
    }
}
/// One potentially visible set of field of them.
#[derive(Clone)]
pub struct StructurePvs {
    /// Ranges
    visible: Vec<Vec<(u32, u32)>>,
}

impl StructurePvs {
    pub fn reduce(limit: u32) {}
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

pub fn reduce_visible(
    molecules_atoms: &[u32],
    mut molecules_ranges: Vec<Vec<(u32, u32)>>,
    molecules_faces: &[[u32; 6]],
    limit: usize,
) -> Vec<Vec<(u32, u32)>> {
    let mut gaps = vec![Vec::new(); molecules_ranges.len()];

    // Find the gaps
    for (molecule_index, ranges) in molecules_ranges.iter().enumerate() {
        'ranges: for range_index in 1..ranges.len() {
            let distance = ranges[range_index].0 - ranges[range_index - 1].1;

            if distance > 0 {
                // Check that it doesn't cross faces
                for face in molecules_faces[molecule_index].iter() {
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

    // Run the greedy staff
    let mut ranges_num: usize = molecules_ranges.iter().map(|v| v.len()).sum();
    // While we are not under imposed limit
    while ranges_num > limit {
        // Run through gaps of each molecule and find the gap with smallest cost (distance * number of atoms)
        let mut min_gap = None;
        let mut min_cost = u32::MAX;
        let mut min_index = 0;
        for (molecule_index, gaps) in gaps.iter().enumerate() {
            if let Some(gap) = gaps.last() {
                let cost = (gap.1 - gap.0) * molecules_atoms[molecule_index];

                if cost < min_cost {
                    min_gap = Some(gap);
                    min_cost = cost;
                    min_index = molecule_index;
                }
            }
        }

        if let Some(gap) = min_gap {
            molecules_ranges[min_index].push(*gap);
            gaps[min_index].pop();
            ranges_num -= 1;
        } else {
            panic!("Somethin went wrong");
        }
    }

    for i in 0..molecules_ranges.len() {
        molecules_ranges[i].sort_by(|a, b| a.0.cmp(&b.0));
        molecules_ranges[i] = compress_ranges(molecules_ranges[i].clone(), 0);
    }

    molecules_ranges
}

#[cfg(test)]
mod tests {
    #[test]
    fn compress_ranges() {
        use super::compress_ranges;

        let input = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let output = vec![(0, 4)];
        assert_eq!(compress_ranges(input, 0), output);

        let input = vec![(34, 35), (35, 36), (36, 39)];
        let output = vec![(34, 39)];
        assert_eq!(compress_ranges(input, 0), output);
    }
}
