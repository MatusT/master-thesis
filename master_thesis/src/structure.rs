///!
///! Vec<Atoms>
///!      ^
///!      | has multiple
///! Vec<Molecules>
///!      ^
///!      | has multiple
///! Vec<Structure>
use bytemuck::cast_slice;
use nalgebra_glm::{distance, length, vec3, Vec3};
use rpdb;
use rpdb::BoundingBox;
use rpdb::FromRon;
use wgpu::util::*;
use wgpu::*;

use crate::hilbert;

/// GPU represantion of a molecule for visualization.
pub struct Molecule {
    name: String,

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

    ///
    color: Vec3,
}

impl Molecule {
    pub fn from_ron<P: AsRef<std::path::Path>>(device: &Device, path: P) -> Self {
        let name = path.as_ref().file_stem().unwrap().to_str().unwrap();

        let molecule = rpdb::molecule::Molecule::from_ron(&path);

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

        let atoms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: cast_slice(&atoms),
            usage: BufferUsage::STORAGE,
        });

        let bounding_box = *molecule.bounding_box();
        let bounding_radius = distance(&bounding_box.max, &bounding_box.min) / 2.0;

        Self {
            name: name.to_string(),
            atoms,
            lods,
            bounding_box,
            bounding_radius,
            color: vec3(1.0, 1.0, 1.0),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
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

    pub fn color(&self) -> Vec3 {
        self.color
    }

    pub fn set_color(&mut self, color: &Vec3) {
        self.color = *color;
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
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: cast_slice(&molecule_model_matrices),
                    usage: BufferUsage::STORAGE,
                }),
                molecule_model_matrices_len,
            ));

            transforms_sides.push(hilbert.2);

            bind_groups.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &per_molecule_bind_group_layout,
                entries: (&[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: &new_molecule.atoms(),
                            offset: 0,
                            size: None,
                        },
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer {
                            buffer: &transforms.last().unwrap().0,
                            offset: 0,
                            size: None,
                        },
                    },
                ]),
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

    pub fn molecules_mut(&mut self) -> &mut [Molecule] {
        &mut self.molecules
    }

    pub fn transforms(&self) -> &[(Buffer, usize)] {
        &self.transforms
    }

    pub fn transforms_sides(&self) -> Option<&Vec<[u32; 6]>> {
        self.transforms_sides.as_ref()
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
            // if self.structure.molecules()[molecule_id].name() != "CRYSTALL_CUT_SINGLE"
            // && self.structure.molecules()[molecule_id].name() != "CRYSTALL_CUT_SINGLE2"
            // && self.structure.molecules()[molecule_id].name() != "FLUID_CUT_SINGLE"
            // && self.structure.molecules()[molecule_id].name() != "FLUID_CUT_SINGLE2"
            // && self.structure.molecules()[molecule_id].name() != "E"
            // && self.structure.molecules()[molecule_id].name() != "M" {
            //     continue;
            // }
            // if self.molecules()[molecule_id].name() == "S"  {
            //     continue;
            // }

            rpass.set_bind_group(1, &self.bind_groups()[molecule_id], &[]);

            let start = self.molecules()[molecule_id].lods()[0].1.start;
            let end = self.molecules()[molecule_id].lods()[0].1.end;

            let color: [f32; 3] = self.molecules()[molecule_id].color.into();
            rpass.set_push_constants(ShaderStage::FRAGMENT, 16, cast_slice(&color));
            rpass.draw(start..end, 0..self.transforms()[molecule_id].1 as u32);
        }
    }

    pub fn draw_lod<'a>(&'a self, rpass: &mut RenderPass<'a>, distance: f32) {
        for molecule_id in 0..self.molecules().len() {
            // println!("{} {}", self.structure.molecules()[molecule_id].name(), self.structure.transforms()[molecule_id].1);
            // if self.structure.molecules()[molecule_id].name() != "CRYSTALL_CUT_SINGLE"
            // && self.structure.molecules()[molecule_id].name() != "CRYSTALL_CUT_SINGLE2"
            // && self.structure.molecules()[molecule_id].name() != "FLUID_CUT_SINGLE"
            // && self.structure.molecules()[molecule_id].name() != "FLUID_CUT_SINGLE2"
            // && self.structure.molecules()[molecule_id].name() != "E"
            // && self.structure.molecules()[molecule_id].name() != "M" {
            //     continue;
            // }
            // if self.molecules()[molecule_id].name() == "S"  {
            //     continue;
            // }

            rpass.set_bind_group(1, &self.bind_groups()[molecule_id], &[]);

            // Select Its LOD
            for i in 0..self.molecules()[molecule_id].lods().len() {
                if (i == self.molecules()[molecule_id].lods().len() - 1)
                    || (distance > self.molecules()[molecule_id].lods()[i].0
                        && distance < self.molecules()[molecule_id].lods()[i + 1].0)
                {
                    let start = self.molecules()[molecule_id].lods()[i].1.start;
                    let end = self.molecules()[molecule_id].lods()[i].1.end;

                    let color: [f32; 3] = self.molecules()[molecule_id].color.into();
                    rpass.set_push_constants(ShaderStage::FRAGMENT, 16, cast_slice(&color));
                    rpass.draw(start..end, 0..self.transforms()[molecule_id].1 as u32);

                    break;
                }
            }
        }
    }
}
