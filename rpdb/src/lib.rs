pub mod molecule;

use nalgebra_glm::{max2, min2, vec3, Vec3, Vec4, vec4};
use serde::{Deserialize, Serialize};
use lib3dmol::{parser::read_pdb, structures::{atom::AtomType, GetAtom}};
#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

pub fn bounding_box(atoms: &[Vec4]) -> BoundingBox {
    let mut bb_max = vec3(
        std::f32::NEG_INFINITY,
        std::f32::NEG_INFINITY,
        std::f32::NEG_INFINITY,
    );
    let mut bb_min = vec3(std::f32::INFINITY, std::f32::INFINITY, std::f32::INFINITY);
    for atom in atoms.iter() {
        let atom_position = atom.xyz();
        let atom_radius = atom[3];
        bb_max = max2(
            &bb_max,
            &(atom_position + vec3(atom_radius, atom_radius, atom_radius)),
        );
        bb_min = min2(
            &bb_min,
            &(atom_position - vec3(atom_radius, atom_radius, atom_radius)),
        );
    }

    BoundingBox {
        min: bb_min,
        max: bb_max,
    }
}

pub fn center_atoms(mut atoms: Vec<Vec4>) -> Vec<Vec4> {
    // Find bounding box of the entire structure
    let mut bb_max = vec3(
        std::f32::NEG_INFINITY,
        std::f32::NEG_INFINITY,
        std::f32::NEG_INFINITY,
    );
    let mut bb_min = vec3(std::f32::INFINITY, std::f32::INFINITY, std::f32::INFINITY);
    for atom in atoms.iter() {
        let atom_position = atom.xyz();
        let atom_radius = atom[3];
        bb_max = max2(
            &bb_max,
            &(atom_position + vec3(atom_radius, atom_radius, atom_radius)),
        );
        bb_min = min2(
            &bb_min,
            &(atom_position - vec3(atom_radius, atom_radius, atom_radius)),
        );
    }

    // Center the molecules (+their bounding box)
    let bb_center: Vec3 = (bb_max + bb_min) * 0.5;
    for atom in atoms.iter_mut() {
        atom.x -= bb_center.x;
        atom.y -= bb_center.y;
        atom.z -= bb_center.z;
    }

    atoms
}

pub trait FromRon {
    fn from_ron<P: AsRef<std::path::Path>>(p: P) -> molecule::Molecule;
}

pub trait ToRon {
    fn to_ron<P: AsRef<std::path::Path>>(&self, p: P);
}

impl FromRon for molecule::Molecule {
    fn from_ron<P: AsRef<std::path::Path>>(path: P) -> molecule::Molecule {
        let file = std::fs::read_to_string(&path).expect("Could not open structure file.");

        ron::de::from_str(&file).expect("Could not deserialize structure file.")
    }
}

impl ToRon for molecule::Molecule {
    fn to_ron<P: AsRef<std::path::Path>>(&self, path: P) {
        let data = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::new()).expect("Could not serialize the molecule.");
        std::fs::write(&path, data).expect("Could not write the molecule.");
    }
}

pub trait FromPdb {
    fn from_pdb<P: AsRef<std::path::Path>>(p: P) -> molecule::Molecule;
}

impl FromPdb for molecule::Molecule {
    fn from_pdb<P: AsRef<std::path::Path>>(path: P) -> molecule::Molecule {
        let molecule_structure = read_pdb(path.as_ref().to_str().unwrap(), "");

        let mut atoms = Vec::new();        
        for atom in molecule_structure.get_atom() {
            let radius = match atom.a_type {
                AtomType::Carbon => 1.548,
                AtomType::Hydrogen => 1.100,
                AtomType::Nitrogen => 1.400,
                AtomType::Oxygen => 1.348,
                AtomType::Phosphorus => 1.880,
                AtomType::Sulfur => 1.880,
                _ => 1.0, // 'A': 1.5
            };
            atoms.push(vec4(atom.coord[0], atom.coord[1], atom.coord[2], radius));
        }
    
        atoms = center_atoms(atoms);

        molecule::Molecule {
            name: molecule_structure.name().to_string(),
            bounding_box: bounding_box(&atoms),
            lods: vec![molecule::MoleculeLod::new(atoms, 0.0)],
        }
    }
}

// pub fn load_molecules(path: &Path) -> Vec<Vec4> {
//     let mut atoms = Vec::new();
//     let mut molecules = HashMap::new();

//     let directory = path.parent().expect("File must be in a directory.");
//     for entry in fs::read_dir(directory).expect("read_dir call failed") {
//         let entry = entry.unwrap();
//         let molecule_path = entry.path();

//         let molecule_path_str = molecule_path.to_str().unwrap();
//         if molecule_path_str.ends_with(".pdb") {
//             let pdb_name = molecule_path
//                 .file_name()
//                 .unwrap()
//                 .to_str()
//                 .unwrap()
//                 .trim_end_matches(".pdb")
//                 .to_ascii_uppercase();
//             let pdb_molecule = load_molecule(&molecule_path);

//             molecules.insert(pdb_name, pdb_molecule);
//         }
//     }

//     let structure_file = File::open(path).expect("Could not open structure file.");
//     let structure_reader = BufReader::new(&structure_file);
//     for line in structure_reader.lines() {
//         if let Ok(line) = line {
//             let parts: Vec<&str> = line.split(' ').collect();

//             if parts.len() == 9 {
//                 let molecule_name = parts[0];

//                 let molecule_position = vec3(
//                     parts[1].parse::<f32>().unwrap(),
//                     parts[2].parse::<f32>().unwrap(),
//                     parts[3].parse::<f32>().unwrap(),
//                 );
//                 let molecule_quaternion = quat(
//                     -parts[7].parse::<f32>().unwrap(),
//                     parts[4].parse::<f32>().unwrap(),
//                     parts[5].parse::<f32>().unwrap(),
//                     -parts[6].parse::<f32>().unwrap(),
//                 );

//                 let translation = translation(&(3333.33 * molecule_position));
//                 let rotation = quat_to_mat4(&molecule_quaternion);

//                 let molecule_atoms: Vec<Vec4> = molecules[molecule_name]
//                     .clone()
//                     .iter()
//                     .map(|v| translation * rotation * v)
//                     .collect();
//                 atoms.extend(molecule_atoms);
//             }
//         }
//     }

//     atoms
// }
