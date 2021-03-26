use nalgebra_glm::*;
use rpdb::lod::*;
use rpdb::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{prelude::*, BufReader};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_file_path: &str = &args[1];

    if in_file_path.ends_with(".pdb") {
        let name = std::path::Path::new(in_file_path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .trim_end_matches(".pdb")
            .to_ascii_uppercase();

        let molecule = molecule::Molecule::from_pdb(in_file_path);
        println!("Number of atoms: {}", molecule.lods()[0].atoms().len());

        // TODO: Optionally generate LODs

        let out_file_path = if args.len() >= 3 {
            args[2].clone()
        } else {
            std::path::Path::new(in_file_path)
                .parent()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned()
                + "\\"
                + &name
                + ".ron"
        };
        println!("Writing molecule to: {}", out_file_path);

        molecule.to_ron(out_file_path);
    } else if in_file_path.ends_with(".txt") {
        let in_file_path = std::path::Path::new(in_file_path);
        let name = in_file_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .trim_end_matches(".txt")
            .to_ascii_uppercase();

        let file = File::open(in_file_path).expect("Could not open file.");
        let file_reader = BufReader::new(file);
        let mut file_lines = file_reader.lines();

        let mut scale = 1.0f32;
        // if let Some(line) = file_lines.next() {
        //     if let Ok(line) = line {
        //         scale = 1.0f32 / line.parse::<f32>().unwrap();
        //     }
        // }

        let mut molecules: HashMap<String, Vec<Mat4>> = HashMap::new();
        let mut min_position = vec3(std::f32::INFINITY, std::f32::INFINITY, std::f32::INFINITY);
        let mut max_position = vec3(
            std::f32::NEG_INFINITY,
            std::f32::NEG_INFINITY,
            std::f32::NEG_INFINITY,
        );
        for line in file_lines {
            if let Ok(line) = line {
                let parts: Vec<&str> = line.split(' ').collect();

                if parts.len() >= 8 {
                    let molecule_name = parts[0];

                    if molecule_name == "hiv1_sp1_hack_0_1_0"
                    || molecule_name == "1esx"
                    || molecule_name == "nefstef_1"
                    || molecule_name == "1ak4a"
                    || molecule_name == "modbase_vif"
                    || molecule_name == "hiv1_p6_vpr"
                    || molecule_name == "hiv1_sp2_hack_0_1_0"
                    || molecule_name == "hiv1_p6_swissmod_0_1_0"
                    || molecule_name == "1ak4fitto1vu4hex_manu"
                    || molecule_name == "7hvp" {
                        continue;
                    }

                    let molecule_position = vec3(
                        parts[1].parse::<f32>().unwrap(),
                        parts[2].parse::<f32>().unwrap(),
                        parts[3].parse::<f32>().unwrap(),
                    );
                    let molecule_quaternion = quat(
                        parts[5].parse::<f32>().unwrap(),
                        parts[6].parse::<f32>().unwrap(),
                        parts[7].parse::<f32>().unwrap(),
                        parts[4].parse::<f32>().unwrap(),
                    );

                    if molecule_name == "3j3q_1vu4_a_biomt" {
                        min_position = min2(&min_position, &molecule_position);
                        max_position = max2(&max_position, &molecule_position);
                    }

                    let translation = translation(&(scale * molecule_position));
                    let rotation = quat_to_mat4(&molecule_quaternion);
                    let model_matrix = translation * rotation;

                    if !molecules.contains_key(molecule_name) {
                        println!("Converting molecule: {}", molecule_name);

                        // Load existing molecule
                        let mut molecule = molecule::Molecule::from_pdb(
                            in_file_path.with_file_name(molecule_name.to_lowercase() + ".pdb"),
                        );

                        // Create its LODs
                        molecule.create_lods();

                        // Write it to .ron file next the main file
                        molecule
                            .to_ron(in_file_path.with_file_name(molecule_name.to_owned() + ".ron"));

                        // Insert the molecule into the HashMap of molecules
                        molecules.insert(molecule_name.to_string(), vec![]);
                    }

                    if let Some(v) = molecules.get_mut(&molecule_name.to_string()) {
                        v.push(model_matrix);
                    }
                }
            }
        }

        // Center the molecules (+their bounding box)
        // let bb_center: Vec3 = (max_position + min_position) * 0.5;
        // for (_, model_matrices) in molecules.iter_mut() {
        //     for model_matrix in model_matrices.iter_mut() {
        //         *model_matrix = translation(&vec3(-bb_center.x, -bb_center.y, -bb_center.z)) * (*model_matrix);
        //     }
        // }

        for (_, model_matrices) in molecules.iter_mut() {
            model_matrices.retain(|&m| {
                let translation = m.column(3).xyz();

                translation < max_position && translation > min_position
            });
        }

        let structure = structure::Structure { molecules };

        let structure_string =
            ron::ser::to_string(&structure).expect("Could not convert the structure to RON.");

        let out_file_path = if args.len() >= 3 {
            args[2].clone()
        } else {
            std::path::Path::new(in_file_path)
                .parent()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned()
                + "\\"
                + &name
                + ".ron"
        };
        std::fs::write(&out_file_path, structure_string).expect("Could not write the structure.");
    } else {
        println!("Unknown format.");
    }
}
