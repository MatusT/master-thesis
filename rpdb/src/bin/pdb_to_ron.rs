use rpdb::*;
use std::fs::File;
use std::io::{prelude::*, BufReader};
use std::collections::HashMap;
use nalgebra_glm::{vec3, quat_to_mat4, translation, quat, Mat4};
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

        // TODO: Optionally generate LODs

        let out_file_path = if args.len() >= 3 {
            args[2].clone()
        } else {
            std::path::Path::new(in_file_path).parent().unwrap().to_str().unwrap().to_owned() + "\\" + &name + ".ron"
        };
        println!("Writing to: {}", out_file_path);
        molecule.to_ron(out_file_path);
    } else if in_file_path.ends_with(".txt") {
        let name = std::path::Path::new(in_file_path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .trim_end_matches(".txt")
            .to_ascii_uppercase();

        let file = File::open(in_file_path).expect("Could not open file.");
        let file_reader = BufReader::new(file);

        let mut molecules: HashMap<String, Vec<Mat4>> = HashMap::new();
        for line in file_reader.lines() {
            if let Ok(line) = line {
                let parts: Vec<&str> = line.split(' ').collect();

                if parts.len() == 9 {
                    let molecule_name = parts[0];
    
                    let molecule_position = vec3(
                        parts[1].parse::<f32>().unwrap(),
                        parts[2].parse::<f32>().unwrap(),
                        parts[3].parse::<f32>().unwrap(),
                    );
                    let molecule_quaternion = quat(
                        -parts[7].parse::<f32>().unwrap(),
                        parts[4].parse::<f32>().unwrap(),
                        parts[5].parse::<f32>().unwrap(),
                        -parts[6].parse::<f32>().unwrap(),
                    );
    
                    let translation = translation(&(3333.33 * molecule_position));
                    let rotation = quat_to_mat4(&molecule_quaternion);
                    let model_matrix = translation * rotation;

                    if !molecules.contains_key(molecule_name) {
                        molecules.insert(molecule_name.to_string(), vec![]);
                    }

                    if let Some(v) = molecules.get_mut(&molecule_name.to_string()) {
                        v.push(model_matrix);
                    }
                }
            }            
        }

        let structure = structure::Structure {
            molecules
        };

        let structure_string = ron::ser::to_string(&structure).expect("Could not convert the structure to RON.");

        let out_file_path = if args.len() >= 3 {
            args[2].clone()
        } else {
            std::path::Path::new(in_file_path).parent().unwrap().to_str().unwrap().to_owned() + "\\" + &name + ".ron"
        };
        std::fs::write(&out_file_path, structure_string).expect("Could not write the structure.");
    } else {
        println!("Unknown format.");
    }
}

