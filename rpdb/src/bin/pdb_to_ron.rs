use rpdb::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let in_file_path: &str = &args[1];

    let name = std::path::Path::new(in_file_path)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .trim_end_matches(".pdb")
        .to_ascii_uppercase();

    let molecule = molecule::Molecule::from_pdb(in_file_path);

    // TODO: Optionally generate LODs

    // Convert the molecule to a new RON format
    let out_file_path = if args.len() >= 3 {
        args[2].clone()
    } else {
        std::path::Path::new(in_file_path).parent().unwrap().to_str().unwrap().to_owned() + "\\" + &name + ".ron"
    };
    println!("Writing to: {}", out_file_path);

    molecule.to_ron(out_file_path);
}

