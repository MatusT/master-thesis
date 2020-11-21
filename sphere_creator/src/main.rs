use nalgebra_glm::*;
use noise::{NoiseFn, Perlin, Worley};
use rand::prelude::*;
use rand_distr::{Distribution, UnitBall, UnitSphere};
use ron;
use rpdb::{molecule::*, structure::*, FromRon};
use std::collections::HashMap;
use genmesh;
use genmesh::Vertices;

pub fn fmodulo(mut input: f32, modulus: f32) -> f32 {
    if input.is_sign_negative() {
        while input <= -modulus {
            input = input + modulus;
        }
    } else if input.is_sign_positive() {
        while input >= modulus {
            input = input - modulus;
        }
    }

    input
}

/// Converts (φ, θ)/(azimuth, latitude) spherical coordinates to (X, Y, Z) normalized cartesian coordinates
pub fn spherical_to_cartesian(spherical: &TVec2<f32>) -> TVec3<f32> {
    let mut theta = fmodulo(spherical.x, std::f32::consts::TAU);

    if theta >= std::f32::consts::PI {
        theta = -(std::f32::consts::TAU - theta);
    }
    if theta <= -std::f32::consts::PI {
        theta = std::f32::consts::TAU + theta;
    }

    let phi = spherical.y;

    let x = phi.sin() * theta.cos();
    let y = phi.sin() * theta.sin();
    let z = phi.cos();

    vec3(x, z, y)
}

pub fn cartesian_to_spherical(cartesian: &TVec3<f32>) -> TVec2<f32> {
    let x = cartesian.x;
    let y = cartesian.z;
    let z = cartesian.y;

    let theta = y.atan2(x);
    let phi = (x * x + y * y).sqrt().atan2(z);

    vec2(theta, phi)
}

fn main() {
    // Load existing molecule
    let args: Vec<String> = std::env::args().collect();
    let out_path = args[1].clone();

    let mut rng = rand::thread_rng();
    let noise_generator = noise::SuperSimplex::new();

    let factor = 300.0;

    let mut map: HashMap<String, Vec<Mat4>> = HashMap::new();
    for i in 1..=40 {
        map.insert("OUTER_".to_string() + &i.to_string(), vec![]);
        map.insert("INNER_".to_string() + &i.to_string(), vec![]);
    }
    // let mut main_shell = Vec::new();
    for x in (0..360).step_by(6) {
        for y in (0..360).step_by(6) {
            let i = y * 240 + x;
            let xs = (x as f32).to_radians();
            let ys = (y as f32).to_radians();

            let v = spherical_to_cartesian(&vec2(xs, ys));

            let noise = noise_generator.get([xs as f64, ys as f64]) as f32;
            let position = v * factor + v * noise * 80.0;
            
            let t = translation(&position);

            let yaw = std::f32::consts::PI * rng.gen::<f32>();
            let pitch = std::f32::consts::PI * rng.gen::<f32>();
            let roll = std::f32::consts::PI * rng.gen::<f32>();
            let rotation = rotation(yaw, &vec3(1.0, 0.0, 0.0))
                * rotation(pitch, &vec3(0.0, 1.0, 0.0))
                * rotation(roll, &vec3(0.0, 0.0, 1.0));

            let model_matrix = t * rotation;

            let i = rng.gen_range(10, 20);
            let name = "OUTER_".to_string() + &i.to_string();
            map.get_mut(&name).unwrap().push(model_matrix);

            // main_shell.push(model_matrix);

            // if x % 5 == 0 || y % 5 == 0 {
            //     let xs = xs + 0.5f32.to_radians();
            //     let ys = ys + 0.5f32.to_radians();
            //     let noise = noise_generator.get([xs as f64, ys as f64]) as f32;
            //     let position = v * factor + v * noise * 80.0;
            //     let t = translation(&position);
            //     let model_matrix = t * rotation;

            //     let i = rng.gen_range(1, 5);
            //     let name = "OUTER_".to_string() + &i.to_string();
            //     map.get_mut(&name).unwrap().push(model_matrix);
            // }
        }
    }    

    // let mut molecules_inner = Vec::new();
    for _ in 0..10000 {
        let factor = 150.0;
        let v: [f32; 3] = UnitBall.sample(&mut rand::thread_rng());
        let position = vec3(v[0] * factor, v[1] * factor, v[2] * factor);
        let translation = translation(&position);

        let yaw = std::f32::consts::PI * rng.gen::<f32>();
        let pitch = std::f32::consts::PI * rng.gen::<f32>();
        let roll = std::f32::consts::PI * rng.gen::<f32>();
        let rotation = rotation(yaw, &vec3(1.0, 0.0, 0.0))
            * rotation(pitch, &vec3(0.0, 1.0, 0.0))
            * rotation(roll, &vec3(0.0, 0.0, 1.0));

        let model_matrix = translation * rotation;

        let i = rng.gen_range(10, 20);
        let name = "INNER_".to_string() + &i.to_string();
        map.get_mut(&name).unwrap().push(model_matrix);

        // molecules_inner.push(model_matrix);
    }
    // map.insert("C_INNER".to_string(), molecules_inner);
    // map.insert("C_SHELL".to_string(), main_shell);

    map.retain(|_, v| !v.is_empty());


    let structure = Structure { molecules: map };
    let data = ron::ser::to_string(&structure).expect("Could not serialize the structure.");
    std::fs::write(out_path, data).expect("Could not write the structure.");
}
