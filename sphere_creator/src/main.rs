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

    // let sphere = genmesh::generators::IcoSphere::subdivide(5);
    // let sphere = genmesh::generators::SphereUv::new(90, 180);

    let mut rng = rand::thread_rng();
    let mut noise_generator = noise::SuperSimplex::new();
    // noise_generator.set

    let mut molecules_shell1 = Vec::new();
    let mut molecules_shell2 = Vec::new();
    let factor = 400.0;

    for x in 0..240 {
        for y in 0..240 {
            let x = x as f32 * 1.5;
            let y = y as f32 * 1.5;
            let x = (x as f32).to_radians();
            let y = (y as f32).to_radians();

            let v = spherical_to_cartesian(&vec2(x, y));

            let noise = noise_generator.get([x as f64, y as f64]) as f32;
            let position = v * factor + v * noise * 50.0;
            
            let translation = translation(&position);

            let yaw = std::f32::consts::PI * rng.gen::<f32>();
            let pitch = std::f32::consts::PI * rng.gen::<f32>();
            let roll = std::f32::consts::PI * rng.gen::<f32>();
            let rotation = rotation(yaw, &vec3(1.0, 0.0, 0.0))
                * rotation(pitch, &vec3(0.0, 1.0, 0.0))
                * rotation(roll, &vec3(0.0, 0.0, 1.0));

            let model_matrix = translation * rotation;

            if noise > 0.0 {
                molecules_shell1.push(model_matrix);
            } else {
                molecules_shell2.push(model_matrix);
            }
        }
    }

    let mut molecules_inner = Vec::new();
    for _ in 0..250000 {
        let factor = 300.0;
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

        molecules_inner.push(model_matrix);
    }

    let mut map: HashMap<String, Vec<Mat4>> = HashMap::new();
    map.insert("C1".to_string(), molecules_shell1);
    map.insert("C2".to_string(), molecules_shell2);
    map.insert("C_INNER".to_string(), molecules_inner);
    let structure = Structure { molecules: map };

    let data = ron::ser::to_string(&structure).expect("Could not serialize the structure.");
    std::fs::write(out_path, data).expect("Could not write the structure.");
}
