pub mod camera;
pub mod framework;
pub mod frustrum_culler;
pub mod hilbert;
pub mod pipelines;
pub mod pvs;
pub mod ssao;
pub mod structure;

use nalgebra_glm::*;
pub enum ApplicationEvent<'a> {
    WindowEvent(winit::event::WindowEvent<'a>),
    DeviceEvent(winit::event::DeviceEvent),
}

pub fn fmodulo(mut input: f64, modulus: f64) -> f64 {
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

/// Converts (X, Y, Z) normalized cartesian coordinates to (φ, θ)/(azimuth, latitude) spherical coordinate
pub fn cartesian_to_spherical(cartesian: &TVec3<f64>) -> TVec2<f64> {
    let x = cartesian.x;
    let y = cartesian.z;
    let z = cartesian.y;

    let theta = y.atan2(x);
    let phi = (x * x + y * y).sqrt().atan2(z);

    vec2(theta, phi)
}

/// Converts (φ, θ)/(azimuth, latitude) spherical coordinates to (X, Y, Z) normalized cartesian coordinates
pub fn spherical_to_cartesian(spherical: &TVec2<f64>) -> TVec3<f64> {
    let mut theta = fmodulo(spherical.x, std::f64::consts::TAU);

    if theta >= std::f64::consts::PI {
        theta = -(std::f64::consts::TAU - theta);
    }
    if theta <= -std::f64::consts::PI {
        theta = std::f64::consts::TAU + theta;
    }

    let phi = spherical.y;

    let x = phi.sin() * theta.cos();
    let y = phi.sin() * theta.sin();
    let z = phi.cos();

    vec3(x, z, y)
}
