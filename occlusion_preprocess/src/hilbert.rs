use nalgebra_glm::{max2, min2, normalize, vec2, vec3, Mat4, Vec2, Vec3};
/// X+, X-, Y+, Y-, Z+, Z-
pub enum CubeFace {
    Right,
    Left,
    Top,
    Bottom,
    Front,
    Back,
}

impl From<CubeFace> for u32 {
    fn from(face: CubeFace) -> Self {
        match face {
            CubeFace::Right => 0,
            CubeFace::Left => 1,
            CubeFace::Top => 2,
            CubeFace::Bottom => 3,
            CubeFace::Front => 4,
            CubeFace::Back => 5,
        }
    }
}

fn vector_cube_face(v: &Vec3) -> CubeFace {
    let v = normalize(&v);

    let mut max_i = 0;
    let mut max = 0.0;

    for i in 0..3 {
        if v[i] > max {
            max = v[i];
            max_i = i;
        }
    }

    match max_i {
        0 => {
            if max > 0.0 {
                CubeFace::Right
            } else {
                CubeFace::Left
            }
        }
        1 => {
            if max > 0.0 {
                CubeFace::Top
            } else {
                CubeFace::Bottom
            }
        }
        2 => {
            if max > 0.0 {
                CubeFace::Front
            } else {
                CubeFace::Back
            }
        }
        _ => panic!("Not a possible value."),
    }
}

fn intersect_inside_no(v: &Vec3) -> Vec3 {
    let v_norm = normalize(&v);

    let box_min = vec3(-1.0, -1.0, -1.0);
    let box_max = vec3(1.0, 1.0, 1.0);

    let t_min = box_min.component_div(&v_norm);
    let t_max = box_max.component_div(&v_norm);

    let t1 = min2(&t_min, &t_max);
    let t2 = max2(&t_min, &t_max);

    // let near = t1[0].max(t1[1]).max(t1[2]);
    let far = t2[0].min(t2[1]).min(t2[2]);

    far * v
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        use super::intersect_inside_no;
        use nalgebra_glm::vec3;

        assert_eq!(
            intersect_inside_no(&vec3(1.0, 1.0, 1.0)),
            vec3(1.7320509, 1.7320509, 1.7320509)
        );
        assert_eq!(
            intersect_inside_no(&vec3(-1.0, -1.0, -1.0)),
            vec3(-1.7320509, -1.7320509, -1.7320509)
        );
        assert_eq!(
            intersect_inside_no(&vec3(1.0, 0.0, 0.0)),
            vec3(1.0, 0.0, 0.0)
        );
    }
}

/// Converts (X, Y, Z) normalized cartesian coordinate to (φ, θ)/(azimuth, latitude) spherical coordinate
fn cartesian_to_spherical(v: &Vec3) -> Vec2 {
    let v = normalize(&v);

    vec2(v[1].atan2(v[2]), v[2].acos())
}

/// Rotate/flip a quadrant appropriately
fn rot(n: u32, x: &mut u32, y: &mut u32, rx: u32, ry: u32) {
    if ry == 0 {
        if rx == 1 {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }

        //Swap x and y
        let t = *x;
        *x = *y;
        *y = t;
    }
}

/// Converts (X, Y) coordinates to hilbert index
fn hilbert_xy_to_d(n: u32, mut x: u32, mut y: u32) -> u32 {
    let mut d = 0;
    let mut s = n / 2;
    while s > 0 {
        let rx = ((x & s) > 0) as u32;
        let ry = ((y & s) > 0) as u32;
        d += s * s * ((3 * rx) ^ ry);
        rot(n, &mut x, &mut y, rx, ry);

        s /= 2;
    }

    return d;
}

fn vector_to_xy_face(v: &Vec3) -> (Vec2, CubeFace) {
    let pos = intersect_inside_no(v);
    let face = vector_cube_face(v);

    match face {
        CubeFace::Right => (vec2(-pos.z, pos.y), face),
        CubeFace::Left => (vec2(pos.z, pos.y), face),
        CubeFace::Top => (vec2(-pos.x, pos.z), face),
        CubeFace::Bottom => (vec2(pos.x, pos.z), face),
        CubeFace::Front => (vec2(pos.x, pos.y), face),
        CubeFace::Back => (vec2(-pos.x, pos.y), face),
    }
}

/// n - dimension of the curve
pub fn vector_to_hilbert_face(n: u32, v: &Vec3) -> (u32, CubeFace) {
    // Find the intersection on the cube
    let (xy, face) = vector_to_xy_face(v);

    // Map to integer dimension
    let x = ((xy[0] + 1.0) * (n / 2) as f32).round() as u32;
    let y = ((xy[0] + 1.0) * (n / 2) as f32).round() as u32;

    // Map to hilbert curve on the face
    let hilbert = hilbert_xy_to_d(n, x, y);

    (hilbert, face)
}

pub fn sort_by_hilbert(matrices: &[Mat4]) -> (Vec<Mat4>, Vec<Vec<Vec<Mat4>>>) {
    let n = 64;

    let mut faces = vec![vec![Vec::new(); n * n]; 6];

    let mut new_order = Vec::new();
    for m in matrices {
        let translation = m.column(3).xyz();

        let (d, face) = vector_to_hilbert_face(n as u32, &translation);

        faces[face as usize][d as usize].push(m);
    }

    for face in faces {
        for hilbert_point in face {
            for m in hilbert_point {
                new_order.push(m.clone());
            }
        }
    }

    new_order
}
