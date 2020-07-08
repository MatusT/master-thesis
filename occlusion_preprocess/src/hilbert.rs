use nalgebra_glm::{max2, min2, normalize, vec2, vec3, Mat4, Vec2, Vec3};
/// X+, X-, Y+, Y-, Z+, Z-
#[derive(Copy, Clone, PartialEq, Eq)]
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

pub fn vector_cube_face(v: &Vec3) -> CubeFace {
    let v = normalize(&v);
    let v_abs = v.abs();

    if v_abs.x > v_abs.y && v_abs.x > v_abs.z {
        // X major
        if v.x >= 0.0 {
            CubeFace::Right
        } else {
            CubeFace::Left
        }
    } else if v_abs.y > v_abs.z {
        // Y major
        if v.y >= 0.0 {
            CubeFace::Top
        } else {
            CubeFace::Bottom
        }
    } else {
        // Z major
        if v.z >= 0.0 {
            CubeFace::Front
        } else {
            CubeFace::Back
        }
    }
}

pub fn intersect_inside_no(v: &Vec3) -> Vec3 {
    let v = if *v == vec3(0.0, 0.0, 0.0) {
        vec3(1.0, 0.0, 0.0)
    } else {
        normalize(&v)
    };

    let box_min = vec3(-1.0, -1.0, -1.0);
    let box_max = vec3(1.0, 1.0, 1.0);

    let t_min = box_min.component_div(&v);
    let t_max = box_max.component_div(&v);

    let t1 = min2(&t_min, &t_max);
    let t2 = max2(&t_min, &t_max);

    let near = t1[0].max(t1[1]).max(t1[2]);
    let far = t2[0].min(t2[1]).min(t2[2]);

    let result = far * v;

    assert!(result[0] >= -1.0);
    assert!(result[1] >= -1.0);
    assert!(result[2] >= -1.0);
    assert!(result[0] <= 1.0);
    assert!(result[1] <= 1.0);
    assert!(result[2] <= 1.0);

    result
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
pub fn hilbert_xy_to_d(n: u32, mut x: u32, mut y: u32) -> u32 {
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

pub fn vector_to_xy_face(v: &Vec3) -> (Vec2, CubeFace) {
    let pos = intersect_inside_no(v);
    let face = vector_cube_face(&pos);

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

    // println!("{:?} {:?}", v, xy);

    assert!(xy[0] >= -1.0);
    assert!(xy[0] >= -1.0);
    assert!(xy[1] <= 1.0);
    assert!(xy[1] <= 1.0);

    // Map to integer dimension
    let x = ((xy[0] + 1.0) * (n / 2) as f32).round() as u32;
    let y = ((xy[1] + 1.0) * (n / 2) as f32).round() as u32;

    // Map to hilbert curve on the face
    let hilbert = hilbert_xy_to_d(n, x, y);

    (hilbert, face)
}

pub fn sort_by_hilbert(matrices: &[Mat4]) -> (Vec<Mat4>, Vec<Vec<Vec<Mat4>>>, [u32; 6]) {
    let n = 128;

    let mut faces: Vec<Vec<Vec<Mat4>>> = vec![vec![Vec::new(); n * n]; 6];

    let mut new_order: Vec<Mat4> = Vec::new();
    for m in matrices {
        let translation = m.column(3).xyz();

        let (d, face) = vector_to_hilbert_face(n as u32, &translation);

        faces[face as usize][d as usize].push(*m);
    }

    let mut faces_starts = [0u32; 6];
    let mut sum = 0u32;
    for (face_index, face) in faces.iter().enumerate() {
        for hilbert_point in face.iter() {
            for m in hilbert_point.iter() {
                new_order.push(m.clone());
            }
            sum += hilbert_point.len() as u32;
        }
        faces_starts[face_index] = sum;
    }

    (new_order, faces, faces_starts)
}

#[cfg(test)]
mod tests {
    #[test]
    fn vector_cube_face() {
        use super::{vector_cube_face, CubeFace};
        use nalgebra_glm::vec3;

        assert!(vector_cube_face(&vec3(1.0, 0.0, 0.0)) == CubeFace::Right);
        assert!(vector_cube_face(&vec3(-1.0, 0.0, 0.0)) == CubeFace::Left);
        assert!(vector_cube_face(&vec3(0.0, 0.0, 1.0)) == CubeFace::Front);
        assert!(vector_cube_face(&vec3(0.0, 0.0, -1.0)) == CubeFace::Back);
        assert!(vector_cube_face(&vec3(0.0, 1.0, 0.0)) == CubeFace::Top);
        assert!(vector_cube_face(&vec3(0.0, -1.0, 0.0)) == CubeFace::Bottom);
    }

    #[test]
    fn intersect_inside_no() {
        use super::intersect_inside_no;
        use nalgebra_glm::vec3;

        assert_eq!(
            intersect_inside_no(&vec3(10.0, 10.0, 10.0)),
            vec3(1.0, 1.0, 1.0)
        );
        assert_eq!(
            intersect_inside_no(&vec3(-1.0, -1.0, -1.0)),
            vec3(-1.0, -1.0, -1.0)
        );
        assert_eq!(
            intersect_inside_no(&vec3(1000.0, 0.0, 0.0)),
            vec3(1.0, 0.0, 0.0)
        );
    }
}
