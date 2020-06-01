use nalgebra_glm::*;

fn sphere_sreen_space_area(projection: Mat4, dimensions: Vec2, center: Vec3, radius: f32) -> f32 {
    let d2 = dot(&center, &center);
    let a = (d2 - radius * radius).sqrt();

    // view-aligned "right" vector (right angle to the view plane from the center of the sphere. Since  "up" is always (0,n,0), replaced cross product with vec3(-c.z, 0, c.x)
    let right = (radius / a) * vec3(-center.z, 0.0, center.x);
    let up = vec3(0.0, radius, 0.0);

    let projected_right = projection * vec4(right.x, right.y, right.z, 0.0);
    let projected_up = projection * vec4(up.x, up.y, up.z, 0.0);

    let projected_center = projection * vec4(center.x, center.y, center.z, 1.0);

    let mut north = projected_center + projected_up;
    let mut east = projected_center + projected_right;
    let mut south = projected_center - projected_up;
    let mut west = projected_center - projected_right;

    north /= north.w;
    east /= east.w;
    west /= west.w;
    south /= south.w;

    let north = vec2(north.x, north.y);
    let east = vec2(east.x, east.y);
    let west = vec2(west.x, west.y);
    let south = vec2(south.x, south.y);

    let box_min = min2(&min2(&min2(&east, &west), &north), &south);
    let box_max = max2(&max2(&max2(&east, &west), &north), &south);

    let box_min = box_min * 0.5 + vec2(0.5, 0.5);
    let box_max = box_max * 0.5 + vec2(0.5, 0.5);

    let area = box_max - box_min;
    let area = area.component_mul(&dimensions);
    let area = area.x * area.y;

    area
}

fn main() {
    let width = 3840.0;
    let height = 2060.0;
    let aspect = width / height;

    let projection = infinite_perspective_rh_no(aspect, 0.785398163, 0.1);
    let radius = 0.5;

    for z in ((radius * 2.0) as u32..1000).step_by(1) {
        let view = look_at(
            &vec3(0.0, 0.0, z as f32),
            &vec3(0.0, 0.0, 0.0),
            &vec3(0.0, 1.0, 0.0),
        );
        let position = view * vec4(0.0, 0.0, 0.0, 1.0);
        let area = sphere_sreen_space_area(projection, vec2(width, height), position.xyz(), radius);

        println!("{:?}", area);

        if area < 64.0 {
            println!("Distance: {}", z);
            break;
        }
    }
}

// let mut img = RgbImage::new(width as u32, height as u32);
// Draw the sphere/rectangle
// for ss_x in 0..dimensions.x as u32 {
//     for ss_y in 0..dimensions.y as u32 {
//         let img_x = center_ss.x as u32 - dimensions.x as u32 / 2 + ss_x;
//         let img_y = center_ss.y as u32 - dimensions.y as u32 / 2 + ss_y;

//         if img_x >= width as u32 || img_y >= height as u32 {
//             continue;
//         }

//         img.put_pixel(img_x, img_y, Rgb([255, 0, 0]));
//     }
// }

// img.save(format!("test_{}.png", z as u32)).unwrap();
