use kmeans;
use nalgebra_glm::*;
use rpdb::{molecule::Molecule, molecule::MoleculeLod, FromRon, ToRon};

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
    // Load existing molecule
    let args: Vec<String> = std::env::args().collect();
    let mut molecule = Molecule::from_ron(&args[1]);

    // Create new LODs
    let mut lods = Vec::new();
    lods.push(molecule.lods()[0].clone());

    // Constants
    let width = 1920.0;
    let height = 1080.0;
    let aspect = width / height;
    let projection = infinite_perspective_rh_no(aspect, 0.785398163, 0.1);
    let ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                  0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
                  0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001
    ];
    let area_threshold = 32.0;

    // Current largest radius that is being projected
    let mut radius = lods[0].max_radius();
    let mut z = radius * 2.0;

    // Iteratively zoom out
    let mut current_ratio_index = 0;
    'main: loop {
        z += 1.0;

        if current_ratio_index >= ratios.len() - 1 {
            break;
        }

        let view = look_at(
            &vec3(0.0, 0.0, z as f32),
            &vec3(0.0, 0.0, 0.0),
            &vec3(0.0, 1.0, 0.0),
        );
        let position = view * vec4(0.0, 0.0, 0.0, 1.0);
        let area = sphere_sreen_space_area(projection, vec2(width, height), position.xyz(), radius);

        if !area.is_finite() {
            continue;
        }

        // Breakpoint at area limit
        // Currently 64 = 8x8 pixel area
        if area < area_threshold {
            println!("{:?}", area);
            println!("Distance: {}", z);

            // Continue along the reduction ratios
            for reduction_ratio in current_ratio_index..ratios.len() {                
                let new_centroids_num = (lods[0].atoms().len() as f32 * ratios[reduction_ratio]) as usize;
                // println!("Current ratio: {}. New centroids: {}", ratios[reduction_ratio], new_centroids_num);

                // End if It is not possible to reduce further
                if new_centroids_num < 1 {
                    break 'main;
                }

                let new_means = kmeans::reduce(
                    lods[0].atoms(),
                    new_centroids_num,
                );

                println!("Current ratio: {}. New centroids: {}. New means real len: {}", ratios[reduction_ratio], new_centroids_num, new_means.len());
                if new_means.len() == 0 {
                    continue;
                }

                let mut new_lod = MoleculeLod::new(new_means, 0.0);

                let new_area = sphere_sreen_space_area(projection, vec2(width, height), position.xyz(), new_lod.max_radius());
                println!("Possible candidate with {} means of {} area", new_lod.atoms().len(), new_lod.max_radius());

                // If the new are is now above the area limit, save It and continue
                if new_area > area_threshold {
                    radius = new_lod.max_radius();
                    
                    new_lod.set_breakpoint(z);
                    lods.push(new_lod);
                                
                    current_ratio_index = reduction_ratio + 1;
                    break;
                }
            }
        }        
    }

    molecule.lods = lods;
    molecule.to_ron(&args[1]);
}
