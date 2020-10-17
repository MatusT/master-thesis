use nalgebra_glm::{distance, distance2, vec4, zero, Vec3, Vec4};
use rand::prelude::*;
use rayon::prelude::*;

pub fn reduce(points: &[Vec4], centroids_num: usize) -> Vec<Vec4> {
    let mut rng = rand::thread_rng();

    let mut centroids: Vec<Vec4> = Vec::new();
    let mut memberships: Vec<i32> = vec![0; points.len()];    

    // Init
    let centroid_step = (points.len() as f32 / centroids_num as f32).ceil() as usize;
    for i in 0..centroids_num {
        if i * centroid_step < points.len() {
            centroids.push(points[i * centroid_step]);
        } else {
            centroids.push(points[(rng.gen::<f32>() * points.len() as f32).floor() as usize]);
        }
    }

    // Iterations
    for _ in 0..5 {
        // Find centroids
        points.par_iter().zip_eq(memberships.par_iter_mut()).for_each(|(point, membership)| {
            let (closest_centroid, _) = centroids.iter().enumerate().fold((0, std::f32::INFINITY), |acc, (i, c)| { 
                let d = distance2(&point, &c);
                if d < acc.1 { (i, d) } 
                else { acc }         
            });

            *membership = closest_centroid as i32;
        });

        // Update centroids
        centroids.par_iter_mut().enumerate().for_each(|(centroid_index, centroid)| {
            let mut member_count = 0;
            let mut bounding_radius = 0.0f32;

            let mut new_centroid: Vec3 = zero();
            let old_centroid = centroid.clone();

            for i in 0..points.len() {
                let add_centroid: bool = memberships[i] == centroid_index as i32;
                let inc_point: Vec4 = if add_centroid { points[i] } else { zero() };

                new_centroid += inc_point.xyz();
                member_count += if add_centroid { 1 } else { 0 };
                bounding_radius = if add_centroid {
                    if distance(&old_centroid.xyz(), &inc_point.xyz()) + inc_point[3] > bounding_radius {
                        distance(&old_centroid.xyz(), &inc_point.xyz()) + inc_point[3]
                    } else {
                        bounding_radius
                    }
                } else {
                    bounding_radius
                };
            }

            new_centroid = new_centroid * (1.0 / member_count as f32);

            *centroid = vec4(new_centroid[0], new_centroid[1], new_centroid[2], bounding_radius);
        });
    }

    centroids.into_iter().filter(|v| v[3] > 0.0).collect()
}

