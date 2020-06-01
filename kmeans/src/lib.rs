use nalgebra_glm::{distance, distance2, vec4, zero, Vec3, Vec4};
use rand::prelude::*;

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
        for (point_index, point) in points.iter().enumerate() {
            let mut min_dist = std::f32::INFINITY;
            let mut closest_centroid = -1;
            for (centroid_index, centroid) in centroids.iter().enumerate() {
                let dist = distance2(&point, &centroid);

                if dist < min_dist {
                    min_dist = dist;
                    closest_centroid = centroid_index as i32;
                }
            }

            memberships[point_index] = closest_centroid;
        }

        // Update centroids
        for id in 0..centroids.len() {
            let mut member_count = 0;
            let mut bounding_radius = 0.0f32;

            let mut new_centroid: Vec3 = zero();
            let old_centroid: Vec3 = centroids[id].xyz();

            for i in 0..points.len() {
                let add_centroid: bool = memberships[i] == id as i32;
                let inc_point: Vec3 = if add_centroid { points[i].xyz() } else { zero() };

                new_centroid += inc_point;
                member_count += if add_centroid { 1 } else { 0 };
                bounding_radius = if add_centroid {
                    if distance(&old_centroid, &inc_point) > bounding_radius {
                        distance(&old_centroid, &inc_point)
                    } else {
                        bounding_radius
                    }
                } else {
                    bounding_radius
                };
            }

            new_centroid = new_centroid * (1.0 / member_count as f32);
            centroids[id] = vec4(new_centroid[0], new_centroid[1], new_centroid[2], bounding_radius);
        }
    }

    centroids.into_iter().filter(|v| v[3] > 0.0).collect()
}

