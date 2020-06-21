use wgpu::*;

/// Biological structure, like SARS-Cov-19
pub struct MolecularStructure {
    /// Buffers containing atoms of a molecule
    pub molecules: Vec<(Buffer, u32)>,

    /// Transforms (Rotation, Translation) of molecules on a GPU
    pub transforms: Vec<(Buffer, u32)>,

    /// Bind groups for each molecule type containing reference to `molecules` and `transforms`
    pub bind_groups: Vec<BindGroup>,
}

/// `BiologicalStructure`'s potentially visible set
pub struct MolecularStructurePvs<'a> {
    parent: &'a MolecularStructure,

    /// Ranges
    pub visible: Vec<Vec<(u32, u32)>>,
}

impl<'a> MolecularStructurePvs<'a> {
    pub fn reduce(limit: u32) {}
}

pub fn compress_ranges(list: Vec<(u32, u32)>, threshold: u32) -> Vec<(u32, u32)> {
    if list.is_empty() {
        return Vec::new();
    }

    let mut new_list = Vec::new();

    let mut start = list[0].0;
    for (index, range) in list.iter().enumerate() {
        if index >= list.len() - 1 {
            new_list.push((start, range.1));
            break;
        }

        if list[index + 1].0 - range.1 > threshold {
            new_list.push((start, range.1));
            start = list[index + 1].0;
        }
    }

    new_list
}

pub fn reduce_visible(
    molecules_atoms: &[u32],
    mut molecules_ranges: Vec<Vec<(u32, u32)>>,
    limit: usize,
) -> Vec<Vec<(u32, u32)>> {
    let mut gaps = vec![Vec::new(); molecules_ranges.len()];

    // Find the gaps
    for (molecule_index, ranges) in molecules_ranges.iter().enumerate() {
        for range_index in 1..ranges.len() {
            let distance = ranges[range_index].0 - ranges[range_index - 1].1;

            if distance > 0 {
                gaps[molecule_index].push((ranges[range_index - 1].1, ranges[range_index].0));
            }
        }

        // Sort the gaps by their length in decreasing order
        gaps[molecule_index].sort_by(|a, b| {
            let a_distance = a.1 - a.0;
            let b_distance = b.1 - b.1;

            b_distance.cmp(&a_distance)
        });
    }

    // Run the greedy staff
    let mut ranges_num: usize = molecules_ranges.iter().map(|v| v.len()).sum();
    // While we are not under imposed limit
    while ranges_num > limit {
        // Run through gaps of each molecule and find the gap with smallest cost (distance * number of atoms)
        let mut min_gap = None;
        let mut min_cost = u32::MAX;
        let mut min_index = 0;
        for (molecule_index, gaps) in gaps.iter().enumerate() {
            if let Some(gap) = gaps.last() {
                let cost = (gap.1 - gap.0) * molecules_atoms[molecule_index];

                if cost < min_cost {
                    min_gap = Some(gap);
                    min_cost = cost;
                    min_index = molecule_index;
                }
            }
        }

        if let Some(gap) = min_gap {
            molecules_ranges[min_index].push(*gap);
            gaps[min_index].pop();
            ranges_num -= 1;
        } else {
            panic!("Somethin went wrong");
        }
    }

    for i in 0..molecules_ranges.len() {
        molecules_ranges[i].sort_by(|a, b| a.0.cmp(&b.0));
        molecules_ranges[i] = compress_ranges(molecules_ranges[i].clone(), 0);
    }

    molecules_ranges
}

#[cfg(test)]
mod tests {
    #[test]
    fn compress_ranges() {
        use super::compress_ranges;

        let input = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let output = vec![(0, 4)];
        assert_eq!(compress_ranges(input, 0), output);

        let input = vec![(34, 35), (35, 36), (36, 39)];
        let output = vec![(34, 39)];
        assert_eq!(compress_ranges(input, 0), output);
    }
}
