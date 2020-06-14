use crate::BoundingBox;

use nalgebra_glm::Vec4;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct MoleculeLod {
    max_radius: f32,

    #[serde(default)]
    breakpoint: f32,
    atoms: Vec<Vec4>,
}

impl MoleculeLod {
    pub fn new(atoms: Vec<Vec4>, breakpoint: f32) -> Self {
        let mut max_radius = atoms[0].w;

        for atom in &atoms {
            if atom.w > max_radius {
                max_radius = atom.w;
            }
        }

        Self { max_radius, breakpoint, atoms }
    }

    pub fn max_radius(&self) -> f32 {
        self.max_radius
    }

    pub fn atoms(&self) -> &[Vec4] {
        &self.atoms
    }

    pub fn breakpoint(&self) -> f32 {
        self.breakpoint
    }

    pub fn set_breakpoint(&mut self, breakpoint: f32) {
        self.breakpoint = breakpoint;
    }
}
#[derive(Serialize, Deserialize)]
pub struct Molecule {
    pub name: String,
    pub bounding_box: BoundingBox,
    pub lods: Vec<MoleculeLod>,
}

impl Molecule {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn bounding_box(&self) -> &BoundingBox {
        &self.bounding_box
    }

    pub fn lods(&self) -> &[MoleculeLod] {
        &self.lods
    }
}
