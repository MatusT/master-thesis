use nalgebra_glm::Mat4;
use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
pub struct Structure {
    pub molecules: std::collections::HashMap<String, Vec<Mat4>>,
}
