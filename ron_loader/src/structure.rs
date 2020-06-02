#[derive(Serialize, Deserialize)]
pub struct Structure {
    pub names: Vec<String>,
    pub model_matrices: Vec<Mat4>,
}