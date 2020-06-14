#[derive(Serialize, Deserialize)]
pub struct Structure {
    pub molecules: Vec<(String, Vec<Mat4>)>,
}