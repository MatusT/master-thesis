pub mod camera;
pub mod hilbert;
pub mod pipelines;
pub mod pvs;
pub mod ssao;
pub mod structure;

pub enum ApplicationEvent<'a> {
    WindowEvent(winit::event::WindowEvent<'a>),
    DeviceEvent(winit::event::DeviceEvent),
}
