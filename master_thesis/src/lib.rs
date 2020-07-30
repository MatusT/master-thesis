pub mod camera;
pub mod hilbert;
pub mod pvs;
pub mod structure;
pub mod pipelines;

pub enum ApplicationEvent<'a> {
    WindowEvent(winit::event::WindowEvent<'a>),
    DeviceEvent(winit::event::DeviceEvent),
}