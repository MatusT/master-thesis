use bytemuck::*;
use nalgebra_glm as glm;
use wgpu::*;
use winit;

use crate::ApplicationEvent;
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CameraUbo {
    pub projection: glm::Mat4,
    pub view: glm::Mat4,
    pub projection_view: glm::Mat4,
    pub position: glm::Vec4,
}

impl CameraUbo {
    pub fn size() -> std::num::NonZeroU64 {
        std::num::NonZeroU64::new(std::mem::size_of::<Self>() as u64)
            .expect("CameraUbo can't be zero.")
    }
}

unsafe impl Zeroable for CameraUbo {}
unsafe impl Pod for CameraUbo {}
pub trait Camera {
    fn update(&mut self, event: &ApplicationEvent);
    fn update_gpu(&mut self, queue: &Queue);

    fn ubo(&mut self) -> CameraUbo;

    fn bind_group(&self) -> &BindGroup;

    fn set_projection(&mut self, projection: &glm::Mat4);

    fn speed(&self) -> f32;
    fn set_speed(&mut self, speed: f32);
}

pub struct RotationCamera {
    ubo: CameraUbo,

    pub yaw: f32,
    pub pitch: f32,
    distance: f32,

    speed: f32,
    mouse_pressed: bool,

    buffer: Buffer,
    bind_group: BindGroup,
}

impl RotationCamera {
    pub fn new(
        device: &Device,
        bind_group_layout: &BindGroupLayout,
        projection: &glm::Mat4,
        distance: f32,
        speed: f32,
    ) -> RotationCamera {
        let eye = distance * glm::vec3(distance, 0.0, 0.0);
        let view = glm::look_at(&eye, &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 1.0, 0.0));
        let projection_view = projection * view;
        let position = glm::vec4(eye[0], eye[1], eye[2], 1.0);

        let ubo = CameraUbo {
            projection: *projection,
            view,
            projection_view,
            position,
        };

        let buffer = device.create_buffer_with_data(
            cast_slice(&[ubo]),
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        );

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[Binding {
                binding: 0,
                resource: BindingResource::Buffer(buffer.slice(..)),
            }],
            label: None,
        });

        RotationCamera {
            ubo,

            yaw: 0.0,
            pitch: 0.0,

            distance,
            speed,

            mouse_pressed: false,

            buffer,
            bind_group,
        }
    }

    pub fn direction_vector(&self) -> glm::Vec3 {
        let yaw = self.yaw.to_radians();
        let pitch = self.pitch.to_radians();

        glm::vec3(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        )
    }

    pub fn distance(&self) -> f32 {
        self.distance
    }

    pub fn set_distance(&mut self, distance: f32) {
        self.distance = distance;
    }
}

impl Camera for RotationCamera {
    fn update<'a>(&mut self, event: &ApplicationEvent<'a>) {
        match event {
            ApplicationEvent::WindowEvent(event) => match event {
                winit::event::WindowEvent::MouseWheel { delta, .. } => {
                    if let winit::event::MouseScrollDelta::LineDelta(_, change) = delta {
                        self.distance -= change * self.speed;
                    }
                }
                winit::event::WindowEvent::MouseInput { state, button, .. } => {
                    if *button == winit::event::MouseButton::Left {
                        if *state == winit::event::ElementState::Pressed {
                            self.mouse_pressed = true;
                        } else {
                            self.mouse_pressed = false;
                        }
                    }
                }
                _ => {}
            },
            ApplicationEvent::DeviceEvent(event) => match event {
                winit::event::DeviceEvent::MouseMotion { delta: (x, y) } => {
                    if self.mouse_pressed {
                        self.yaw += *x as f32;
                        self.pitch += *y as f32;
                    }
                }
                _ => {}
            },
        };
    }

    fn update_gpu(&mut self, queue: &Queue) {
        let ubo = self.ubo();
        queue.write_buffer(&self.buffer, 0, cast_slice(&[ubo]));
    }

    fn ubo(&mut self) -> CameraUbo {
        let eye = self.distance * self.direction_vector();
        self.ubo.view = glm::look_at(&eye, &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 1.0, 0.0));
        self.ubo.projection_view = self.ubo.projection * self.ubo.view;
        self.ubo.position = glm::vec4(eye.x, eye.y, eye.z, 1.0);

        self.ubo
    }

    fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }

    fn set_projection(&mut self, projection: &glm::Mat4) {
        self.ubo.projection = *projection;
    }

    fn speed(&self) -> f32 {
        self.speed
    }

    fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }
}
