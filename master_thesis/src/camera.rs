use bytemuck::*;
use nalgebra_glm as glm;
use wgpu::util::DeviceExt;
use wgpu::*;
use winit;

use crate::*;

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
    fn window_event(&mut self, event: &winit::event::WindowEvent);
    fn device_event(&mut self, event: &winit::event::DeviceEvent);

    fn update_gpu(&mut self, queue: &Queue);

    fn ubo(&mut self) -> CameraUbo;

    fn bind_group(&self) -> &BindGroup;

    fn set_projection(&mut self, projection: &glm::Mat4);

    fn speed(&self) -> f32;
    fn set_speed(&mut self, speed: f32);
}
pub struct RotationCamera {
    pub ubo: CameraUbo,

    yaw: f64,
    pitch: f64,
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
        let yaw = 0.0f64;
        let pitch = 90.0f64.to_radians();

        let eye = distance * glm::vec3(1.0, 0.0, 0.0);
        let view = glm::look_at(&eye, &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 1.0, 0.0));
        let projection_view = projection * view;
        let position = glm::vec4(eye[0], eye[1], eye[2], 1.0);

        let ubo = CameraUbo {
            projection: *projection,
            view,
            projection_view,
            position,
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: cast_slice(&[ubo]),
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &buffer,
                    offset: 0,
                    size: None,
                },
            }],
            label: None,
        });

        RotationCamera {
            ubo,

            yaw,
            pitch,

            distance,
            speed,

            mouse_pressed: false,

            buffer,
            bind_group,
        }
    }

    pub fn eye(&self) -> glm::TVec3<f64> {
        self.distance as f64 * self.direction_vector()
    }

    pub fn direction_vector(&self) -> glm::TVec3<f64> {
        let yaw = self.yaw;
        let pitch = self.pitch;

        spherical_to_cartesian(&vec2(yaw, pitch))
    }

    pub fn distance(&self) -> f32 {
        self.distance
    }

    pub fn set_distance(&mut self, distance: f32) {
        self.distance = distance;
    }

    pub fn yaw(&self) -> f64 {
        self.yaw
    }

    pub fn set_yaw(&mut self, yaw: f64) {
        // assert!(yaw >= 0.0);
        // assert!(yaw <= TAU);

        self.yaw = yaw;
    }

    pub fn add_yaw(&mut self, yaw_delta: f64) {
        self.set_yaw(self.yaw + yaw_delta);
    }

    pub fn pitch(&self) -> f64 {
        self.pitch
    }

    pub fn set_pitch(&mut self, pitch: f64) {
        // assert!(pitch >= 0.0);
        // assert!(pitch <= TAU);

        self.pitch = pitch;
    }

    pub fn add_pitch(&mut self, pitch_delta: f64) {
        let result = self.pitch + pitch_delta;
        if result <= std::f64::consts::PI && result >= 1.0f64 {
            self.set_pitch(result);
        }
    }
}

impl Camera for RotationCamera {
    fn window_event(&mut self, event: &winit::event::WindowEvent) {
        match event {
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
        };
    }

    fn device_event(&mut self, event: &winit::event::DeviceEvent) {
        match event {
            winit::event::DeviceEvent::MouseMotion { delta: (x, y) } => {
                if self.mouse_pressed {
                    self.add_yaw(*x as f64 / 100.0f64);
                    self.add_pitch(*y as f64 / 100.0f64);
                }
            }
            _ => {}
        };
    }

    fn update_gpu(&mut self, queue: &Queue) {
        let ubo = self.ubo();
        queue.write_buffer(&self.buffer, 0, cast_slice(&[ubo]));
    }

    fn ubo(&mut self) -> CameraUbo {
        let eye = vec3(
            self.eye().x as f32,
            self.eye().y as f32,
            self.eye().z as f32,
        );
        self.ubo.view = glm::look_at_rh(&eye, &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 1.0, 0.0));
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
