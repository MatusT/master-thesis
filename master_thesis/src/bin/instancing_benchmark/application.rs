use master_thesis::camera::*;
use master_thesis::pipelines::SphereBillboardsPipeline;
use master_thesis::ApplicationEvent;

use bytemuck::cast_slice;
use nalgebra_glm::*;
use wgpu::*;

use std::borrow::Cow::Borrowed;

struct ApplicationState {}

pub struct Application {
    width: u32,
    height: u32,

    device: Device,
    queue: Queue,

    state: ApplicationState,

    depth_texture: TextureView,
    multisampled_texture: TextureView,

    camera: RotationCamera,
}

impl Application {
    pub fn new(
        width: u32,
        height: u32,
        device: Device,
        queue: Queue,
        swapchain_format: TextureFormat,
        sample_count: u32,
    ) -> Self {
        // Shared bind group layouts
        let camera_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(Borrowed("Camera bind group layout")),
                entries: Borrowed(&[BindGroupLayoutEntry::new(
                    0,
                    ShaderStage::all(),
                    BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: Some(CameraUbo::size()),
                    },
                )]),
            });

        // Camera
        let camera = RotationCamera::new(
            &device,
            &camera_bind_group_layout,
            &reversed_infinite_perspective_rh_zo(width as f32 / height as f32, 0.785398163, 0.1),
            1500.0,
            100.0,
        );

        // Pipelines

        // Default framebuffer
        let depth_texture = device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::OUTPUT_ATTACHMENT,
            })
            .create_default_view();

        let multisampled_texture = device
            .create_texture(&TextureDescriptor {
                size: Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: swapchain_format,
                usage: TextureUsage::OUTPUT_ATTACHMENT,
                label: None,
            })
            .create_default_view();

        let state = ApplicationState {};

        Self {
            width,
            height,

            device,
            queue,

            state,

            depth_texture,
            multisampled_texture,

            camera,
        }
    }

    pub fn resize(&mut self, _: u32, _: u32) {
        //
    }

    pub fn update<'a>(&mut self, event: &ApplicationEvent<'a>) {
        self.camera.update(event);
    }

    pub fn render(&mut self, frame: &TextureView) {
        //================== CAMERA DATA UPLOAD
        self.camera.update_gpu(&self.queue);

        //================== RENDER MOLECULES
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: Borrowed(&[RenderPassColorAttachmentDescriptor {
                    attachment: &frame,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::WHITE),
                        store: true,
                    },
                }]),
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
