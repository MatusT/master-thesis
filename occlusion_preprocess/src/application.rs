use crate::biological_structure::*;
use crate::camera::*;
use crate::pipelines::{LinesPipeline, SphereBillboardsPipeline};
use crate::ApplicationEvent;

use nalgebra_glm::reversed_infinite_perspective_rh_zo;
use wgpu::*;

use std::rc::Rc;

pub struct Application {
    width: u32,
    height: u32,

    device: Device,
    queue: Queue,

    depth_texture: TextureView,
    multisampled_texture: TextureView,

    camera: RotationCamera,
    camera_bind_group_layout: BindGroupLayout,

    billboards_pipeline: SphereBillboardsPipeline,

    pub recalculate: bool,

    structure: Rc<Structure>,
    pvs_module: Rc<StructurePvsModule>,
    pvs_field: StructurePvsField,
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
                label: Some("Camera bind group layout"),
                bindings: &[BindGroupLayoutEntry::new(
                    0,
                    ShaderStage::all(),
                    BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: Some(CameraUbo::size()),
                    },
                )],
            });

        let per_molecule_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Molecule bind group layout"),
                bindings: &[
                    BindGroupLayoutEntry::new(
                        0,
                        ShaderStage::all(),
                        BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: None,
                        },
                    ),
                    BindGroupLayoutEntry::new(
                        1,
                        ShaderStage::all(),
                        BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: None,
                        },
                    ),
                ],
            });

        // Camera
        let camera = RotationCamera::new(
            &device,
            &camera_bind_group_layout,
            &reversed_infinite_perspective_rh_zo(width as f32 / height as f32, 0.785398163, 0.1),
            1500.0,
            100.0,
        );

        // Data
        let args: Vec<String> = std::env::args().collect();

        let structure = Rc::new(Structure::from_ron(
            &device,
            &args[1],
            &per_molecule_bind_group_layout,
        ));

        // Pipelines
        let billboards_pipeline = SphereBillboardsPipeline::new(
            &device,
            &camera_bind_group_layout,
            &per_molecule_bind_group_layout,
            sample_count,
        );

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

        let pvs_module = Rc::new(StructurePvsModule::new(
            &device,
            &camera_bind_group_layout,
            &per_molecule_bind_group_layout,
        ));
        let pvs_field = pvs_module.pvs_field(
            &device,
            &camera_bind_group_layout,
            structure.clone(),
            2,
            1024,
        );

        Self {
            width,
            height,

            device,
            queue,

            depth_texture,
            multisampled_texture,

            camera,
            camera_bind_group_layout,

            billboards_pipeline,

            recalculate: true,

            structure: structure.clone(),
            pvs_module: pvs_module.clone(),
            pvs_field,
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
                color_attachments: &[RenderPassColorAttachmentDescriptor {
                    attachment: &frame,
                    resolve_target: None,
                    // attachment: &self.multisampled_texture,
                    // resolve_target: Some(&frame),
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: Color::WHITE,
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture,
                    depth_load_op: LoadOp::Clear,
                    depth_store_op: StoreOp::Store,
                    stencil_load_op: LoadOp::Load,
                    stencil_store_op: StoreOp::Store,
                    clear_depth: 0.0,
                    clear_stencil: 0,
                    stencil_read_only: true,
                    depth_read_only: false,
                }),
            });

            rpass.set_pipeline(&self.billboards_pipeline.pipeline);
            rpass.set_bind_group(0, &self.camera.bind_group(), &[]);

            let distance = self.camera.distance() - self.structure.bounding_radius();

            // self.structure.draw_lod(&mut rpass, distance);
            self.pvs_field.draw(
                &self.device,
                &self.queue,
                &mut rpass,
                self.camera.direction_vector(),
            );
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
