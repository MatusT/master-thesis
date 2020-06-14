use crate::camera::*;
use crate::pipelines::SphereBillboardsPipeline;
use crate::ApplicationEvent;

use bytemuck::*;
use nalgebra_glm::*;
use rpdb::*;
use wgpu::*;

pub struct Application {
    width: u32,
    height: u32,

    device: Device,
    queue: Queue,

    depth_texture: TextureView,
    multisampled_texture: TextureView,

    camera: RotationCamera,
    camera_buffer: Buffer,

    billboards_pipeline: SphereBillboardsPipeline,

    /// Array of molecules. Each element contains GPU buffer of atoms of a molecule.
    molecules: Vec<Buffer>,

    /// Array of structure's molecules. Each element contains GPU buffer of model matrices of one type of molecule.
    structure: Vec<Buffer>,
}

impl Application {
    pub fn new(width: u32, height: u32, device: Device, queue: Queue, swapchain_format: TextureFormat, sample_count: u32) -> Self {
        let mut camera = RotationCamera::new(width as f32 / height as f32, 0.785398163, 0.1);
        let camera_buffer = device.create_buffer_with_data(
            cast_slice(&[camera.ubo()]),
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        );

        let depth_texture = device.create_texture(&TextureDescriptor {
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
        });
        let depth_texture = depth_texture.create_default_view();

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

        let args: Vec<String> = std::env::args().collect();
        
        let structure_file = structure::Structure::from_ron(&args[1]);
        let mut molecules = Vec::new();
        let mut structure = Vec::new();
        for (molecule_name, molecule_model_matrices) in structure_file.molecules {
            let molecule = molecule::Molecule::from_ron(std::path::Path::new(&args[1]).with_file_name(molecule_name));
            let atoms = {
                let atoms = molecule.lods()[0].atoms();
                let mut atoms_flat: Vec<f32> = Vec::new();
                for atom in atoms {
                    atoms_flat.extend_from_slice(atom.into());
                }

                atoms_flat
            };            
            molecules.push(device.create_buffer_with_data(cast_slice(&atoms), BufferUsage::STORAGE));

            let molecule_model_matrices = {
                let mut matrices_flat: Vec<f32> = Vec::new();
                for molecule_model_matrix in molecule_model_matrices {
                    for i in 0..16 {
                        matrices_flat.push(molecule_model_matrix[i]);
                    }
                }

                matrices_flat
            };
            structure.push(device.create_buffer_with_data(cast_slice(&molecule_model_matrices), BufferUsage::STORAGE));
        }

        // camera.set_distance(distance(
        //     &molecule.bounding_box.min,
        //     &molecule.bounding_box.max,
        // ));
        // camera.set_speed(distance(&molecule.bounding_box.min, &molecule.bounding_box.max) / 10.0);

        let billboards_pipeline = SphereBillboardsPipeline::new(&device, 8);

        Self {
            width,
            height,

            device,
            queue,

            depth_texture,
            multisampled_texture,

            camera,
            camera_buffer,

            billboards_pipeline,

            molecules,
            structure,
        }
    }

    pub fn resize(&mut self, _: u32, _: u32) {
        //
    }

    pub fn update<'a>(&mut self, event: &ApplicationEvent<'a>) {
        self.camera.update(event);
    }

    pub fn render(&mut self, frame: &TextureView) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
            let size = std::mem::size_of::<CameraUbo>();
            let camera_buffer = self
                .device
                .create_buffer_with_data(cast_slice(&[self.camera.ubo()]), BufferUsage::COPY_SRC);

            encoder.copy_buffer_to_buffer(
                &camera_buffer,
                0,
                &self.camera_buffer,
                0,
                size as BufferAddress,
            );
        }

        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[RenderPassColorAttachmentDescriptor {
                    attachment: &self.multisampled_texture,
                    resolve_target: Some(&frame),
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: Color::WHITE,
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture,
                    depth_load_op: LoadOp::Clear,
                    depth_store_op: StoreOp::Store,
                    stencil_load_op: LoadOp::Clear,
                    stencil_store_op: StoreOp::Store,
                    clear_depth: 0.0,
                    clear_stencil: 0,
                    stencil_read_only: true,
                    depth_read_only: false,
                }),
            });

            rpass.set_pipeline(&self.billboards_pipeline.pipeline);
            // rpass.set_bind_group(0, &self.billboards_bind_group, &[]);

            // for i in 0..self.lods.len() {
            //     if (i == self.lods.len() - 1)
            //         || (self.camera.distance() > self.lods[i].0
            //             && self.camera.distance() < self.lods[i + 1].0)
            //     {
            //         println!(
            //             "Choosing LOD: {} with {} of spheres",
            //             i,
            //             (self.lods[i].1.end - self.lods[i].1.start) / 3
            //         );
            //         rpass.draw(self.lods[i].1.clone(), 0..1);
            //         break;
            //     }
            // }
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
