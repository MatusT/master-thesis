use bytemuck::*;
use nalgebra_glm::*;
use ron::de::from_str;
use wgpu;
use wgpu_experiments::camera::*;
use wgpu_experiments::kmeans::*;
use wgpu_experiments::pdb_loader;
use wgpu_experiments::pipelines::sphere_billboards::SphereBillboardPipeline;
use wgpu_experiments::rpdb;
use wgpu_experiments::{ApplicationEvent, ApplicationSkeleton};


pub struct Application {
    width: u32,
    height: u32,

    device: wgpu::Device,
    queue: wgpu::Queue,

    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,

    camera: RotationCamera,
    camera_buffer: wgpu::Buffer,

    billboards_pipeline: SphereBillboardPipeline,
    billboards_bind_group: wgpu::BindGroup,

    lods: Vec<std::ops::Range<u32>>,
}

impl Application {
    pub fn new(width: u32, height: u32, device: &wgpu::Device, surface: &wgpu::Surface) -> Self {
        let mut camera = RotationCamera::new(width as f32 / height as f32, 0.785398163, 0.1);
        let camera_buffer = device.create_buffer_with_data(
            cast_slice(&[camera.ubo()]),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width, height, depth: 1 },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });
        let depth_texture_view = depth_texture.create_default_view();

        let args: Vec<String> = std::env::args().collect();
        let molecule_file_path: &str = &args[1];
        let molecule_ron = std::fs::read_to_string(molecule_file_path).unwrap();
        let molecule: rpdb::Molecule = from_str(&molecule_ron).unwrap();

        let mut atoms = Vec::new();
        let mut lods: Vec<std::ops::Range<u32>> = Vec::new();
        let mut sum = 0u32;
        for lod in molecule.lods() {
            for atom in lod.atoms() {
                atoms.extend_from_slice(&[atom.x, atom.y, atom.z, atom.w]);
            }
            lods.push(sum * 3..(sum + lod.atoms().len() as u32) * 3);
            sum += lod.atoms().len() as u32;
        }

        camera.set_distance(distance(&molecule.bounding_box.min, &molecule.bounding_box.max));
        camera.set_speed(distance(&molecule.bounding_box.min, &molecule.bounding_box.max));

        let spheres_positions =
            device.create_buffer_with_data(cast_slice(&atoms), wgpu::BufferUsage::STORAGE_READ | wgpu::BufferUsage::COPY_DST);

        let billboards_pipeline = SphereBillboardPipeline::new(&device);
        let billboards_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &billboards_pipeline.bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &camera_buffer,
                        range: 0..std::mem::size_of::<CameraUbo>() as u64,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &spheres_positions,
                        range: 0..(atoms.len() * std::mem::size_of::<f32>()) as u64,
                    },
                },
            ],
        });

        Self {
            width,
            height,
            options,

            device,
            queue,

            depth_texture,
            depth_texture_view,

            camera,
            camera_buffer,

            billboards_pipeline,
            billboards_bind_group,

            lods,
        }
    }

    pub fn options(&self) -> &ApplicationOptions {
        &self.options
    }

    pub fn queue_mut(&mut self) -> &mut wgpu::Queue {
        &mut self.queue
    }
}

impl ApplicationSkeleton for Application {
    fn resize(&mut self, _: u32, _: u32) {
        //
    }

    fn update(&mut self, event: ApplicationEvent) {
        use winit::event::VirtualKeyCode;
        match event {
            ApplicationEvent::KeyboardInput { input, .. } => {
                if let Some(key) = input.virtual_keycode {
                    match key {
                        VirtualKeyCode::Numpad0 => {
                            self.options.selected_lod = 0;
                        }
                        VirtualKeyCode::Numpad1 => {
                            self.options.selected_lod = 1;
                        }
                        VirtualKeyCode::Numpad2 => {
                            self.options.selected_lod = 2;
                        }
                        VirtualKeyCode::Numpad3 => {
                            self.options.selected_lod = 3;
                        }
                        VirtualKeyCode::Numpad4 => {
                            self.options.selected_lod = 4;
                        }
                        VirtualKeyCode::Numpad5 => {
                            self.options.selected_lod = 5;
                        }
                        VirtualKeyCode::Numpad6 => {
                            self.options.selected_lod = 6;
                        }
                        _ => {}
                    };
                }
            }
            _ => {}
        }

        self.camera.update(event);
    }

    fn render(&mut self, frame: &wgpu::TextureView) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let size = std::mem::size_of::<CameraUbo>();
            let camera_buffer = self
                .device
                .create_buffer_with_data(cast_slice(&[self.camera.ubo()]), wgpu::BufferUsage::COPY_SRC);

            encoder.copy_buffer_to_buffer(&camera_buffer, 0, &self.camera_buffer, 0, size as wgpu::BufferAddress);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::WHITE,
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture_view,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 0.0,
                    clear_stencil: 0,
                }),
            });

            rpass.set_pipeline(&self.billboards_pipeline.pipeline);
            rpass.set_bind_group(0, &self.billboards_bind_group, &[]);
            rpass.draw(self.lods[self.options.selected_lod as usize].clone(), 0..1);
        }

        self.queue.submit(&[encoder.finish()]);
    }
}
