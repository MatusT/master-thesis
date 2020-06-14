use wgpu::*;

pub struct SphereBillboardsPipeline {
    pub pipeline: RenderPipeline,
    pub bind_group_layout: BindGroupLayout,
}

impl SphereBillboardsPipeline {
    pub fn new(device: &Device, sample_count: u32) -> Self {
        // Shaders
        let vs = include_bytes!("billboards.vert.spv");
        let vs_module =
            device.create_shader_module(&read_spirv(std::io::Cursor::new(&vs[..])).unwrap());
        let fs = include_bytes!("billboards.frag.spv");
        let fs_module =
            device.create_shader_module(&read_spirv(std::io::Cursor::new(&fs[..])).unwrap());

        // Bind Groups
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Sphere billbaord bind group layout"),
            bindings: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                    ty: BindingType::UniformBuffer { dynamic: false },
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::VERTEX,
                    ty: BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                    },
                },
            ],
        });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(RasterizationStateDescriptor {
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: PrimitiveTopology::TriangleList,
            color_states: &[ColorStateDescriptor {
                format: TextureFormat::Bgra8UnormSrgb,
                color_blend: BlendDescriptor::REPLACE,
                alpha_blend: BlendDescriptor::REPLACE,
                write_mask: ColorWrite::ALL,
            }],
            depth_stencil_state: Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil_front: StencilStateFaceDescriptor::IGNORE,
                stencil_back: StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            vertex_state: VertexStateDescriptor {
                index_format: IndexFormat::Uint16,
                vertex_buffers: &[],
            },
            sample_count,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn create_bind_group(
        &self,
        device: &Device,
        camera_buffer: &Buffer,
        spheres_buffer: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            bindings: &[
                Binding {
                    binding: 0,
                    resource: BindingResource::Buffer(camera_buffer.slice(0..!0)),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::Buffer(spheres_buffer.slice(0..!0)),
                },
            ],
        })
    }
}
