use wgpu::*;
pub struct SphereBillboardsPipeline {
    pub pipeline: RenderPipeline,
    pub per_molecule_bind_group_layout: BindGroupLayout,
}

impl SphereBillboardsPipeline {
    pub fn new(
        device: &Device,
        camera_bind_group_layout: &BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        // Shaders
        let vs_module = device.create_shader_module(include_spirv!("billboards.vert.spv"));
        let fs_module = device.create_shader_module(include_spirv!("billboards.frag.spv"));

        // Bind group layouts
        let per_molecule_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Molecule bind group layout"),
                bindings: &[
                    BindGroupLayoutEntry::new(
                        0,
                        ShaderStage::VERTEX,
                        BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: NonZeroBufferAddress::new(16),
                        },
                    ),
                    BindGroupLayoutEntry::new(
                        1,
                        ShaderStage::VERTEX,
                        BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: NonZeroBufferAddress::new(64),
                        },
                    ),
                ],
            });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&camera_bind_group_layout, &per_molecule_bind_group_layout],
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
            per_molecule_bind_group_layout,
        }
    }
}

pub struct SphereBillboardsDepthPipeline {
    pub pipeline: RenderPipeline,
}

impl SphereBillboardsDepthPipeline {
    pub fn new(
        device: &Device,
        camera_bind_group_layout: &BindGroupLayout,
        per_molecule_bind_group_layout: &BindGroupLayout,
        sample_count: u32,
        write_visibility: bool,
    ) -> Self {
        // Shaders
        let vs_module = device.create_shader_module(include_spirv!("billboards_depth.vert.spv"));
        let fs_module = if write_visibility {
            device.create_shader_module(include_spirv!("billboards_depth_write.frag.spv"))
        } else {
            device.create_shader_module(include_spirv!("billboards_depth.frag.spv"))
        };

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&camera_bind_group_layout, &per_molecule_bind_group_layout],
        });

        let depth_stencil_state = if write_visibility {
            Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: CompareFunction::GreaterEqual,
                stencil_front: StencilStateFaceDescriptor::IGNORE,
                stencil_back: StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            })
        } else {
            Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil_front: StencilStateFaceDescriptor::IGNORE,
                stencil_back: StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            })
        };

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
            color_states: &[],
            depth_stencil_state,
            vertex_state: VertexStateDescriptor {
                index_format: IndexFormat::Uint16,
                vertex_buffers: &[],
            },
            sample_count,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self { pipeline }
    }
}
