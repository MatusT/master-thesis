use wgpu::*;
pub struct SphereBillboardsPipeline {
    pub pipeline: RenderPipeline,
}

impl SphereBillboardsPipeline {
    pub fn new(
        device: &Device,
        camera_bind_group_layout: &BindGroupLayout,
        per_molecule_bind_group_layout: &BindGroupLayout,
        per_structure_bind_group_layout: &BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        // Shaders
        let vs_module = device.create_shader_module(include_spirv!("billboards.vert.spv"));
        let fs_module = device.create_shader_module(include_spirv!("billboards.frag.spv"));

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &per_molecule_bind_group_layout,
                &per_structure_bind_group_layout,
            ],
            push_constant_ranges: &[
                PushConstantRange {
                    stages: ShaderStage::VERTEX,
                    range: 0..12,
                },
                PushConstantRange {
                    stages: ShaderStage::FRAGMENT,
                    range: 4..32,
                },
            ],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
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
                clamp_depth: false,
                polygon_mode: PolygonMode::Fill,
            }),
            primitive_topology: PrimitiveTopology::TriangleList,
            color_states: &[
                // Output color
                ColorStateDescriptor {
                    format: TextureFormat::Rgba8Unorm,
                    color_blend: BlendDescriptor::REPLACE,
                    alpha_blend: BlendDescriptor::REPLACE,
                    write_mask: ColorWrite::ALL,
                },
                // Instance
                ColorStateDescriptor {
                    format: TextureFormat::R32Uint,
                    color_blend: BlendDescriptor::REPLACE,
                    alpha_blend: BlendDescriptor::REPLACE,
                    write_mask: ColorWrite::ALL,
                },
            ],
            depth_stencil_state: Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: StencilStateDescriptor::default(),
            }),
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

    pub fn new_debug(
        device: &Device,
        camera_bind_group_layout: &BindGroupLayout,
        per_molecule_bind_group_layout: &BindGroupLayout,
        per_structure_bind_group_layout: &BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        // Shaders
        let vs_module = device.create_shader_module(include_spirv!("billboards_debug.vert.spv"));
        let fs_module = device.create_shader_module(include_spirv!("billboards_debug.frag.spv"));

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &per_molecule_bind_group_layout,
                &per_structure_bind_group_layout,
            ],
            push_constant_ranges: &[
                PushConstantRange {
                    stages: ShaderStage::VERTEX,
                    range: 0..12,
                },
                PushConstantRange {
                    stages: ShaderStage::FRAGMENT,
                    range: 4..32,
                },
            ],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
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
                clamp_depth: false,
                polygon_mode: PolygonMode::Fill,
            }),
            primitive_topology: PrimitiveTopology::TriangleList,
            color_states: &[
                // Output color
                ColorStateDescriptor {
                    format: TextureFormat::Rgba8Unorm,
                    color_blend: BlendDescriptor::REPLACE,
                    alpha_blend: BlendDescriptor::REPLACE,
                    write_mask: ColorWrite::ALL,
                },
                // Instance
                ColorStateDescriptor {
                    format: TextureFormat::R32Uint,
                    color_blend: BlendDescriptor::REPLACE,
                    alpha_blend: BlendDescriptor::REPLACE,
                    write_mask: ColorWrite::ALL,
                },
                // Normals
                ColorStateDescriptor {
                    format: TextureFormat::Rgba32Float,
                    color_blend: BlendDescriptor::REPLACE,
                    alpha_blend: BlendDescriptor::REPLACE,
                    write_mask: ColorWrite::ALL,
                },
            ],
            depth_stencil_state: Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: StencilStateDescriptor::default(),
            }),
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

    pub fn new_normals(
        device: &Device,
        camera_bind_group_layout: &BindGroupLayout,
        per_molecule_bind_group_layout: &BindGroupLayout,
        per_structure_bind_group_layout: &BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        // Shaders
        let vs_module = device.create_shader_module(include_spirv!("billboards.vert.spv"));
        let fs_module = device.create_shader_module(include_spirv!("billboards_normals.frag.spv"));

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &per_molecule_bind_group_layout,
                &per_structure_bind_group_layout,
            ],
            push_constant_ranges: &[
                PushConstantRange {
                    stages: ShaderStage::VERTEX,
                    range: 0..12,
                },
                PushConstantRange {
                    stages: ShaderStage::FRAGMENT,
                    range: 4..32,
                },
            ],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
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
                clamp_depth: false,
                polygon_mode: PolygonMode::Fill,
            }),
            primitive_topology: PrimitiveTopology::TriangleList,
            color_states: &[
                // Output color
                ColorStateDescriptor {
                    format: TextureFormat::Rgba8Unorm,
                    color_blend: BlendDescriptor::REPLACE,
                    alpha_blend: BlendDescriptor::REPLACE,
                    write_mask: ColorWrite::ALL,
                },
                // Instance
                ColorStateDescriptor {
                    format: TextureFormat::R32Uint,
                    color_blend: BlendDescriptor::REPLACE,
                    alpha_blend: BlendDescriptor::REPLACE,
                    write_mask: ColorWrite::ALL,
                },
                // Normals
                ColorStateDescriptor {
                    format: TextureFormat::Rgba32Float,
                    color_blend: BlendDescriptor::REPLACE,
                    alpha_blend: BlendDescriptor::REPLACE,
                    write_mask: ColorWrite::ALL,
                },
            ],
            depth_stencil_state: Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: StencilStateDescriptor::default(),
            }),
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

pub struct SphereBillboardsDepthPipeline {
    pub pipeline: RenderPipeline,
}

impl SphereBillboardsDepthPipeline {
    pub fn new(
        device: &Device,
        camera_bind_group_layout: &BindGroupLayout,
        per_molecule_bind_group_layout: &BindGroupLayout,
        per_visibility_bind_group_layout: Option<&BindGroupLayout>,
        sample_count: u32,
        write_visibility: bool,
    ) -> Self {
        // Shaders
        let vs_module = if write_visibility {
            device.create_shader_module(include_spirv!("billboards_depth_write.vert.spv"))
        } else {
            device.create_shader_module(include_spirv!("billboards_depth.vert.spv"))
        };
        let fs_module = if write_visibility {
            device.create_shader_module(include_spirv!("billboards_depth_write.frag.spv"))
        } else {
            device.create_shader_module(include_spirv!("billboards_depth.frag.spv"))
        };

        // Pipeline
        let bind_group_layouts = if write_visibility {
            vec![
                camera_bind_group_layout,
                per_molecule_bind_group_layout,
                per_visibility_bind_group_layout.unwrap(),
            ]
        } else {
            vec![camera_bind_group_layout, per_molecule_bind_group_layout]
        };

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        });

        let depth_stencil_state = if write_visibility {
            Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilStateDescriptor::default(),
            })
        } else {
            Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: StencilStateDescriptor::default(),
            })
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
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
                clamp_depth: false,
                polygon_mode: PolygonMode::Fill,
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
