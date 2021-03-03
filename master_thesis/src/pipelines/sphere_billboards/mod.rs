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
        let vs_module = device.create_shader_module(&include_spirv!("billboards.vert.spv"));
        let fs_module = device.create_shader_module(&include_spirv!("billboards.frag.spv"));

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
            vertex: VertexState {
                module: &vs_module,
                entry_point: "main",
                buffers: &[],
            },
            primitive: PrimitiveState {
                cull_mode: Some(Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
                clamp_depth: false,
            }),
            fragment: Some(FragmentState {
                module: &fs_module,
                entry_point: "main",
                targets: &[
                    // Output color
                    ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        write_mask: ColorWrite::ALL,
                        blend: None,
                    },
                    // Instance
                    ColorTargetState {
                        format: TextureFormat::R32Uint,
                        write_mask: ColorWrite::ALL,
                        blend: None,
                    },
                ],
            }),
            multisample: wgpu::MultisampleState::default(),
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
        let vs_module = device.create_shader_module(&include_spirv!("billboards_debug.vert.spv"));
        let fs_module = device.create_shader_module(&include_spirv!("billboards_debug.frag.spv"));

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
            vertex: VertexState {
                module: &vs_module,
                entry_point: "main",
                buffers: &[],
            },
            primitive: PrimitiveState {
                cull_mode: Some(Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
                clamp_depth: false,
            }),
            fragment: Some(FragmentState {
                module: &fs_module,
                entry_point: "main",
                targets: &[
                    // Output color
                    ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        write_mask: ColorWrite::ALL,
                        blend: None,
                    },
                    // Instance
                    ColorTargetState {
                        format: TextureFormat::R32Uint,
                        write_mask: ColorWrite::ALL,
                        blend: None,
                    },
                    // Normals
                    ColorTargetState {
                        format: TextureFormat::Rgba32Float,
                        write_mask: ColorWrite::ALL,
                        blend: None,
                    },
                ],
            }),
            multisample: wgpu::MultisampleState::default(),
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
        let vs_module = device.create_shader_module(&include_spirv!("billboards.vert.spv"));
        let fs_module = device.create_shader_module(&include_spirv!("billboards_normals.frag.spv"));

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
            vertex: VertexState {
                module: &vs_module,
                entry_point: "main",
                buffers: &[],
            },
            primitive: PrimitiveState {
                cull_mode: Some(Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
                clamp_depth: false,
            }),
            fragment: Some(FragmentState {
                module: &fs_module,
                entry_point: "main",
                targets: &[
                    // Output color
                    ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        write_mask: ColorWrite::ALL,
                        blend: None,
                    },
                    // Instance
                    ColorTargetState {
                        format: TextureFormat::R32Uint,
                        write_mask: ColorWrite::ALL,
                        blend: None,
                    },
                    // Normals
                    ColorTargetState {
                        format: TextureFormat::Rgba32Float,
                        write_mask: ColorWrite::ALL,
                        blend: None,
                    },
                ],
            }),
            multisample: wgpu::MultisampleState::default(),
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
            device.create_shader_module(&include_spirv!("billboards_depth_write.vert.spv"))
        } else {
            device.create_shader_module(&include_spirv!("billboards_depth.vert.spv"))
        };
        let fs_module = if write_visibility {
            device.create_shader_module(&include_spirv!("billboards_depth_write.frag.spv"))
        } else {
            device.create_shader_module(&include_spirv!("billboards_depth.frag.spv"))
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

        let depth_stencil = if write_visibility {
            Some(DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
                clamp_depth: false,
            })
        } else {
            Some(DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
                clamp_depth: false,
            })
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &vs_module,
                entry_point: "main",
                buffers: &[],
            },
            primitive: PrimitiveState {
                cull_mode: Some(Face::Back),
                ..Default::default()
            },
            depth_stencil,
            fragment: Some(FragmentState {
                module: &fs_module,
                entry_point: "main",
                targets: &[],
            }),
            multisample: wgpu::MultisampleState::default(),
        });

        Self { pipeline }
    }
}
