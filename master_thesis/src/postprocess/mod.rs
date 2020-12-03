use bytemuck::*;
use nalgebra_glm::*;
use wgpu::*;
use crate::camera::*;

#[derive(Copy, Clone)]
pub struct PostProcessOptions {
    // Chroma
    pub chroma_amount: f32,

    // DOF
    pub dof: f32,
    pub focus: f32,
    pub focus_point: Vec2,
    pub focus_amount: f32,

    // Monotone
    pub gauss_amount: f32,

    pub ssao_pow: [f32; 2],
    pub fog: f32,
}

impl Default for PostProcessOptions {
    fn default() -> Self {
        PostProcessOptions {
            chroma_amount: 10.0,

            dof: 0.005,
            focus: 0.002,
            focus_point: vec2(0.5, 0.5),
            focus_amount: 2.0,

            gauss_amount: 4.0,

            ssao_pow: [1.0f32; 2],
            fog: 10000.0,
        }
    }
}

pub struct PostProcessModule {
    width: u32,
    height: u32,

    pub options: PostProcessOptions,

    chroma_pass: ComputePipeline,
    dof_pass: ComputePipeline,
    monotone_pass: ComputePipeline,
    contours_pass: ComputePipeline,
    gauss_pass: [ComputePipeline; 2],
    combine_pass: ComputePipeline,

    chroma_bgl: BindGroupLayout,
    dof_bgl: BindGroupLayout,
    contours_bgl: BindGroupLayout,
    combine_bgl: BindGroupLayout,

    linear_clamp_sampler: Sampler,

    pub temporary_texture: TextureView,
    pub bloom_texture: [TextureView; 2],
}

impl PostProcessModule {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let chroma_shader = device.create_shader_module(include_spirv!("chroma.comp.spv"));
        let dof_shader = device.create_shader_module(include_spirv!("dof.comp.spv"));
        let monotone_shader = device.create_shader_module(include_spirv!("mono-tone.comp.spv"));
        let contours_shader = device.create_shader_module(include_spirv!("contours.comp.spv"));
        let gaussx_shader = device.create_shader_module(include_spirv!("gaussx.comp.spv"));
        let gaussy_shader = device.create_shader_module(include_spirv!("gaussy.comp.spv"));
        let combine_shader = device.create_shader_module(include_spirv!("combine.comp.spv"));

        let chroma_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Chroma Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Sampler { comparison: false },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba8Unorm,
                        readonly: false,
                    },
                    count: None,
                },
            ],
        });

        let chroma_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Chroma Pipeline Layout"),
            bind_group_layouts: &[&chroma_bgl],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStage::COMPUTE,
                range: 0..16,
            }],
        });

        let chroma_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Chrome Pass"),
            layout: Some(&chroma_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &chroma_shader,
                entry_point: "main",
            },
        });

        let dof_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Depth of field Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Sampler { comparison: false },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba8Unorm,
                        readonly: false,
                    },
                    count: None,
                },
            ],
        });

        let dof_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Depth of field Pipeline Layout"),
            bind_group_layouts: &[&dof_bgl],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStage::COMPUTE,
                range: 0..28,
            }],
        });

        let dof_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Chrome Pass"),
            layout: Some(&dof_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &dof_shader,
                entry_point: "main",
            },
        });

        let monotone_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Monotone Pipeline Layout"),
            bind_group_layouts: &[&chroma_bgl],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStage::COMPUTE,
                range: 0..8,
            }],
        });

        let monotone_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Monotone Pass"),
            layout: Some(&monotone_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &monotone_shader,
                entry_point: "main",
            },
        });

        let contours_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Contours Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Sampler { comparison: false },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Uint,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba8Unorm,
                        readonly: false,
                    },
                    count: None,
                },
            ],
        });

        let contours_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("contours Pipeline Layout"),
            bind_group_layouts: &[&contours_bgl],
            push_constant_ranges: &[],
        });

        let contours_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("contours Pass"),
            layout: Some(&contours_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &contours_shader,
                entry_point: "main",
            },
        });

        let gauss_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("gaussx Pipeline Layout"),
            bind_group_layouts: &[&chroma_bgl],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStage::COMPUTE,
                range: 0..12
            }],
        });

        let gaussx_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("gaussx Pass"),
            layout: Some(&gauss_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &gaussx_shader,
                entry_point: "main",
            },
        });

        let gaussy_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("gaussy Pass"),
            layout: Some(&gauss_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &gaussy_shader,
                entry_point: "main",
            },
        });

        let gauss_pass = [gaussy_pass, gaussx_pass];

        let combine_bgl =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Combine bing group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStage::all(),
                        ty: BindingType::Sampler { comparison: false },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStage::all(),
                        ty: BindingType::SampledTexture {
                            dimension: TextureViewDimension::D2,
                            component_type: TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStage::all(),
                        ty: BindingType::SampledTexture {
                            dimension: TextureViewDimension::D2,
                            component_type: TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStage::all(),
                        ty: BindingType::SampledTexture {
                            dimension: TextureViewDimension::D2,
                            component_type: TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStage::all(),
                        ty: BindingType::SampledTexture {
                            dimension: TextureViewDimension::D2,
                            component_type: TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStage::all(),
                        ty: BindingType::StorageTexture {
                            dimension: TextureViewDimension::D2,
                            readonly: false,
                            format: TextureFormat::Rgba8Unorm,
                        },
                        count: None,
                    },
                ],
            });

        let combine_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Combine Pipeline Layout"),
            bind_group_layouts: &[&combine_bgl],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStage::COMPUTE,
                range: 0..28
            }],
        });

        let combine_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Combine Pass"),
            layout: Some(&combine_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &combine_shader,
                entry_point: "main",
            },
        });

        let linear_clamp_sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            ..Default::default()
        });

        let temporary_texture = device
            .create_texture(&TextureDescriptor {
                label: Some("Temporary texture"),
                size: Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::STORAGE | TextureUsage::SAMPLED,
            })
            .create_view(&TextureViewDescriptor::default());

        let bloom_texture = [
            device
                .create_texture(&TextureDescriptor {
                    label: Some("Bloom texture"),
                    size: Extent3d {
                        width,
                        height,
                        depth: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::STORAGE | TextureUsage::SAMPLED,
                })
                .create_view(&TextureViewDescriptor::default()),
            device
                .create_texture(&TextureDescriptor {
                    label: Some("Bloom texture"),
                    size: Extent3d {
                        width,
                        height,
                        depth: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::STORAGE | TextureUsage::SAMPLED,
                })
                .create_view(&TextureViewDescriptor::default()),
        ];

        Self {
            width,
            height,

            options: PostProcessOptions::default(),

            chroma_pass,
            dof_pass,
            monotone_pass,
            contours_pass,
            gauss_pass,
            combine_pass,

            chroma_bgl,
            dof_bgl,
            contours_bgl,
            combine_bgl,

            linear_clamp_sampler,
            temporary_texture,
            bloom_texture,
        }
    }

    pub fn compute(
        &mut self,
        camera: &mut dyn Camera,
        device: &Device,
        encoder: &mut CommandEncoder,
        color: &TextureView,
        depth: &TextureView,
        ssao: [&TextureView; 2],
        instances: &TextureView,
        time: f32,
    ) {
        let depth_unpack_mul = -camera.ubo().projection[(2, 3)];
        let mut depth_unpack_add = -camera.ubo().projection[(2, 2)];
        if depth_unpack_mul * depth_unpack_add < 0.0 {
            depth_unpack_add = -depth_unpack_add;
        }

        // Create bind groups
        let monotone_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.chroma_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.linear_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(color),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.bloom_texture[0]),
                },
            ],
        });

        let combine_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Combine bind group"),
            layout: &self.combine_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.linear_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(color),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(ssao[0]),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(ssao[1]),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(depth),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&self.temporary_texture),
                },
            ],
        });

        let gaussx_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.chroma_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.linear_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.bloom_texture[0]),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.bloom_texture[1]),
                },
            ],
        });

        let gaussy_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.chroma_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.linear_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.bloom_texture[1]),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.bloom_texture[0]),
                },
            ],
        });

        let contours_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.contours_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.linear_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.temporary_texture),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(instances),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(color),
                },
            ],
        });

        let dof_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.dof_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.linear_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(color),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(depth),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.temporary_texture),
                },
            ],
        });

        let chroma_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.chroma_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.linear_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(color),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.temporary_texture),
                },
            ],
        });

        let dispatch_size = |tile_size: u32, total_size: u32| -> u32 {
            return (total_size + tile_size - 1) / tile_size;
        };

        let x = dispatch_size(8, self.width);
        let y = dispatch_size(8, self.height);

        let mut cpass = encoder.begin_compute_pass();

        // Monotone
        // {
        //     cpass.set_pipeline(&self.monotone_pass);
        //     cpass.set_bind_group(0, &monotone_bg, &[]);
        //     cpass.set_push_constants(0, cast_slice(&[self.width as f32, self.height as f32]));
        //     cpass.dispatch(x, y, 1);

        //     cpass.set_pipeline(&self.gauss_pass[0]);
        //     cpass.set_bind_group(0, &gaussx_bg, &[]);
        //     cpass.set_push_constants(0, cast_slice(&[self.width as f32, self.height as f32, self.options.gauss_amount]));
        //     cpass.dispatch(x, y, 1);

        //     cpass.set_pipeline(&self.gauss_pass[1]);
        //     cpass.set_bind_group(0, &gaussy_bg, &[]);
        //     cpass.set_push_constants(0, cast_slice(&[self.width as f32, self.height as f32, self.options.gauss_amount]));
        //     cpass.dispatch(x, y, 1);
        // }


        // Combine
        {
            cpass.set_pipeline(&self.combine_pass);
            cpass.set_bind_group(0, &combine_bg, &[]);
            cpass.set_push_constants(
                0,
                cast_slice(&[self.width as f32, self.height as f32, depth_unpack_mul, depth_unpack_add, self.options.fog]),
            );
            cpass.set_push_constants(
                20,
                cast_slice(&[self.options.ssao_pow[0] as f32]),
            );
            cpass.set_push_constants(
                24,
                cast_slice(&[self.options.ssao_pow[0] as f32]),
            );
            cpass.dispatch(x, y, 1);
        }

        // Contours
        {
            cpass.set_pipeline(&self.contours_pass);
            cpass.set_bind_group(0, &contours_bg, &[]);
            cpass.dispatch(x, y, 1);
        }

        // Depth of field
        {
            cpass.set_pipeline(&self.dof_pass);
            cpass.set_bind_group(0, &dof_bg, &[]);
            cpass.set_push_constants(
                0,
                cast_slice(&[
                    self.width as f32,
                    self.height as f32,
                    self.options.focus_point.x,
                    self.options.focus_point.y,
                    self.options.dof,
                    self.options.focus,
                    self.options.focus_amount,
                ]),
            );
            cpass.dispatch(x, y, 1);
        }

        // // Chroma
        // {
        //     cpass.set_pipeline(&self.chroma_pass);
        //     cpass.set_bind_group(0, &chroma_bg, &[]);
        //     cpass.set_push_constants(
        //         0,
        //         cast_slice(&[
        //             self.width as f32,
        //             self.height as f32,
        //             time,
        //             self.options.chroma_amount,
        //         ]),
        //     );
        //     cpass.dispatch(x, y, 1);
        // }
    }
}