use bytemuck::*;
use nalgebra_glm::{clamp_scalar, vec2, Mat4, TVec2, Vec2, Vec4};
use wgpu::util::DeviceExt;
use wgpu::*;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Constants {
    normals_to_viewspace: Mat4,

    depth_buffer_offset: TVec2<i32>,
    depth_unpack_consts: Vec2,

    ndc_to_view_mul: Vec2,
    ndc_to_view_add: Vec2,

    depth_buffer_uv_to_view_mul: Vec2,
    depth_buffer_uv_to_view_add: Vec2,

    EffectRadius: f32,         // world (viewspace) maximum size of the shadow
    EffectShadowStrength: f32, // global strength of the effect (0 - 5)
    EffectShadowPow: f32,
    EffectShadowClamp: f32,

    EffectFadeOutMul: f32,            // effect fade out from distance (ex. 25)
    EffectFadeOutAdd: f32,            // effect fade out to distance   (ex. 100)
    EffectHorizonAngleThreshold: f32, // limit errors on slopes and caused by insufficient geometry tessellation (0.05 to 0.5)
    EffectSamplingRadiusNearLimitRec: f32, // if viewspace pixel closer than this, don't enlarge shadow sampling radius anymore (makes no sense to grow beyond some distance, not enough samples to cover everything, so just limit the shadow growth: f32, could be SSAOSettingsFadeOutFrom * 0.1 or less)

    DepthPrecisionOffsetMod: f32,
    NegRecEffectRadius: f32, // -1.0 / EffectRadius
    DetailAOStrength: f32,
    padd0: f32,

    SSAOBufferDimensions: Vec2,
    SSAOBufferInverseDimensions: Vec2,

    DepthBufferDimensions: Vec2,
    DepthBufferInverseDimensions: Vec2,

    OutputBufferDimensions: Vec2,
    OutputBufferInverseDimensions: Vec2,

    DeinterleavedDepthBufferDimensions: Vec2,
    DeinterleavedDepthBufferInverseDimensions: Vec2,

    Deinterleaveddepth_buffer_offset: Vec2,
    DeinterleavedDepthBufferNormalisedOffset: Vec2,

    inv_sharpness: f32,
    blur_passes: u32,
}

unsafe impl Zeroable for Constants {}
unsafe impl Pod for Constants {}

impl Constants {
    pub fn size() -> std::num::NonZeroU64 {
        std::num::NonZeroU64::new(std::mem::size_of::<Self>() as u64)
            .expect("Constants can't be zero.")
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct PerPassConstants {
    pub PatternRotScaleMatrices: [Vec4; 5],
    pub PassIndex: i32,
}

unsafe impl Zeroable for PerPassConstants {}
unsafe impl Pod for PerPassConstants {}

impl PerPassConstants {
    pub fn size() -> std::num::NonZeroU64 {
        std::num::NonZeroU64::new(std::mem::size_of::<Self>() as u64)
            .expect("PerPassConstants can't be zero.")
    }
}

#[derive(Copy, Clone, Debug)]
struct BufferSizeInfo {
    inputOutputBufferWidth: u32,
    inputOutputBufferHeight: u32,

    ssaoBufferWidth: u32,
    ssaoBufferHeight: u32,

    depthBufferXOffset: u32,
    depthBufferYOffset: u32,

    depthBufferWidth: u32,
    depthBufferHeight: u32,

    deinterleavedDepthBufferXOffset: u32,
    deinterleavedDepthBufferYOffset: u32,

    deinterleavedDepthBufferWidth: u32,
    deinterleavedDepthBufferHeight: u32,

    importanceMapWidth: u32,
    importanceMapHeight: u32,
}

impl BufferSizeInfo {
    pub fn new(width: u32, height: u32) -> Self {
        let half_width = (width + 1) / 2;
        let half_height = (height + 1) / 2;
        let quarter_width = (half_width + 1) / 2;
        let quarter_height = (half_height + 1) / 2;

        let depthBufferWidth = width;
        let depthBufferHeight = height;
        let depthBufferHalfWidth = half_width;
        let depthBufferHalfHeight = half_height;

        let depthBufferXOffset = 0;
        let depthBufferYOffset = 0;
        let depthBufferHalfXOffset = 0;
        let depthBufferHalfYOffset = 0;

        Self {
            inputOutputBufferWidth: width,
            inputOutputBufferHeight: height,

            depthBufferXOffset,
            depthBufferYOffset,
            depthBufferWidth,
            depthBufferHeight,

            ssaoBufferWidth: half_width,
            ssaoBufferHeight: half_height,

            deinterleavedDepthBufferXOffset: depthBufferHalfXOffset,
            deinterleavedDepthBufferYOffset: depthBufferHalfYOffset,
            deinterleavedDepthBufferWidth: depthBufferHalfWidth,
            deinterleavedDepthBufferHeight: depthBufferHalfHeight,

            importanceMapWidth: quarter_width,
            importanceMapHeight: quarter_height,
        }
    }
}

pub struct Settings {
    ///
    pub projection: Mat4,

    ///
    pub normals_to_viewspace: Mat4,

    /// [0.0,  ~ ] World (view) space size of the occlusion sphere.
    pub radius: f32,

    /// [0.0, 5.0] Effect strength linear multiplier
    pub shadowMultiplier: f32,

    /// [0.5, 5.0] Effect strength pow modifier
    pub shadowPower: f32,

    /// [0.0, 1.0] Effect max limit (applied after multiplier but before blur)
    pub shadowClamp: f32,

    /// [0.0, 0.2] Limits self-shadowing (makes the sampling area less of a hemisphere, more of a spherical cone, to avoid self-shadowing and various artifacts due to low tessellation and depth buffer imprecision, etc.)
    pub horizonAngleThreshold: f32,

    /// [0.0,  ~ ] Distance to start start fading out the effect.
    pub fadeOutFrom: f32,

    /// [0.0,  ~ ] Distance at which the effect is faded out.
    pub fadeOutTo: f32,

    /// [  0,   8] Number of edge-sensitive smart blur passes to apply
    pub blurPassCount: u32,

    /// [0.0, 1.0] (How much to bleed over edges: 1: not at all, 0.5: half-half: f32, 0.0: completely ignore edges)
    pub sharpness: f32,

    /// [0.0, 5.0] Used for high-res detail AO using neighboring depth pixels: adds a lot of detail but also reduces temporal stability (adds aliasing).
    pub detailShadowStrength: f32,

    /// This option should be set to true if FidelityFX-CACAO should reconstruct a normal buffer from the depth buffer. It is required to be true if no normal buffer is provided.
    pub generateNormals: bool,

    /// [0.0,  ~ ] Sigma squared value for use in bilateral upsampler giving Gaussian blur term. Should be greater than 0.0.
    pub bilateralSigmaSquared: f32,

    /// [0.0,  ~ ] Sigma squared value for use in bilateral upsampler giving similarity weighting for neighbouring pixels. Should be greater than 0.0.
    pub bilateralSimilarityDistanceSigma: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            projection: Mat4::identity(),
            normals_to_viewspace: Mat4::identity(),
            radius: 1.2,
            shadowMultiplier: 1.0,
            shadowPower: 1.5,
            shadowClamp: 0.98,
            horizonAngleThreshold: 0.06,
            fadeOutFrom: 50.0,
            fadeOutTo: 300.0,
            blurPassCount: 2,
            sharpness: 0.98,
            detailShadowStrength: 0.5,
            generateNormals: false,
            bilateralSigmaSquared: 5.0,
            bilateralSimilarityDistanceSigma: 0.01,
        }
    }
}
pub struct SsaoModule {
    width: u32,
    height: u32,

    buffer_size_info: BufferSizeInfo,

    normals_from_depth_pass: ComputePipeline,
    prepare_normals_pass: ComputePipeline,
    prepare_normals_bgl: BindGroupLayout,

    prepare_depths_pass: ComputePipeline,
    prepare_depths_bgl: BindGroupLayout,

    ssao_pass: ComputePipeline,
    ssao_bgl: BindGroupLayout,
    ssao_per_pass_bgl: BindGroupLayout,

    blur_pass: ComputePipeline,
    blur_bgl: BindGroupLayout,
    blur_per_pass_bgl: BindGroupLayout,

    apply_pass: ComputePipeline,
    apply_bgl: BindGroupLayout,

    point_clamp_sampler: Sampler,
    point_mirror_sampler: Sampler,
    linear_clamp_sampler: Sampler,
    viewspace_depth_sampler: Sampler,

    deinterlaced_normals: TextureView,

    halfdepths_mips: Vec<TextureView>,
    halfdepths_arrays: Vec<TextureView>,

    ssao_results_views: Vec<TextureView>,

    blurred_results: Texture,
    blurred_results_views: Vec<TextureView>,

    constants: Constants,
    constants_buffer: Buffer,

    pass_constants: [PerPassConstants; 4],
    pass_constants_buffers: Vec<Buffer>,
}

impl SsaoModule {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let buffer_size_info = BufferSizeInfo::new(width, height);

        let normals_from_depth_shader =
            device.create_shader_module(include_spirv!("normals_from_depth.comp.spv"));
        let prepare_normals_shader =
            device.create_shader_module(include_spirv!("prepare_normals.comp.spv"));
        let prepare_depths_shader =
            device.create_shader_module(include_spirv!("prepare_depths.comp.spv"));
        let ssao_shader = device.create_shader_module(include_spirv!("ssao.comp.spv"));
        let blur_shader = device.create_shader_module(include_spirv!("blur.comp.spv"));
        let apply_shader = device.create_shader_module(include_spirv!("apply.comp.spv"));

        let prepare_normals_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Prepare normals BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::all(),
                    ty: BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: Some(Constants::size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Sampler { comparison: false },
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
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::Rgba32Float,
                        readonly: false,
                    },
                    count: None,
                },
            ],
        });

        let prepare_normals_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&prepare_normals_bgl],
            push_constant_ranges: &[],
        });
        let normals_from_depth_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&prepare_normals_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &normals_from_depth_shader,
                entry_point: "main",
            },
        });
        let prepare_normals_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&prepare_normals_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &prepare_normals_shader,
                entry_point: "main",
            },
        });

        let prepare_depths_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Prepare depths BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::all(),
                    ty: BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: Some(Constants::size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Sampler { comparison: false },
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
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        readonly: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        readonly: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        readonly: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        readonly: false,
                    },
                    count: None,
                },
            ],
        });
        let prepare_depths_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&prepare_depths_bgl],
            push_constant_ranges: &[],
        });
        let prepare_depths_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&prepare_depths_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &prepare_depths_shader,
                entry_point: "main",
            },
        });

        let ssao_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("SSAO BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::all(),
                    ty: BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: Some(Constants::size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Sampler { comparison: false },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Sampler { comparison: false },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        readonly: false,
                    },
                    count: None,
                },
            ],
        });
        let ssao_per_pass_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("SSAO PER PASS BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::all(),
                    ty: BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: Some(PerPassConstants::size()),
                    },
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
                        format: TextureFormat::Rg32Float,
                        readonly: false,
                    },
                    count: None,
                },
            ],
        });
        let ssao_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&ssao_bgl, &ssao_per_pass_bgl],
            push_constant_ranges: &[],
        });
        let ssao_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&ssao_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &ssao_shader,
                entry_point: "main",
            },
        });

        let blur_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BLUR BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::all(),
                    ty: BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: Some(Constants::size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Sampler { comparison: false },
                    count: None,
                },
            ],
        });
        let blur_per_pass_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BLUR PER PASS BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rg32Float,
                        readonly: false,
                    },
                    count: None,
                },
            ],
        });
        let blur_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&blur_bgl, &blur_per_pass_bgl],
            push_constant_ranges: &[],
        });
        let blur_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&blur_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &blur_shader,
                entry_point: "main",
            },
        });

        let apply_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("APPLY BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::all(),
                    ty: BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: Some(Constants::size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Sampler { comparison: false },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Sampler { comparison: false },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2Array,
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2,
                        format: TextureFormat::R32Float,
                        readonly: false,
                    },
                    count: None,
                },
            ],
        });
        let apply_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&apply_bgl],
            push_constant_ranges: &[],
        });
        let apply_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&apply_pl),
            compute_stage: ProgrammableStageDescriptor {
                module: &apply_shader,
                entry_point: "main",
            },
        });

        let point_clamp_sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            ..Default::default()
        });

        let point_mirror_sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::MirrorRepeat,
            address_mode_v: AddressMode::MirrorRepeat,
            address_mode_w: AddressMode::MirrorRepeat,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            ..Default::default()
        });

        let viewspace_depth_sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::MirrorRepeat,
            address_mode_v: AddressMode::MirrorRepeat,
            address_mode_w: AddressMode::MirrorRepeat,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            ..Default::default()
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

        let deinterlaced_normals = device
            .create_texture(&TextureDescriptor {
                label: Some("Deinterlaced Normals"),
                size: Extent3d {
                    width: (width + 1) / 2,
                    height: (height + 1) / 2,
                    depth: 4,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsage::STORAGE | TextureUsage::SAMPLED,
            })
            .create_view(&TextureViewDescriptor {
                format: Some(TextureFormat::Rgba32Float),
                dimension: Some(TextureViewDimension::D2Array),
                aspect: TextureAspect::All,
                ..Default::default()
            });

        let halfdepths = device.create_texture(&TextureDescriptor {
            label: Some("Half depths"),
            size: Extent3d {
                width: (width + 1) / 2,
                height: (height + 1) / 2,
                depth: 4,
            },
            mip_level_count: 4,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsage::STORAGE | TextureUsage::SAMPLED,
        });

        let ssao_results = device.create_texture(&TextureDescriptor {
            label: Some("Final SSAO"),
            size: Extent3d {
                width: (width + 1) / 2,
                height: (height + 1) / 2,
                depth: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg8Unorm,
            usage: TextureUsage::STORAGE | TextureUsage::SAMPLED,
        });

        let blurred_results = device.create_texture(&TextureDescriptor {
            label: Some("Blurred SSAO"),
            size: Extent3d {
                width: (width + 1) / 2,
                height: (height + 1) / 2,
                depth: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg8Unorm,
            usage: TextureUsage::STORAGE | TextureUsage::SAMPLED,
        });

        let mut halfdepths_mips = Vec::new();
        let mut halfdepths_arrays = Vec::new();
        let mut ssao_results_views = Vec::new();
        let mut blurred_results_views = Vec::new();
        for pass in 0..4 {
            halfdepths_mips.push(halfdepths.create_view(&TextureViewDescriptor {
                format: Some(TextureFormat::R32Float),
                dimension: Some(TextureViewDimension::D2Array),
                aspect: TextureAspect::All,
                base_mip_level: pass,
                level_count: std::num::NonZeroU32::new(1),
                base_array_layer: 0,
                array_layer_count: None,
                ..Default::default()
            }));

            halfdepths_arrays.push(halfdepths.create_view(&TextureViewDescriptor {
                format: Some(TextureFormat::R32Float),
                dimension: Some(TextureViewDimension::D2Array),
                aspect: TextureAspect::All,
                base_mip_level: 0,
                level_count: None,
                base_array_layer: pass,
                array_layer_count: std::num::NonZeroU32::new(1),
                ..Default::default()
            }));

            ssao_results_views.push(ssao_results.create_view(&TextureViewDescriptor {
                format: Some(TextureFormat::Rg8Unorm),
                dimension: Some(TextureViewDimension::D2Array),
                aspect: TextureAspect::All,
                base_array_layer: pass,
                array_layer_count: std::num::NonZeroU32::new(1),
                ..Default::default()
            }));

            blurred_results_views.push(blurred_results.create_view(&TextureViewDescriptor {
                format: Some(TextureFormat::Rg8Unorm),
                dimension: Some(TextureViewDimension::D2),
                aspect: TextureAspect::All,
                base_array_layer: pass,
                array_layer_count: std::num::NonZeroU32::new(1),
                ..Default::default()
            }));
        }

        let constants = Constants::default();
        let constants_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: cast_slice(&[constants]),
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        });

        let mut pass_constants = [PerPassConstants::default(); 4];
        let mut pass_constants_buffers = Vec::new();
        for i in 0..4 {
            pass_constants[i].PassIndex = i as i32;
            pass_constants_buffers.push(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: cast_slice(&[pass_constants[i]]),
                    usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                },
            ));
        }

        println!("{:?}", buffer_size_info);

        Self {
            width,
            height,

            buffer_size_info,

            normals_from_depth_pass,
            prepare_normals_pass,
            prepare_normals_bgl,

            prepare_depths_pass,
            prepare_depths_bgl,

            ssao_pass,
            ssao_bgl,
            ssao_per_pass_bgl,

            blur_pass,
            blur_bgl,
            blur_per_pass_bgl,

            apply_pass,
            apply_bgl,

            point_clamp_sampler,
            point_mirror_sampler,
            viewspace_depth_sampler,
            linear_clamp_sampler,

            deinterlaced_normals,

            halfdepths_mips,
            halfdepths_arrays,

            ssao_results_views,

            blurred_results,
            blurred_results_views,

            constants,
            constants_buffer,

            pass_constants,
            pass_constants_buffers,
        }
    }

    fn update_constants(&mut self, queue: &Queue, settings: &Settings) {
        let mut constants = &mut self.constants;

        constants.normals_to_viewspace = settings.normals_to_viewspace;
        let depthLinearizeMul = -settings.projection[(2, 3)];
        let mut depthLinearizeAdd = -settings.projection[(2, 2)];
        if depthLinearizeMul * depthLinearizeAdd <= 0.0 {
            depthLinearizeAdd = -depthLinearizeAdd;
        }
        constants.depth_unpack_consts[0] = depthLinearizeMul;
        constants.depth_unpack_consts[1] = depthLinearizeAdd;

        let CameraTanHalfFOV = vec2(
            1.0 / settings.projection[(1, 1)],
            1.0 / settings.projection[(0, 0)],
        );

        constants.ndc_to_view_mul[0] = CameraTanHalfFOV[0] * 2.0;
        constants.ndc_to_view_mul[1] = CameraTanHalfFOV[1] * -2.0;
        constants.ndc_to_view_add[0] = CameraTanHalfFOV[0] * -1.0;
        constants.ndc_to_view_add[1] = CameraTanHalfFOV[1] * 1.0;

        let ratio = (self.buffer_size_info.inputOutputBufferWidth as f32)
            / (self.buffer_size_info.depthBufferWidth as f32);
        let border = (1.0 - ratio) / 2.0;
        for i in 0..2 {
            constants.depth_buffer_uv_to_view_mul[i] = constants.ndc_to_view_mul[i] / ratio;
            constants.depth_buffer_uv_to_view_add[i] =
                constants.ndc_to_view_add[i] - constants.ndc_to_view_mul[i] * border / ratio;
        }

        constants.EffectRadius = clamp_scalar(settings.radius, 0.0, 100000.0);
        constants.EffectShadowStrength = clamp_scalar(settings.shadowMultiplier * 4.3, 0.0, 10.0);
        constants.EffectShadowPow = clamp_scalar(settings.shadowPower, 0.0, 10.0);
        constants.EffectShadowClamp = clamp_scalar(settings.shadowClamp, 0.0, 1.0);
        constants.EffectFadeOutMul = -1.0 / (settings.fadeOutTo - settings.fadeOutFrom);
        constants.EffectFadeOutAdd =
            settings.fadeOutFrom / (settings.fadeOutTo - settings.fadeOutFrom) + 1.0;
        constants.EffectHorizonAngleThreshold =
            clamp_scalar(settings.horizonAngleThreshold, 0.0, 1.0);

        // 1.2 seems to be around the best trade off - 1.0 means on-screen radius will stop/slow growing when the camera is at 1.0 distance, so, depending on FOV, basically filling up most of the screen
        // This setting is viewspace-dependent and not screen size dependent intentionally, so that when you change FOV the effect stays (relatively) similar.
        let mut effectSamplingRadiusNearLimit = settings.radius * 1.2;
        effectSamplingRadiusNearLimit /= CameraTanHalfFOV[1]; // to keep the effect same regardless of FOV

        // if the depth precision is switched to 32bit float, this can be set to something closer to 1 (0.9999 is fine)
        constants.DepthPrecisionOffsetMod = 0.9999;

        constants.EffectSamplingRadiusNearLimitRec = 1.0 / effectSamplingRadiusNearLimit;

        constants.NegRecEffectRadius = -1.0 / constants.EffectRadius;

        constants.DetailAOStrength = settings.detailShadowStrength;

        // set buffer size constants.
        constants.SSAOBufferDimensions[0] = self.buffer_size_info.ssaoBufferWidth as f32;
        constants.SSAOBufferDimensions[1] = self.buffer_size_info.ssaoBufferHeight as f32;
        constants.SSAOBufferInverseDimensions[0] =
            1.0 / (self.buffer_size_info.ssaoBufferWidth as f32);
        constants.SSAOBufferInverseDimensions[1] =
            1.0 / (self.buffer_size_info.ssaoBufferHeight as f32);

        constants.DepthBufferDimensions[0] = self.buffer_size_info.depthBufferWidth as f32;
        constants.DepthBufferDimensions[1] = self.buffer_size_info.depthBufferHeight as f32;
        constants.DepthBufferInverseDimensions[0] =
            1.0 / (self.buffer_size_info.depthBufferWidth as f32);
        constants.DepthBufferInverseDimensions[1] =
            1.0 / (self.buffer_size_info.depthBufferHeight as f32);

        constants.OutputBufferDimensions[0] = self.buffer_size_info.depthBufferWidth as f32;
        constants.OutputBufferDimensions[1] = self.buffer_size_info.depthBufferHeight as f32;
        constants.OutputBufferInverseDimensions[0] =
            1.0 / (self.buffer_size_info.depthBufferWidth as f32);
        constants.OutputBufferInverseDimensions[1] =
            1.0 / (self.buffer_size_info.depthBufferHeight as f32);

        constants.DeinterleavedDepthBufferDimensions[0] =
            self.buffer_size_info.deinterleavedDepthBufferWidth as f32;
        constants.DeinterleavedDepthBufferDimensions[1] =
            self.buffer_size_info.deinterleavedDepthBufferHeight as f32;
        constants.DeinterleavedDepthBufferInverseDimensions[0] =
            1.0 / (self.buffer_size_info.deinterleavedDepthBufferWidth as f32);
        constants.DeinterleavedDepthBufferInverseDimensions[1] =
            1.0 / (self.buffer_size_info.deinterleavedDepthBufferHeight as f32);

        constants.Deinterleaveddepth_buffer_offset[0] =
            self.buffer_size_info.deinterleavedDepthBufferXOffset as f32;
        constants.Deinterleaveddepth_buffer_offset[1] =
            self.buffer_size_info.deinterleavedDepthBufferYOffset as f32;
        constants.DeinterleavedDepthBufferNormalisedOffset[0] =
            (self.buffer_size_info.deinterleavedDepthBufferXOffset as f32)
                / (self.buffer_size_info.deinterleavedDepthBufferWidth as f32);
        constants.DeinterleavedDepthBufferNormalisedOffset[1] =
            (self.buffer_size_info.deinterleavedDepthBufferYOffset as f32)
                / (self.buffer_size_info.deinterleavedDepthBufferHeight as f32);

        constants.blur_passes = settings.blurPassCount;

        queue.write_buffer(&self.constants_buffer, 0, cast_slice(&[self.constants]));

        for pass in 0..4 {
            let constants = &mut self.pass_constants[pass];

            let subpass_count = 5;
            let spmap = [0, 1, 4, 3, 2];
            for subpass in 0..subpass_count {
                let a = pass as f32;
                let b = spmap[subpass] as f32;

                let angle0: f32 =
                    (a + b / subpass_count as f32) * (3.1415926535897932384626433832795) * 0.5;

                let ca: f32 = angle0.cos();
                let sa: f32 = angle0.sin();

                let scale: f32 = 1.0
                    + (a - 1.5 + (b - (subpass_count as f32 - 1.0) * 0.5) / subpass_count as f32)
                        * 0.07;

                constants.PatternRotScaleMatrices[subpass][0] = scale * ca;
                constants.PatternRotScaleMatrices[subpass][1] = scale * -sa;
                constants.PatternRotScaleMatrices[subpass][2] = -scale * sa;
                constants.PatternRotScaleMatrices[subpass][3] = -scale * ca;
            }

            queue.write_buffer(
                &self.pass_constants_buffers[pass],
                0,
                cast_slice(&[self.pass_constants[pass]]),
            );
        }
    }

    pub fn compute(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        settings: &Settings,
        depth: &TextureView,
        normals: Option<&TextureView>,
        final_result: &TextureView,
    ) {       
        self.update_constants(&queue, settings);

        // Create bind groups
        let prepare_depths_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.prepare_depths_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &self.constants_buffer,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.point_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&depth),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.halfdepths_mips[0]),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&self.halfdepths_mips[1]),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&self.halfdepths_mips[2]),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&self.halfdepths_mips[3]),
                },
            ],
        });

        let (prepare_normals_input, prepare_normals_pass) = if let Some(normals) = normals {
            (normals, &self.prepare_normals_pass)
        } else {
            (depth, &self.normals_from_depth_pass)
        };

        let prepare_normals_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.prepare_normals_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &self.constants_buffer,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.point_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&prepare_normals_input),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.deinterlaced_normals),
                },
            ],
        });

        let ssao_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.ssao_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &self.constants_buffer,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.point_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&self.viewspace_depth_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.deinterlaced_normals),
                },
            ],
        });

        let mut ssao_pass_bgs = Vec::new();
        for pass in 0..4 as usize {
            ssao_pass_bgs.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.ssao_per_pass_bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: &self.pass_constants_buffers[pass],
                            offset: 0,
                            size: None,
                        },
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&self.halfdepths_arrays[pass]),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(&self.ssao_results_views[pass]),
                    },
                ],
            }));
        }

        let blur_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.blur_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &self.constants_buffer,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.point_mirror_sampler),
                },
            ],
        });
        let mut blur_pass_bgs = Vec::new();
        for pass in 0..4 {
            blur_pass_bgs.push(device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.blur_per_pass_bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&self.ssao_results_views[pass]),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&self.blurred_results_views[pass]),
                    },
                ],
            }));
        }

        let apply_bgs = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.apply_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &self.constants_buffer,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.point_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&self.linear_clamp_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(
                        &self
                            .blurred_results
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&final_result),
                },
            ],
        });

        let dispatch_size = |tile_size: u32, total_size: u32| -> u32 {
            return (total_size + tile_size - 1) / tile_size;
        };

        let mut cpass = encoder.begin_compute_pass();

        // Prepare depths
        {
            cpass.set_pipeline(&self.prepare_depths_pass);
            cpass.set_bind_group(0, &prepare_depths_bg, &[]);

            let x = dispatch_size(8, self.buffer_size_info.deinterleavedDepthBufferWidth);
            let y = dispatch_size(8, self.buffer_size_info.deinterleavedDepthBufferHeight);
            cpass.dispatch(x, y, 1);
        }

        // Prepare normals from input normals
        cpass.set_pipeline(prepare_normals_pass);
        cpass.set_bind_group(0, &prepare_normals_bg, &[]);

        let x = dispatch_size(8, self.buffer_size_info.ssaoBufferWidth);
        let y = dispatch_size(8, self.buffer_size_info.ssaoBufferHeight);
        cpass.dispatch(x, y, 1);

        // SSAO
        cpass.set_pipeline(&self.ssao_pass);
        cpass.set_bind_group(0, &ssao_bg, &[]);
        for pass in 0..4 {
            cpass.set_bind_group(1, &ssao_pass_bgs[pass], &[]);

            let x = dispatch_size(8, self.buffer_size_info.ssaoBufferWidth);
            let y = dispatch_size(8, self.buffer_size_info.ssaoBufferHeight);
            cpass.dispatch(x, y, 1);
        }

        // Blur
        let blur_pass_count = settings.blurPassCount;
        let w = 4 * 16 - 2 * blur_pass_count;
        let h = 3 * 16 - 2 * blur_pass_count;
        let x = dispatch_size(w, self.buffer_size_info.ssaoBufferWidth);
        let y = dispatch_size(h, self.buffer_size_info.ssaoBufferHeight);

        cpass.set_pipeline(&self.blur_pass);
        cpass.set_bind_group(0, &blur_bg, &[]);
        for pass in 0..4 as usize {
            cpass.set_bind_group(1, &blur_pass_bgs[pass], &[]);
            cpass.dispatch(x, y, 1);
        }

        // Combine
        {
            cpass.set_pipeline(&self.apply_pass);
            cpass.set_bind_group(0, &apply_bgs, &[]);

            let x = dispatch_size(8, self.buffer_size_info.inputOutputBufferWidth);
            let y = dispatch_size(8, self.buffer_size_info.inputOutputBufferHeight);
            cpass.dispatch(x, y, 1);
        }
    }
}
