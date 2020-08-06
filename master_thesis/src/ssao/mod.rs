use bytemuck::*;
use nalgebra_glm::{Mat4, TVec2, Vec2, Vec4};
use wgpu::*;

use std::borrow::Cow::Borrowed;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Constants {
    NormalsWorldToViewspaceMatrix: Mat4,

    DepthBufferOffset: TVec2<i32>,
    DepthUnpackConsts: Vec2,

    NDCToViewMul: Vec2,
    NDCToViewAdd: Vec2,

    DepthBufferUVToViewMul: Vec2,
    DepthBufferUVToViewAdd: Vec2,

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
    PassIndex: i32,

    PatternRotScaleMatrices: [Vec4; 5],

    SSAOBufferDimensions: Vec2,
    SSAOBufferInverseDimensions: Vec2,

    DepthBufferDimensions: Vec2,
    DepthBufferInverseDimensions: Vec2,

    OutputBufferDimensions: Vec2,
    OutputBufferInverseDimensions: Vec2,

    DeinterleavedDepthBufferDimensions: Vec2,
    DeinterleavedDepthBufferInverseDimensions: Vec2,

    DeinterleavedDepthBufferOffset: Vec2,
    DeinterleavedDepthBufferNormalisedOffset: Vec2,
}

impl Constants {
    pub fn size() -> std::num::NonZeroU64 {
        std::num::NonZeroU64::new(std::mem::size_of::<Self>() as u64)
            .expect("Constants can't be zero.")
    }
}

unsafe impl Zeroable for Constants {}
unsafe impl Pod for Constants {}

struct ScreenSizeInfo {
    width: u32,
    height: u32,
    halfWidth: u32,
    halfHeight: u32,
    quarterWidth: u32,
    quarterHeight: u32,
    eighthWidth: u32,
    eighthHeight: u32,
    depthBufferWidth: u32,
    depthBufferHeight: u32,
    depthBufferHalfWidth: u32,
    depthBufferHalfHeight: u32,
    depthBufferQuarterWidth: u32,
    depthBufferQuarterHeight: u32,
    depthBufferOffsetX: u32,
    depthBufferOffsetY: u32,
    depthBufferHalfOffsetX: u32,
    depthBufferHalfOffsetY: u32,
}

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

struct Settings {
    radius: f32,
    /// [0.0,  ~ ] World (view) space size of the occlusion sphere.
    shadowMultiplier: f32,
    /// [0.0, 5.0] Effect strength linear multiplier
    shadowPower: f32,
    /// [0.5, 5.0] Effect strength pow modifier
    shadowClamp: f32,
    /// [0.0, 1.0] Effect max limit (applied after multiplier but before blur)
    horizonAngleThreshold: f32,
    /// [0.0, 0.2] Limits self-shadowing (makes the sampling area less of a hemisphere, more of a spherical cone, to avoid self-shadowing and various artifacts due to low tessellation and depth buffer imprecision, etc.)
    fadeOutFrom: f32,
    /// [0.0,  ~ ] Distance to start start fading out the effect.
    fadeOutTo: f32,
    /// [0.0,  ~ ] Distance at which the effect is faded out.
    blurPassCount: u32,
    /// [  0,   8] Number of edge-sensitive smart blur passes to apply
    sharpness: f32,
    /// [0.0, 1.0] (How much to bleed over edges: 1: not at all, 0.5: half-half: f32, 0.0: completely ignore edges)
    detailShadowStrength: f32,
    /// [0.0, 5.0] Used for high-res detail AO using neighboring depth pixels: adds a lot of detail but also reduces temporal stability (adds aliasing).
    generateNormals: bool,
    /// This option should be set to true if FidelityFX-CACAO should reconstruct a normal buffer from the depth buffer. It is required to be true if no normal buffer is provided.
    bilateralSigmaSquared: f32,
    /// [0.0,  ~ ] Sigma squared value for use in bilateral upsampler giving Gaussian blur term. Should be greater than 0.0.
    /// [0.0,  ~ ] Sigma squared value for use in bilateral upsampler giving similarity weighting for neighbouring pixels. Should be greater than 0.0.
    bilateralSimilarityDistanceSigma: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
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

    prepare_normals_pass: ComputePipeline,
    prepare_normals_bgl: BindGroupLayout,

    prepare_depths_pass: ComputePipeline,
    prepare_depths_bgl: BindGroupLayout,

    ssao_pass: ComputePipeline,
    ssao_bgl: BindGroupLayout,

    point_clamp_sampler: Sampler,
    point_mirror_sampler: Sampler,
    viewspace_depth_sampler: Sampler,

    deinterlaced_normals: Texture,
    halfdepths: Vec<Texture>,
    final_results: Vec<Texture>,
}

impl SsaoModule {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let prepare_normals_shader =
            device.create_shader_module(include_spirv!("prepare_normals.comp.spv"));
        let prepare_depths_shader =
            device.create_shader_module(include_spirv!("prepare_depths.comp.spv"));
        let ssao_shader = device.create_shader_module(include_spirv!("ssao.comp.spv"));

        let constants_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(Borrowed("Constants BGL")),
            entries: Borrowed(&[BindGroupLayoutEntry::new(
                0,
                ShaderStage::all(),
                BindingType::UniformBuffer {
                    dynamic: false,
                    min_binding_size: Some(Constants::size()),
                },
            )]),
        });

        let prepare_normals_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(Borrowed("Prepare normals BGL")),
            entries: Borrowed(&[
                BindGroupLayoutEntry::new(
                    0,
                    ShaderStage::COMPUTE,
                    BindingType::Sampler { comparison: false },
                ),
                BindGroupLayoutEntry::new(
                    1,
                    ShaderStage::COMPUTE,
                    BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                    },
                ),
                BindGroupLayoutEntry::new(
                    2,
                    ShaderStage::COMPUTE,
                    BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::Rgba32Float,
                        readonly: false,
                    },
                ),
            ]),
        });
        let prepare_normals_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: Borrowed(&[&constants_bgl, &prepare_normals_bgl]),
            push_constant_ranges: Borrowed(&[]),
        });
        let prepare_normals_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            layout: &prepare_normals_pl,
            compute_stage: ProgrammableStageDescriptor {
                module: &prepare_normals_shader,
                entry_point: Borrowed("main"),
            },
        });

        let prepare_depths_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(Borrowed("Prepare depths BGL")),
            entries: Borrowed(&[
                BindGroupLayoutEntry::new(
                    0,
                    ShaderStage::COMPUTE,
                    BindingType::Sampler { comparison: false },
                ),
                BindGroupLayoutEntry::new(
                    1,
                    ShaderStage::COMPUTE,
                    BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                    },
                ),
                BindGroupLayoutEntry::new(
                    2,
                    ShaderStage::COMPUTE,
                    BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        readonly: false,
                    },
                ),
                BindGroupLayoutEntry::new(
                    3,
                    ShaderStage::COMPUTE,
                    BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        readonly: false,
                    },
                ),
                BindGroupLayoutEntry::new(
                    4,
                    ShaderStage::COMPUTE,
                    BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        readonly: false,
                    },
                ),
                BindGroupLayoutEntry::new(
                    5,
                    ShaderStage::COMPUTE,
                    BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        readonly: false,
                    },
                ),
            ]),
        });
        let prepare_depths_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: Borrowed(&[&constants_bgl, &prepare_depths_bgl]),
            push_constant_ranges: Borrowed(&[]),
        });
        let prepare_depths_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            layout: &prepare_depths_pl,
            compute_stage: ProgrammableStageDescriptor {
                module: &prepare_depths_shader,
                entry_point: Borrowed("main"),
            },
        });

        let ssao_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(Borrowed("SSAO BGL")),
            entries: Borrowed(&[
                BindGroupLayoutEntry::new(
                    0,
                    ShaderStage::COMPUTE,
                    BindingType::Sampler { comparison: false },
                ),
                BindGroupLayoutEntry::new(
                    1,
                    ShaderStage::COMPUTE,
                    BindingType::Sampler { comparison: false },
                ),
                BindGroupLayoutEntry::new(
                    2,
                    ShaderStage::COMPUTE,
                    BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                    },
                ),
                BindGroupLayoutEntry::new(
                    3,
                    ShaderStage::COMPUTE,
                    BindingType::SampledTexture {
                        dimension: TextureViewDimension::D2Array,
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                    },
                ),
                BindGroupLayoutEntry::new(
                    4,
                    ShaderStage::COMPUTE,
                    BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rg32Float,
                        readonly: false,
                    },
                ),
            ]),
        });
        let ssao_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: Borrowed(&[&constants_bgl, &ssao_bgl]),
            push_constant_ranges: Borrowed(&[]),
        });
        let ssao_pass = device.create_compute_pipeline(&ComputePipelineDescriptor {
            layout: &prepare_depths_pl,
            compute_stage: ProgrammableStageDescriptor {
                module: &ssao_shader,
                entry_point: Borrowed("main"),
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

        let deinterlaced_normals = device.create_texture(&TextureDescriptor {
            label: Some(Borrowed("Deinterlaced Normals")),
            size: Extent3d {
                width,
                height,
                depth: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Snorm,
            usage: TextureUsage::STORAGE | TextureUsage::SAMPLED,
        });

        let mut halfdepths = Vec::new();
        let mut final_results = Vec::new();
        for i in 0..4 {
            halfdepths.push(device.create_texture(&TextureDescriptor {
                label: Some(Borrowed("Half depths")),
                size: Extent3d {
                    width,
                    height,
                    depth: 4,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Snorm,
                usage: TextureUsage::STORAGE | TextureUsage::SAMPLED,
            }));

            final_results.push(device.create_texture(&TextureDescriptor {
                label: Some(Borrowed("Final SSAO")),
                size: Extent3d {
                    width,
                    height,
                    depth: 4,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Snorm,
                usage: TextureUsage::STORAGE | TextureUsage::SAMPLED,
            }));
        }

        Self {
            width,
            height,

            prepare_normals_pass,
            prepare_normals_bgl,

            prepare_depths_pass,
            prepare_depths_bgl,

            ssao_pass,
            ssao_bgl,

            point_clamp_sampler,
            point_mirror_sampler,
            viewspace_depth_sampler,

            deinterlaced_normals,
            halfdepths,
            final_results,
        }
    }
}
