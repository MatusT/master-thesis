use nalgebra_glm::{Vec2, Vec4};
use wgpu::*;

struct Constants
{
	DepthBufferOffset: TVec2<i32>,
	DepthUnpackConsts: Vec2,

	NDCToViewMul: Vec2,
	NDCToViewAdd: Vec2,

	DepthBufferUVToViewMul: Vec2,
	DepthBufferUVToViewAdd: Vec2,

	EffectRadius: f32,                           // world (viewspace) maximum size of the shadow
	EffectShadowStrength: f32,                   // global strength of the effect (0 - 5)
	EffectShadowPow: f32,
	EffectShadowClamp: f32,

	EffectFadeOutMul: f32,                       // effect fade out from distance (ex. 25)
	EffectFadeOutAdd: f32,                       // effect fade out to distance   (ex. 100)
	EffectHorizonAngleThreshold: f32,            // limit errors on slopes and caused by insufficient geometry tessellation (0.05 to 0.5)
	EffectSamplingRadiusNearLimitRec: f32,       // if viewspace pixel closer than this, don't enlarge shadow sampling radius anymore (makes no sense to grow beyond some distance, not enough samples to cover everything, so just limit the shadow growth: f32, could be SSAOSettingsFadeOutFrom * 0.1 or less)

	DepthPrecisionOffsetMod: f32,
	NegRecEffectRadius: f32,                     // -1.0 / EffectRadius
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
	radius: f32,                            ///< [0.0,  ~ ] World (view) space size of the occlusion sphere.
	shadowMultiplier: f32,                  ///< [0.0, 5.0] Effect strength linear multiplier
	shadowPower: f32,                       ///< [0.5, 5.0] Effect strength pow modifier
	shadowClamp: f32,                       ///< [0.0, 1.0] Effect max limit (applied after multiplier but before blur)
	horizonAngleThreshold: f32,             ///< [0.0, 0.2] Limits self-shadowing (makes the sampling area less of a hemisphere, more of a spherical cone, to avoid self-shadowing and various artifacts due to low tessellation and depth buffer imprecision, etc.)
	fadeOutFrom: f32,                       ///< [0.0,  ~ ] Distance to start start fading out the effect.
	fadeOutTo: f32,                         ///< [0.0,  ~ ] Distance at which the effect is faded out.
	blurPassCount: u32,                     ///< [  0,   8] Number of edge-sensitive smart blur passes to apply
	sharpness: f32,                         ///< [0.0, 1.0] (How much to bleed over edges: 1: not at all, 0.5: half-half: f32, 0.0: completely ignore edges)
	detailShadowStrength: f32,              ///< [0.0, 5.0] Used for high-res detail AO using neighboring depth pixels: adds a lot of detail but also reduces temporal stability (adds aliasing).
	generateNormals: bool,                  ///< This option should be set to true if FidelityFX-CACAO should reconstruct a normal buffer from the depth buffer. It is required to be true if no normal buffer is provided.
	bilateralSigmaSquared: f32,             ///< [0.0,  ~ ] Sigma squared value for use in bilateral upsampler giving Gaussian blur term. Should be greater than 0.0. 
	bilateralSimilarityDistanceSigma: f32,  ///< [0.0,  ~ ] Sigma squared value for use in bilateral upsampler giving similarity weighting for neighbouring pixels. Should be greater than 0.0.
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

	prepare_pass: ComputePipeline,
	ssao_pass: ComputePipeline, 

	point_clamp_sampler: Sampler,
	point_mirror_sampler: Sampler,
	viewspace_depth_sampler: Sampler,

	deinterlaced_normals: [TextureView; 4],

	output: Texture,
}

impl SsaoModule {
	pub fn new() -> Self {

	}
}