// ====================================================================
// Prepare shader dimensions

const int PREPARE_DEPTHS_AND_MIPS_WIDTH  = 8;
const int PREPARE_DEPTHS_AND_MIPS_HEIGHT = 8;

const int PREPARE_DEPTHS_WIDTH  = 8;
const int PREPARE_DEPTHS_HEIGHT = 8;

const int PREPARE_DEPTHS_HALF_WIDTH  = 8;
const int PREPARE_DEPTHS_HALF_HEIGHT = 8;

const int PREPARE_DEPTHS_NORMALS_AND_MIPS_WIDTH  = 8;
const int PREPARE_DEPTHS_NORMALS_AND_MIPS_HEIGHT = 8;

const int PREPARE_DEPTHS_AND_NORMALS_WIDTH  = 8;
const int PREPARE_DEPTHS_AND_NORMALS_HEIGHT = 8;

const int PREPARE_DEPTHS_AND_NORMALS_HALF_WIDTH  = 8;
const int PREPARE_DEPTHS_AND_NORMALS_HALF_HEIGHT = 8;

const int PREPARE_NORMALS_WIDTH  = 8;
const int PREPARE_NORMALS_HEIGHT = 8;

const int PREPARE_NORMALS_FROM_INPUT_NORMALS_WIDTH  = 8;
const int PREPARE_NORMALS_FROM_INPUT_NORMALS_HEIGHT = 8;

// ====================================================================
// Generate SSAO shader dimensions

const int GENERATE_WIDTH  = 8;
const int GENERATE_HEIGHT = 8;

// ====================================================================
// Importance map shader dimensions

const int IMPORTANCE_MAP_WIDTH  = 8;
const int IMPORTANCE_MAP_HEIGHT = 8;

const int IMPORTANCE_MAP_A_WIDTH  = 8;
const int IMPORTANCE_MAP_A_HEIGHT = 8;

const int IMPORTANCE_MAP_B_WIDTH  = 8;
const int IMPORTANCE_MAP_B_HEIGHT = 8;

// ====================================================================
// Blur shader dimensions

const int BLUR_WIDTH  = 16;
const int BLUR_HEIGHT = 16;

// ====================================================================
// Apply shader dimensions

const int APPLY_WIDTH  = 8;
const int APPLY_HEIGHT = 8;

// ====================================================================
// Reinterleave shader dimensions

const int REINTERLEAVE_WIDTH = 16;
const int REINTERLEAVE_HEIGHT = 8;

// ====================================================================
// Upscale

const int UPSCALE_WIDTH = 8;
const int UPSCALE_HEIGHT = 8;

const int BILATERAL_UPSCALE_WIDTH  = 8;
const int BILATERAL_UPSCALE_HEIGHT = 4;

const int SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION = 1;

const int INTELSSAO_MAIN_DISK_SAMPLE_COUNT = 32;

const vec4 g_samplePatternMain[INTELSSAO_MAIN_DISK_SAMPLE_COUNT] =
{
	vec4(0.78488064,  0.56661671,  1.500000, -0.126083),     
	vec4(0.26022232, -0.29575172,  1.500000, -1.064030),     
	vec4(0.10459357,  0.08372527,  1.110000, -2.730563),    
	vec4(-0.68286800,  0.04963045,  1.090000, -0.498827),
	vec4(-0.13570161, -0.64190155,  1.250000, -0.532765),    
	vec4(-0.26193795, -0.08205118,  0.670000, -1.783245),    
	vec4(-0.61177456,  0.66664219,  0.710000, -0.044234),     
	vec4(0.43675563,  0.25119025,  0.610000, -1.167283),
	vec4(0.07884444,  0.86618668,  0.640000, -0.459002),    
	vec4(-0.12790935, -0.29869005,  0.600000, -1.729424),    
	vec4(-0.04031125,  0.02413622,  0.600000, -4.792042),     
	vec4(0.16201244, -0.52851415,  0.790000, -1.067055),
	vec4(-0.70991218,  0.47301072,  0.640000, -0.335236),     
	vec4(0.03277707, -0.22349690,  0.600000, -1.982384),     
	vec4(0.68921727,  0.36800742,  0.630000, -0.266718),     
	vec4(0.29251814,  0.37775412,  0.610000, -1.422520),
	vec4(-0.12224089,  0.96582592,  0.600000, -0.426142),     
	vec4(0.11071457, -0.16131058,  0.600000, -2.165947),     
	vec4(0.46562141, -0.59747696,  0.600000, -0.189760),    
	vec4(-0.51548797,  0.11804193,  0.600000, -1.246800),
	vec4(0.89141309, -0.42090443,  0.600000,  0.028192),    
	vec4(-0.32402530, -0.01591529,  0.600000, -1.543018),     
	vec4(0.60771245,  0.41635221,  0.600000, -0.605411),     
	vec4(0.02379565, -0.08239821,  0.600000, -3.809046),
	vec4(0.48951152, -0.23657045,  0.600000, -1.189011),    
	vec4(-0.17611565, -0.81696892,  0.600000, -0.513724),    
	vec4(-0.33930185, -0.20732205,  0.600000, -1.698047),    
	vec4(-0.91974425,  0.05403209,  0.600000,  0.062246),
	vec4(-0.15064627, -0.14949332,  0.600000, -1.896062),     
	vec4(0.53180975, -0.35210401,  0.600000, -0.758838),     
	vec4(0.41487166,  0.81442589,  0.600000, -0.505648),    
	vec4(-0.24106961, -0.32721516,  0.600000, -1.665244)
};

const int SSAO_MAX_TAPS = 32;
const int SSAO_MAX_REF_TAPS = 512;
const int SSAO_ADAPTIVE_TAP_BASE_COUNT = 5;
const int SSAO_ADAPTIVE_TAP_FLEXIBLE_COUNT = SSAO_MAX_TAPS - SSAO_ADAPTIVE_TAP_BASE_COUNT;
const int SSAO_DEPTH_MIP_LEVELS = 4;

// these values can be changed (up to SSAO_MAX_TAPS) with no changes required elsewhere; values for 4th and 5th preset are ignored but array needed to avoid compilation errors
// the actual number of texture samples is two times this value (each "tap" has two symmetrical depth texture samples)
// TODO: Fixed number of taps. Wait for the answer at https://github.com/GPUOpen-Effects/FidelityFX-CACAO/issues/4?fbclid=IwAR0ZnsXhnAJGsqlzTwdri1M7P0zYjCDsJcsocFIrnNARR9Ndi1RQ8Avb1yc
const uint g_numTaps[5] = { 3, 3, 5, 12, 0 };

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Optional parts that can be enabled for a required quality preset level and above (0 == Low, 1 == Medium, 2 == High, 3 == Highest/Adaptive, 4 == reference/unused )
// Each has its own cost. To disable just set to 5 or above.
//
// (experimental) tilts the disk (although only half of the samples!) towards surface normal; this helps with effect uniformity between objects but reduces effect distance and has other side-effects
const int   SSAO_TILT_SAMPLES_ENABLE_AT_QUALITY_PRESET                    = (99);        // to disable simply set to 99 or similar
const float SSAO_TILT_SAMPLES_AMOUNT                                      = (0.4);
//
const int   SSAO_HALOING_REDUCTION_ENABLE_AT_QUALITY_PRESET               = (1);         // to disable simply set to 99 or similar
const float SSAO_HALOING_REDUCTION_AMOUNT                                 = (0.6);       // values from 0.0 - 1.0, 1.0 means max weighting (will cause artifacts, 0.8 is more reasonable)
//
const int   SSAO_NORMAL_BASED_EDGES_ENABLE_AT_QUALITY_PRESET              = (2); //2        // to disable simply set to 99 or similar
const float SSAO_NORMAL_BASED_EDGES_DOT_THRESHOLD                         = (0.5);       // use 0-0.1 for super-sharp normal-based edges
//
const int   SSAO_DETAIL_AO_ENABLE_AT_QUALITY_PRESET                       = (1); //1         // whether to use DetailAOStrength; to disable simply set to 99 or similar
//
const int   SSAO_DEPTH_MIPS_ENABLE_AT_QUALITY_PRESET                      = (2);         // !!warning!! the MIP generation on the C++ side will be enabled on quality preset 2 regardless of this value, so if changing here, change the C++ side too
const float SSAO_DEPTH_MIPS_GLOBAL_OFFSET                                 = (-4.3);      // best noise/quality/performance tradeoff, found empirically
//
// !!warning!! the edge handling is hard-coded to 'disabled' on quality level 0, and enabled above, on the C++ side; while toggling it here will work for 
// testing purposes, it will not yield performance gains (or correct results)
const int SSAO_DEPTH_BASED_EDGES_ENABLE_AT_QUALITY_PRESET                 = (1);     
//
const int SSAO_REDUCE_RADIUS_NEAR_SCREEN_BORDER_ENABLE_AT_QUALITY_PRESET  = (99);        // 99 means disabled; only helpful if artifacts at the edges caused by lack of out of screen depth data are not acceptable with the depth sampler in either clamp or mirror modes
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
