struct CACAOConstants
{
	ivec2                 DepthBufferOffset;
	vec2                  DepthUnpackConsts;

	vec2                  NDCToViewMul;
	vec2                  NDCToViewAdd;

	vec2                  DepthBufferUVToViewMul;
	vec2                  DepthBufferUVToViewAdd;

	float                 EffectRadius;                           // world (viewspace) maximum size of the shadow
	float                 EffectShadowStrength;                   // global strength of the effect (0 - 5)
	float                 EffectShadowPow;
	float                 EffectShadowClamp;

	float                 EffectFadeOutMul;                       // effect fade out from distance (ex. 25)
	float                 EffectFadeOutAdd;                       // effect fade out to distance   (ex. 100)
	float                 EffectHorizonAngleThreshold;            // limit errors on slopes and caused by insufficient geometry tessellation (0.05 to 0.5)
	float                 EffectSamplingRadiusNearLimitRec;       // if viewspace pixel closer than this, don't enlarge shadow sampling radius anymore (makes no sense to grow beyond some distance, not enough samples to cover everything, so just limit the shadow growth; could be SSAOSettingsFadeOutFrom * 0.1 or less)

	float                 DepthPrecisionOffsetMod;
	float                 NegRecEffectRadius;                     // -1.0 / EffectRadius
	float                 DetailAOStrength;
	int                   PassIndex;

	vec4                  PatternRotScaleMatrices[5];

	vec2                  SSAOBufferDimensions;
	vec2                  SSAOBufferInverseDimensions;

	vec2                  DepthBufferDimensions;
	vec2                  DepthBufferInverseDimensions;
	
	vec2                  OutputBufferDimensions;
	vec2                  OutputBufferInverseDimensions;

	vec2                  DeinterleavedDepthBufferDimensions;
	vec2                  DeinterleavedDepthBufferInverseDimensions;

	vec2                  DeinterleavedDepthBufferOffset;
	vec2                  DeinterleavedDepthBufferNormalisedOffset;
};

float saturate(float val) {
	return clamp(val, 0.0, 1.0);
}

vec2 saturate(vec2 val) {
	return clamp(val, 0.0, 1.0);
}

vec3 saturate(vec3 val) {
	return clamp(val, 0.0, 1.0);
}

vec4 saturate(vec4 val) {
	return clamp(val, 0.0, 1.0);
}
