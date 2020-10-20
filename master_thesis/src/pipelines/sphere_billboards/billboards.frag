#version 460

layout(push_constant) uniform PushConstants {
    float time;
    uint id;
    vec4 color;
};

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
} camera;

layout(location = 0) in vec2 uv;
layout(location = 1) in flat vec4 center_vs;
layout(location = 2) in vec4 position_vs;
layout(location = 3) in vec4 position_cs;
layout(location = 4) in flat float scale;
layout(location = 5) in flat uint instance;
#ifdef DEBUG
layout(location = 6) in vec3 in_color;
#endif

layout(location = 0) out vec4 out_color;
#ifdef OUTPUT_NORMALS
layout(location = 1) out vec4 out_normal;
#else
layout(location = 1) out uint out_instance;
#endif


float remap(float value, float low1, float high1, float low2, float high2) {
	return low2 + (value - low1) * (high2 - low2) / (high1 - low1);
}

void main()
{
	const float len = length(uv);
	if (length(uv) > scale) {
		discard;
	}	
	
	float z = scale - len;
	
	// Depth Adjustment
	const vec4 fragment_position_clip = position_cs + camera.projection[2] * z;
	gl_FragDepth = fragment_position_clip.z / fragment_position_clip.w;

	z = remap(z, 0.0, scale, 0.0, 1.0);
	#ifdef DEBUG
	out_color = vec4(z * in_color, 1.0);
	#else	
	out_color = vec4(z * color.xyz, 1.0);
	#endif

	#ifdef OUTPUT_NORMALS
		out_normal = vec4(normalize(position_vs.xyz - center_vs.xyz), 0.0);
	#else
		out_instance = id;
	#endif
}