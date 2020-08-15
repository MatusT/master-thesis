#version 460

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

#ifdef OUTPUT_NORMALS
layout(location = 0) out vec4 out_normal;
#endif

void main()
{
	const float len = length(uv);
	if (length(uv) > scale) {
		discard;
	}	
	
	const float z = scale - len;
	
	// Depth Adjustment
	const vec4 fragment_position_clip = position_cs + camera.projection[2] * z;
	gl_FragDepth = fragment_position_clip.z / fragment_position_clip.w;

	#ifdef OUTPUT_NORMALS
	out_normal = vec4(normalize(position_vs.xyz - center_vs.xyz), 0.0);
	#endif
}