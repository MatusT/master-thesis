#version 460

#ifdef EARLY
layout (early_fragment_tests) in;
#endif

#ifdef LESS
layout (depth_less) out float gl_FragDepth;
#endif

#ifdef GREATER
layout (depth_greater) out float gl_FragDepth;
#endif

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
} camera;

layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 position_cs;

layout(location = 0) out vec4 color;

void main()
{
	// Circle discard
	const float len = length(uv);
	// if (len > 1.0) {
	// 	discard;
	// }
	
	const float z = 1.0 - len;
	
	// Depth Adjustment
	const vec4 fragment_position_clip = position_cs + camera.projection[2] * z;
	gl_FragDepth = fragment_position_clip.z / fragment_position_clip.w;

	color = vec4(z, z, z, 1.0);
}