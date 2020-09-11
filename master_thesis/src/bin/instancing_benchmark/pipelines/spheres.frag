#version 460

#extension GL_ARB_conservative_depth : enable

layout (early_fragment_tests) in;
layout (depth_less) out float gl_FragDepth;

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
} camera;

// layout(location = 0) in vec2 uv;
// layout(location = 1) in vec4 position_cs;

layout(location = 0) out vec4 color;

void main()
{
	// const float len = length(uv);
	// if (length(uv) > 1.0) {
	// 	discard;
	// }	
	
	// const float z = 1.0 - len;
	
	// // Depth Adjustment
	// const vec4 fragment_position_clip = position_cs + camera.projection[2] * z;
	// gl_FragDepth = fragment_position_clip.z / fragment_position_clip.w;

	color = vec4(1.0);
}