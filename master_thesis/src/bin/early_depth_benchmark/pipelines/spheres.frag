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
layout(location = 1) in vec4 position_atom_cs;
layout(location = 2) in vec4 position_billboard_cs;
layout(location = 3) in vec4 position_billboard_ss;

layout(location = 0) out vec4 color;

void main()
{
	// Circle discard
	const float len = length(uv);
	// if (len > 1.0) {
	// 	discard;
	// }
	
	// Depth Adjustment
	const float z = 1.0 - len;
	const vec4 fragment_position_clip = position_atom_cs + camera.projection[2] * z;
	const float new_depth = fragment_position_clip.z / fragment_position_clip.w;
	gl_FragDepth = new_depth;

	#ifdef LESS
	if (new_depth > position_billboard_ss.z) {
		// ALERT with RED COLOR that the shader is writing higher depth
		color = vec4(position_billboard_ss.z, new_depth, 0.0, 1.0);
	}
	#endif
	#ifdef GREATER
	if (new_depth < position_billboard_ss.z) {
		// ALERT with RED COLOR that the shader is writing lower depth
		color = vec4(position_billboard_ss.z, new_depth, 0.0, 1.0);
	}
	#endif
	else {
		color = vec4(z, z, z, 1.0);
	}	
}