#version 460

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
} camera;

layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 position_clip_space;
layout(location = 2) in flat float scale;
layout(location = 3) in flat vec3 color;

layout(location = 0) out vec4 out_color;

void main()
{
	const float len = length(uv);
	if (length(uv) > scale) {
		discard;
	}	
	
	const float z = scale - len;
	const vec3 normal = normalize(vec3(uv.x, uv.y, z));
	
	// Depth Adjustment
	const vec4 fragment_position_clip = position_clip_space + camera.projection[2] * z;
	gl_FragDepth = fragment_position_clip.z / fragment_position_clip.w;

	const float diffuse = max(dot(normal, vec3(0.0, 0.0, 1.0)), 0.0);
	
	out_color = vec4(color * diffuse, 1.0);
}