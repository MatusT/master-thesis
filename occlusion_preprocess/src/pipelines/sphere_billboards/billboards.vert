#version 460

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
};

layout(set = 1, binding = 0, std430) buffer Positions {
  vec4 positions[];
};

layout(set = 1, binding = 1, std430) buffer ModelMatrices {
  mat4 model_matrices[];
};

layout(location = 0) out vec2 uv;
layout(location = 1) out vec4 position_clip_space;
layout(location = 2) out flat float scale;
layout(location = 3) out flat vec3 color;

const vec2 vertices[3] = {
    vec2(-1.72, -1.0),
    vec2(1.72, -1.0),
    vec2(0.0, 3.0),
};

uint hash(uint a)
{
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}

void main() {
  const vec3 camera_right = vec3(view[0][0], view[1][0], view[2][0]);
  const vec3 camera_up = vec3(view[0][1], view[1][1], view[2][1]);

  const mat4 model_matrix = model_matrices[gl_InstanceIndex];
  const vec4 world_position = model_matrix * vec4(positions[gl_VertexIndex / 3].xyz, 1.0);
  scale = positions[gl_VertexIndex / 3].w;

  const vec2 vertex = scale * vertices[gl_VertexIndex % 3];
  const vec4 position_worldspace = vec4(
      world_position.xyz +
      vertex.x * camera_right +
      vertex.y * camera_up, 1.0);

	// uint mhash = hash(gl_InstanceIndex);
	// color = vec3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
  uint mhash = hash(gl_InstanceIndex / 256);
  color = vec3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;

  uv = vertex;
  position_clip_space = projection_view * position_worldspace;    
  gl_Position = position_clip_space;
}
