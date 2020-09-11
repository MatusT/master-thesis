#version 460

const vec2 vertices[3] = {
    vec2(-1.72, -1.0),
    vec2(1.72, -1.0),
    vec2(0.0, 3.0),
};

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
} camera;

layout(set = 1, binding = 0, std430) buffer AtomPositions {
  vec4 atom_positions[];
};

layout(location = 0) out vec2 uv;
layout(location = 1) out vec4 position_atom_cs;
layout(location = 2) out vec4 position_billboard_cs;
layout(location = 3) out vec4 position_billboard_ss;

void main() {
  const vec3 camera_right = vec3(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
  const vec3 camera_up = vec3(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

  const vec2 vertex = vertices[gl_VertexIndex % 3];
  const vec3 atom_center = atom_positions[gl_VertexIndex / 3].xyz;
  const vec3 move_vec = normalize(camera.position.xyz - atom_center);

  vec4 position_atom_ws = vec4(atom_center + vertex.x * camera_right + vertex.y * camera_up, 1.0);

  vec4 billboard_position_ws = vec4(atom_center + move_vec, 1.0);  
  billboard_position_ws = vec4(billboard_position_ws.xyz + vertex.x * camera_right + vertex.y * camera_up, 1.0);

  uv = vertex;

  position_atom_cs = camera.projection_view * position_atom_ws;
  position_billboard_cs = camera.projection_view * billboard_position_ws;
  position_billboard_ss = position_billboard_cs / position_billboard_cs.w;

  gl_Position = position_billboard_cs;
}
