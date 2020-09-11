#version 460

layout(push_constant) uniform PushConstants {
    uint atoms_len;
};

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
};

layout(set = 1, binding = 0, std140) uniform AtomsPositions {
  vec4 positions[1024];
} atom;

layout(set = 1, binding = 1, std430) buffer MoleculePositions {
  vec4 positions[];
} molecules;

const vec2 vertices[3] = {
    vec2(-1.72, -1.0),
    vec2(1.72, -1.0),
    vec2(0.0, 3.0),
};

void main() {
  const vec3 camera_right = vec3(view[0][0], view[1][0], view[2][0]);
  const vec3 camera_up = vec3(view[0][1], view[1][1], view[2][1]);

  const vec4 center_position = vec4(molecules.positions[gl_VertexIndex / (atoms_len * 3)].xyz + atom.positions[(gl_VertexIndex / 3) % atoms_len].xyz, 1.0);
  const vec2 vertex = vertices[gl_VertexIndex % 3];
  
  const vec4 position_ws = vec4(center_position.xyz + vertex.x * camera_right + vertex.y * camera_up, 1.0);

  gl_Position = projection_view * position_ws;
}
