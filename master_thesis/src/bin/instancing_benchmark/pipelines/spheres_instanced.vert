#version 460

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
} camera;

layout(location = 0) in vec4 molecule_position;
layout(location = 1) in vec4 atom_position;

const vec2 vertices[3] = {
    vec2(-1.72, -1.0),
    vec2(1.72, -1.0),
    vec2(0.0, 3.0),
};

void main() {
  const vec3 camera_right = vec3(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
  const vec3 camera_up = vec3(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

  const vec4 center_position = vec4(molecule_position.xyz + atom_position.xyz, 1.0);
  const vec2 vertex = vertices[gl_VertexIndex];
  
  const vec4 position_ws = vec4(center_position.xyz + vertex.x * camera_right + vertex.y * camera_up, 1.0);

  gl_Position = camera.projection_view * position_ws;
}
