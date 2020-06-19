#version 460

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
} camera;

layout(location = 0) in vec3 position;

void main() { 
    gl_Position = camera.projection_view * vec4(position, 1.0); 
}
