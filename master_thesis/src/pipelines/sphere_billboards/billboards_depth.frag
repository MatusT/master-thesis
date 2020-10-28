#version 460

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
};

layout(set = 2, binding = 0, std430) buffer Fragments { int fragments[]; };

layout(location = 0) in flat int instance_index;

layout(early_fragment_tests) in;

void main(void) {
  // const float len = length(uv);
  // if (length(uv) > scale) {
  //   discard;
  // }

#ifdef WRITE_VISIBILITY
  fragments[instance_index] = 1;
#endif
}