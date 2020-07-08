#version 460

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
};

layout(set = 2, binding = 0, std430) buffer Fragments { int fragments[]; };

layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 position_clip_space;
layout(location = 2) in flat float scale;
layout(location = 3) in flat int instance_index;

layout(early_fragment_tests) in;

void main(void) {
  const float len = length(uv);

// #ifdef WRITE_VISIBILITY
  if (length(uv) > scale) {
    discard;
  }
// #else 
//   discard;
//   // if (length(uv) > 0.5 * scale) {
//   //   discard;
//   // }
// #endif

#ifdef WRITE_VISIBILITY
  fragments[instance_index] = 1;
#endif
}