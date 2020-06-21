#version 460

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
}
camera;

layout(location = 0) in vec3 position;

layout(location = 0) out vec3 color;

uint hash(uint a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

void main() {
  // Find cube face
  let v = normalize(&v);
  let v_abs = v.abs();

  if (v_abs.x > v_abs.y && v_abs.x > v_abs.z) {
    // X major
    if (v.x >= 0.0) {
      CubeFace::Right
    } else {
      CubeFace::Left
    }
  } else if (v_abs.y > v_abs.z) {
    // Y major
    if (v.y >= 0.0) {
      CubeFace::Top
    } else {
      CubeFace::Bottom
    }
  } else {
    // Z major
    if (v.z >= 0.0) {
      CubeFace::Front
    } else {
      CubeFace::Back
    }
  }
  const uint mhash = hash((gl_VertexIndex / 2) / 1024);
  color = vec3(float(mhash & 255), float((mhash >> 8) & 255),
               float((mhash >> 16) & 255)) /
          255.0;
  // color = vec3(((gl_VertexIndex / 2) % 255) / 255.0);
  gl_Position = camera.projection_view * vec4(position, 1.0);
}
