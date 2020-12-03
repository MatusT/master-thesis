#version 460

layout(push_constant) uniform PushConstants {
    float time;
    uint object_id;
    float cut;
};

layout(set = 0, binding = 0, std140) uniform CameraMatrices {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
  vec4 position;
};

layout(set = 1, binding = 0, std430) buffer MoleculeAtomsPositions {
  vec4 positions[];
};

layout(set = 1, binding = 1, std430) buffer MoleculeModelMatrices {
  mat4 model_matrices[];
};

layout(set = 2, binding = 0, std140) uniform StructureGlobals {
  mat4 model_matrix;
}
structure;

layout(location = 0) out vec2 uv;
layout(location = 1) out flat vec4 center_vs;
layout(location = 2) out vec4 position_vs;
layout(location = 3) out vec4 position_cs;
layout(location = 4) out flat float scale;
layout(location = 5) out flat uint instance;

#ifdef DEBUG
layout(location = 6) out vec3 color;
#endif

const vec2 vertices[3] = {
    vec2(-1.72, -1.0),
    vec2(1.72, -1.0),
    vec2(0.0, 3.0),
};

int vector_cube_face(vec3 v) {
  v = normalize(v);
  vec3 v_abs = abs(v);

  if (v_abs.x > v_abs.y && v_abs.x > v_abs.z) {
    // X major
    if (v.x >= 0.0) {
      return 0;
    } else {
      return 1;
    }
  } else if (v_abs.y > v_abs.z) {
    // Y major
    if (v.y >= 0.0) {
      return 2;
    } else {
      return 3;
    }
  } else {
    // Z major
    if (v.z >= 0.0) {
      return 4;
    } else {
      return 5;
    }
  }
}

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
  const vec3 camera_right = vec3(view[0][0], view[1][0], view[2][0]);
  const vec3 camera_up = vec3(view[0][1], view[1][1], view[2][1]);

  const mat4 local_model_matrix = model_matrices[gl_InstanceIndex];
  const mat4 model_matrix = structure.model_matrix * local_model_matrix;
  const vec3 translation = local_model_matrix[3].xyz;

  const float offset = 0.75 * sin(4.0f * 3.14159265359f * time + float(gl_InstanceIndex) / 10.0);

  const vec4 center_position =
      (model_matrix * vec4(positions[gl_VertexIndex / 3].xyz, 1.0)) + vec4(offset, offset, offset, 0.0);
  scale = positions[gl_VertexIndex / 3].w;

  const vec2 vertex = scale * vertices[gl_VertexIndex % 3];
  const vec4 position_ws = vec4(
      center_position.xyz + vertex.x * camera_right + vertex.y * camera_up, 1.0);

  uv = vertex;
  center_vs = view * center_position;
  position_vs = view * position_ws;
  position_cs = projection_view * position_ws;

  #ifdef DEBUG
  const uint id = (gl_InstanceIndex / 3) / 256;
  const uint mhash = hash(id);
	color = vec3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
  #endif

  instance = object_id * 100000 + uint(gl_InstanceIndex);

  const vec4 local_center =
      (local_model_matrix * vec4(positions[gl_VertexIndex / 3].xyz, 1.0)) + vec4(offset, offset, offset, 0.0);
  if (local_center.x > cut) {
    gl_Position = vec4(0.0);
  } else {
    gl_Position = position_cs;
  }
}
