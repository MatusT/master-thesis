#version 460

layout(push_constant) uniform PushConstants {
  vec2 size;
};

layout(location = 0) in vec2 uv_input;

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_color;

layout(location = 0) out vec4 output_color;

void main() {
    const vec2 uv = vec2(uv_input.x, 1.0 - uv_input.y);
    const vec3 color = imageLoad(input_color, ivec2(uv * size)).rgb;

    output_color = vec4(color, 1.0);
}