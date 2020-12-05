#version 460

layout(location = 0) in vec2 uv_input;

layout(set = 0, binding = 0) uniform sampler linear_sampler;
layout(set = 0, binding = 1) uniform texture2D input_color;

layout(location = 0) out vec4 output_color;

void main() {
    const vec2 uv = vec2(uv_input.x, 1.0 - uv_input.y);
    const vec3 color = texture(sampler2D(input_color, linear_sampler), uv).rgb;

    output_color = vec4(color, 1.0);
}