#version 460

layout(location = 0) in vec2 uv;

layout(set = 0, binding = 0) uniform sampler linear_sampler;
layout(set = 0, binding = 1) uniform texture2D input_color;
layout(set = 0, binding = 2) uniform texture2D input_ao1;
layout(set = 0, binding = 3) uniform texture2D input_ao2;

layout(location = 0) out vec4 output_color;

void main() {
    const vec3 color = texture(sampler2D(input_color, linear_sampler), uv).rgb;
    const float ao1 = texture(sampler2D(input_ao1, linear_sampler), uv).r;
    const float ao2 = texture(sampler2D(input_ao2, linear_sampler), uv).r;

    const float ao = (ao1 + ao2) / 2.0;
    
    output_color = vec4(ao * color, 1.0);
}