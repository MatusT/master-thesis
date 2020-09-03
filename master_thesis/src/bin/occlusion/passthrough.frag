#version 460

layout(origin_upper_left) in vec4 gl_FragCoord;

layout(set = 0, binding = 0, r32f) uniform image2D input_ao1;
layout(set = 0, binding = 1, r32f) uniform image2D input_ao2;

layout(location = 0) out vec4 output_color;

void main() {
    const float ao1 = imageLoad(input_ao1, ivec2(gl_FragCoord.xy)).x;
    const float ao2 = imageLoad(input_ao2, ivec2(gl_FragCoord.xy)).x;

    const float ao = (ao1 + ao2) / 2.0;
    
    output_color = vec4(ao, ao, ao, 1.0);
}