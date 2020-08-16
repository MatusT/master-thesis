#version 460

layout(origin_upper_left) in vec4 gl_FragCoord;

layout(set = 0, binding = 0, r32f) uniform image2D input_ao;

layout(location = 0) out vec4 output_color;

void main() {
    const float ao = imageLoad(input_ao, ivec2(gl_FragCoord.xy)).x;
    
    output_color = vec4(ao, ao, ao, 1.0);
}