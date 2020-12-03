#version 460

layout(push_constant) uniform PushConstants {
    vec2 depth_unpack;
    float depth_distance;
    float pow_far;
    float pow_near;
};

float ScreenSpaceToViewSpaceDepth(float depth)
{
    float depthLinearizeMul = depth_unpack.x;
    float depthLinearizeAdd = depth_unpack.y;

    return depthLinearizeMul / (depthLinearizeAdd - depth);
}

float remap(float value, float low1, float high1, float low2, float high2) {
	return low2 + (value - low1) * (high2 - low2) / (high1 - low1);
}

layout(location = 0) in vec2 uv_input;

layout(set = 0, binding = 0) uniform sampler linear_sampler;

layout(set = 0, binding = 1) uniform texture2D input_color;
layout(set = 0, binding = 2) uniform texture2D input_ao1;
layout(set = 0, binding = 3) uniform texture2D input_ao2;
layout(set = 0, binding = 4) uniform texture2D input_depth;
// layout(set = 0, binding = 5) uniform texture2D input_bloom;

layout(location = 0) out vec4 output_color;

void main() {
    vec2 uv = vec2(uv_input.x, 1.0 - uv_input.y);
    const vec3 color = texture(sampler2D(input_color, linear_sampler), uv).rgb;
    const float ao_far = 1.0 - texture(sampler2D(input_ao1, linear_sampler), uv).r;
    const float ao_near = 1.0 - texture(sampler2D(input_ao2, linear_sampler), uv).r;
    const float depth = ScreenSpaceToViewSpaceDepth(texture(sampler2D(input_depth, linear_sampler), uv).r);
    // const vec4 bloom = texture(sampler2D(input_bloom, linear_sampler), uv).rgba;

    // const vec3 final_color = (bloom.rgb * bloom.a + color.rgb * (1.0 - bloom.a)).rgb;
    // const vec3 final_color = vec3(color);

    // Fog computation
    // const float fog = 1.0 - (clamp(0.0, depth_distance, depth) / depth_distance);

    // SSAO interpolation
    // const float ao = (ao1 + ao2) / 2.0;

    // output_color = vec4(fog * ao * color, 1.0);
    // float final_ao = 0.0;
    // if (depth < 100.0) {
    //     final_ao = ao_near;
    // } else if (depth > 7500.0) {
    //     final_ao = ao_far;
    // } else {
    //     float v = remap(depth, 100.0, 7500.0, 0.0, 1.0);
    //     final_ao = (1.0 - v) * ao_near + v * ao_far;
    // }
    // float final_ao = clamp(ao_far * ao_near * 2.0, 0.0, 1.0);
    // float final_ao = clamp(pow(ao_far, pow_far) * pow(ao_near, pow_near), 0.0, 1.0);
    float final_ao = clamp(pow(ao_far, pow_far), 0.0, 1.0);
    output_color = vec4(final_ao, final_ao, final_ao, 1.0);
}