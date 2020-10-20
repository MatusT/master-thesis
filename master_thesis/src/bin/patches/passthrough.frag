#version 460

layout(push_constant) uniform PushConstants {
    vec2 depth_unpack;
    float depth_distance;
};

float ScreenSpaceToViewSpaceDepth(float depth)
{
    float depthLinearizeMul = depth_unpack.x;
    float depthLinearizeAdd = depth_unpack.y;

    return depthLinearizeMul / (depthLinearizeAdd - depth);
}

layout(location = 0) in vec2 uv;

layout(set = 0, binding = 0) uniform sampler linear_sampler;

layout(set = 0, binding = 1) uniform texture2D input_color;
layout(set = 0, binding = 2) uniform texture2D input_ao1;
layout(set = 0, binding = 3) uniform texture2D input_ao2;
layout(set = 0, binding = 4) uniform texture2D input_depth;
layout(set = 0, binding = 5) uniform utexture2D input_instance;

layout(location = 0) out vec4 output_color;

void main() {
    const vec3 color = texture(sampler2D(input_color, linear_sampler), uv).rgb;
    const float ao1 = texture(sampler2D(input_ao1, linear_sampler), uv).r;
    const float ao2 = texture(sampler2D(input_ao2, linear_sampler), uv).r;
    const float depth = ScreenSpaceToViewSpaceDepth(texture(sampler2D(input_depth, linear_sampler), uv).r);

    // Fog computation
    const float fog = 1.0 - (clamp(0.0, depth_distance, depth) / depth_distance);

    output_color = vec4(fog, fog, fog, 1.0);

    // SSAO interpolation
    const float ao = (ao1 + ao2) / 2.0;

    // Contour based on instance ID differences
    const vec2 resolution = textureSize(usampler2D(input_instance, linear_sampler), 0);
    const float w_step = 1.0 / resolution.x;
    const float h_step = 1.0 / resolution.y;

    const int X = int(texture(usampler2D(input_instance, linear_sampler), uv).r);
    const int R = int(texture(usampler2D(input_instance, linear_sampler), uv + vec2(w_step, 0)).r);
    const int L = int(texture(usampler2D(input_instance, linear_sampler), uv + vec2(-w_step, 0)).r);
    const int T = int(texture(usampler2D(input_instance, linear_sampler), uv + vec2(0, h_step)).r);
    const int B = int(texture(usampler2D(input_instance, linear_sampler), uv + vec2(0, -h_step)).r);

    if ( (X == R) && (X == L) && (X == T) && (X == B) )
    { //~ current pixel is NOT on the edge
        output_color = vec4(fog * ao * color, 1.0);
    }
    else
    { //~ current pixel lies on the edge
        output_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
}