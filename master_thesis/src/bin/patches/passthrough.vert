#version 460

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec2 uv;

void main() {
    float x = -1.0 + float((gl_VertexIndex & 1) << 2);
    float y = -1.0 + float((gl_VertexIndex & 2) << 1);

    uv.x = (x + 1.0) * 0.5;
    uv.y = (y + 1.0) * 0.5;

    gl_Position = vec4(x, y, 0, 1);
}