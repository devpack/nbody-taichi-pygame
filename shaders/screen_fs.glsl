#version 460 core

uniform sampler2D tex;
uniform float time;

in vec2 uvs;
out vec4 f_color;

void main() {
    //vec2 sample_pos = vec2(uvs.x + sin(uvs.y * 4  + time * 0.01) * 0.1, uvs.y);
    //f_color = vec4(texture(tex, sample_pos).rg, texture(tex, sample_pos).b * 1.5, 1.0);
    vec2 sample_pos = uvs;
    f_color = vec4(texture(tex, sample_pos).rgb, 1.0);
}