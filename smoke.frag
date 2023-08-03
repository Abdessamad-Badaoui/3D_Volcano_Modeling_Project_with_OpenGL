#version 330 core


uniform sampler2D diffuse_map;
in vec2 frag_tex_coords;
out vec4 out_color;

void main() {
    vec4 tex_color = texture(diffuse_map, frag_tex_coords);
    if(tex_color.a < 0.1)
        discard;
    out_color = vec4(tex_color.rgb, tex_color.a);
}
