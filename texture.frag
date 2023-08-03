#version 330 core

uniform sampler2D diffuse_map;
uniform sampler2D second_texture;
in vec2 frag_tex_coords;
out vec4 out_color;

void main() {
    vec4 color1 = texture(diffuse_map,frag_tex_coords);
    vec4 color2 = texture(second_texture,frag_tex_coords);
    
    out_color = mix(color1,color2,color2.a);
}
