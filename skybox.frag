#version 330 core

uniform samplerCube skybox;
in vec3 frag_tex_coords;
out vec4 out_color;

void main() {

    out_color = texture(skybox,frag_tex_coords);
}
