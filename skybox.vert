#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
in vec3 position;
in vec3 tex_coord;

out vec3 frag_tex_coords;

void main() {
    mat4 view = mat4(mat3(view));
    gl_Position = projection * view * model * vec4(position, 1);
    frag_tex_coords = position;
    //frag_tex_coords = tex_coord;   //Pour bunny 
    gl_Position.z = 0;
}

