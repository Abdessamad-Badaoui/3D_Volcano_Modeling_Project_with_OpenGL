#version 330 core

// input attribute variable, given per vertex
in vec3 position;
in vec3 color;

// global matrix variables
uniform mat4 view;
uniform mat4 projection;

// interpolated color for fragment shader, intialized at vertices
out vec3 fragment_color;

void main() {
    // initialize interpolated colors at vertices
    fragment_color = vec3(0.8, 0.0, 0.0);

    // tell OpenGL how to transform the vertex to clip coordinates

    gl_Position = projection * view * vec4(position, 1);
}
