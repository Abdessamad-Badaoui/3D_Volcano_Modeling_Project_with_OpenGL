#version 330 core

// input attribute variable, given per vertex
in vec3 position;
in vec3 color;

// global matrix variables
uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;
uniform vec3 w_camera_position;


// interpolated color for fragment shader, intialized at vertices
out vec3 fragment_color;
out float dis;

void main() {
    // initialize interpolated colors at vertices
    fragment_color = vec3(0.4, 0.4, 0.4);

    gl_Position = projection * view * vec4(position, 1);
    
    // Calcule de la distance entre la caméra et les particules du smoke.
    vec3 w_position = (model * vec4(position, 0)).xyz;
    dis = length(w_camera_position - w_position);

}