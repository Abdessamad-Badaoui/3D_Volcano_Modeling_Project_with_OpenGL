#version 330 core

// receiving interpolated color for fragment shader
in vec3 fragment_color;

// output fragment color for OpenGL
out vec4 out_color;

in float dis;

void main() {
    out_color = vec4(fragment_color, 1 - dis/1000 );
}
