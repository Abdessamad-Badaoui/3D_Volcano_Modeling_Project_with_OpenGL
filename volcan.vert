#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec3 position;
in vec3 normal;
in vec3 tangent;
in vec3 bitangent;
in vec2 uv_vertex;
in vec2 tex_coord;

out vec2 frag_tex_coords;
out vec3 v_position;
out vec3 w_normal;

out vec2 uv;
out mat3 v_TBNMatrix;


void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    frag_tex_coords = position.xy;
        
    //compute the vertex position and normal in world or view coordinates
    v_position = (view * model * vec4(position, 0)).xyz;
    w_normal = (model * vec4(position, 0)).xyz;

    // Normal Mapping

    uv = uv_vertex;

    mat3 modelView_3 = mat3(view*model);
    vec3 tangent = normalize(tangent);
    vec3 bitangent = normalize(bitangent);
    vec3 normal = normalize(normal);

    v_TBNMatrix = modelView_3 * mat3(tangent,bitangent,normal);

}

