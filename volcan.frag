#version 330 core


in vec3 v_position; 
// material properties
uniform vec3 k_d;
uniform vec3 k_a;
uniform vec3 k_s;


uniform sampler2D diffuse_map;
uniform sampler2D normal_map;
in vec2 frag_tex_coords;
in vec2 uv;
in mat3 v_TBNMatrix;
in vec3 w_normal;



out vec4 out_color;

void main() {
    
    vec4 diffuseColor = texture(diffuse_map,frag_tex_coords);
    vec4 normalColor = texture(normal_map,frag_tex_coords);

    // The Phong model :
    vec3 v_normal = v_TBNMatrix * normalize((normalColor.xyz * 2.0) - 1.0);
    vec3 n1 = normalize(v_normal);
    vec3 n2 = normalize(w_normal);
    vec3 l = - vec3(10,25,0);
    vec3 r = reflect(vec3(3,15,3),n1); // On fait le reflet par rapport au normal_map.
    vec3 v = vec3(0.0,0.0,0.0) - v_position; // Camera_position = (0,0,0) in view space. 
    int s = 5;
    vec3 vector = vec3(0.7725, 0.1451, 0.1451)+ vec3(0.7529, 0.1843, 0.0431)*(max( dot(n2,vec3(0,-1,0)) ,0.0)) + vec3(1,1,1)*pow(max( dot(normalize(r),normalize(v)) ,0),s);
    vec4 color1 = vec4(vector,1);

    out_color = mix (color1,diffuseColor,0.6);
    
}
