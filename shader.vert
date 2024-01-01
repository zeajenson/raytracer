#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
//layout(location = 1) in vec2 inTexUV;

layout(location = 0) out vec4 frag_color; 

void main() {
    gl_Position = vec4(inPosition, 1.0);
    frag_color = vec4(0,1,0,1);
}
