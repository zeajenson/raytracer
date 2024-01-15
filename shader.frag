#version 450
#extension GL_ARB_separate_shader_objects : enable

struct Sphere{
    float x,y,z,radius;
};

layout(location = 0) in vec4 in_color;
layout(location = 0) out vec4 out_color;

layout(std140, binding = 0) buffer Spheres{
    Sphere spheres[10];
};

//TODO: get from uniform buffer.
float width = 1024, height = 1024;

float vertical_field_of_view = 20.0;

vec3 look_from = vec3(-12, 5, 12);
vec3 look_at = vec3(0, 0, -1);
vec3 vup = vec3(0, 1, 0); // Cameras relative up direction.
float distance_to_focus = length(look_from - look_at);

float h = tan(radians(vertical_field_of_view) / 2);
float viewport_height = 2 * h * distance_to_focus;
//Assume square.
float viewport_width = viewport_height * (width/height);

//Look at
vec3 w = normalize(look_from - look_at);
vec3 u = normalize(cross(vup, w)); 
vec3 v = cross(w,u);

vec3 viewport_u = 1 * u;
vec3 viewport_v = 1 * -v;
vec3 pixel_delta_u = viewport_u / width;
vec3 pixel_delta_v = viewport_v / height;
vec3 viewport_upper_left = look_from - (distance_to_focus * w) - viewport_u / 2 - viewport_v / 2;
vec3 first_pixel_location = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);


float ray_tmin = 0.001, ray_tmax = 9999999.99;

void main() {
    
    float x =  gl_FragCoord.x/width;
    float y =  gl_FragCoord.y/height;

    vec3 pixel_center = first_pixel_location + (gl_FragCoord.x * pixel_delta_u) + (gl_FragCoord.y * pixel_delta_v);
    vec3 ray_direction = look_from - pixel_center;
    
    out_color = vec4(ray_direction,0);

    for(int i = 1; i < 2; i++){
        Sphere sphere = spheres[i];

        vec3 sphere_position = vec3(sphere.x, sphere.y, sphere.z);

//        vec2 pixel_locatin = vec2(x, y);
//        vec2 sphere_locatin = vec2(sphere.x, sphere.y);
//        if(distance(pixel_locatin, sphere_locatin) < sphere.radius){
//            out_color += vec4(sphere_position, 0);
//        }

        if(sphere.radius < 0.0001){
            out_color = vec4(1,0,0,0);
            continue;
        } 

        vec3 sphere_origin = look_from - sphere_position;
        float a = dot(ray_direction, ray_direction);
        float half_b = dot(sphere_origin, ray_direction);
        float c = dot(sphere_origin, sphere_origin) - (sphere.radius * sphere.radius);
        float discriminant = half_b * half_b - a * c;
        if (discriminant < 0) {
                continue;
        }

        float discriminant_square_root = sqrt(discriminant);
        float root = (-half_b - discriminant_square_root) / a;
        if (root <= ray_tmin || root >= ray_tmax) {
            root = (-half_b + discriminant_square_root) / a;
            if (root <= ray_tmin || root >= ray_tmax) {
                out_color *= vec4(0.1,0.1,0.1,0);
                continue;
            }
        }

        vec3 hit_point = look_from + (root * ray_direction);
        vec3 outward_normal = (hit_point - sphere_position) / sphere.radius;

        bool is_front_face = dot(ray_direction, outward_normal) < 0;

        vec3 face_normal;
        if(is_front_face){
            face_normal = outward_normal;
        }else{
            face_normal = -outward_normal;
        }

        out_color = vec4(face_normal,0);
    }
}
