#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <ostream>
#include <random>
#include <thread>
#include <vector>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

enum { x, y, z };

constexpr auto reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
}

auto dev = std::random_device();
std::mt19937 generator(dev());
inline double random_double(double min = 0, double max = RAND_MAX + 1.0) noexcept {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(generator);
}

inline float random_float(float min = 0, float max = RAND_MAX + 1.0) noexcept {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(generator);
}

struct LAB {
        float l, a, b;
};

struct RGB {
        float r, g, b;
        constexpr auto operator*=(RGB rgb) noexcept {
                r *= rgb.r;
                g *= rgb.b;
                b *= rgb.g;
        }

        constexpr auto operator*=(double v) noexcept {
                r *= v;
                g *= v;
                b *= v;
        }

        constexpr auto operator+=(RGB rgb) noexcept {
                r += rgb.r;
                g += rgb.b;
                b += rgb.g;
        }

        constexpr auto operator+=(double v) noexcept {
                r *= v;
                g *= v;
                b *= v;
        }
};

constexpr auto operator*(RGB const &rgb1, RGB const &rgb2) noexcept {
        return RGB{
                .r = rgb1.r * rgb2.r,
                .g = rgb1.g * rgb2.g,
                .b = rgb1.b * rgb2.b,
        };
}

constexpr auto operator*(RGB const &rgb1, float v) noexcept {
        return RGB{
                .r = rgb1.r * v,
                .g = rgb1.g * v,
                .b = rgb1.b * v,
        };
}

constexpr auto operator*(float v, RGB const &rgb1) noexcept { return rgb1 * v; }

constexpr auto operator+(RGB const &rgb1, RGB const &rgb2) noexcept {
        return RGB{
                .r = rgb1.r + rgb2.r,
                .g = rgb1.g + rgb2.g,
                .b = rgb1.b + rgb2.b,
        };
}

constexpr auto operator+(RGB const &rgb1, float v) noexcept {
        return RGB{
                .r = rgb1.r + v,
                .g = rgb1.g + v,
                .b = rgb1.b + v,
        };
}

constexpr auto operator+(float v, RGB const &rgb1) noexcept { return rgb1 + v; }

struct vec3 {
        float e[3];
        constexpr auto operator[](std::size_t index) const noexcept { return e[index]; }
        constexpr auto operator-() const noexcept { return vec3{-e[0], -e[1], -e[2]}; }

        constexpr auto operator+=(vec3 const &v) noexcept {
                e[x] += v[x];
                e[y] += v[y];
                e[z] += v[z];
        }

        constexpr auto operator*=(double const &t) noexcept {
                e[x] *= t;
                e[y] *= t;
                e[z] *= t;
        }
        constexpr auto length() const noexcept -> double;
        constexpr auto normalize() const noexcept -> vec3;
        static inline auto random(double min = 0, double max = RAND_MAX + 1.0) { return vec3{random_float(min, max), random_float(min, max), random_float(min, max)}; }
        constexpr auto near_zero() const noexcept {
                constexpr auto s = 1e-8;
                return (std::fabs(e[x]) < s && std::fabs(e[y]) < s && std::fabs(e[z]) < s);
        }
};

struct rgba {
        std::uint8_t r, g, b, a;
};

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

constexpr double degreees_to_radians(double degrees) noexcept { return degrees * pi / 180.0; }

constexpr auto operator*(vec3 const &v, float t) noexcept {
        return vec3{{
                v.e[0] * t,
                v.e[1] * t,
                v.e[2] * t,
        }};
}

constexpr auto operator*(double t, vec3 const &v) noexcept { return v * t; }

constexpr auto operator/(vec3 v, double t) noexcept { return v * (1 / t); }

constexpr auto vec3::normalize() const noexcept -> vec3 { return (*this) / this->length(); }

constexpr auto operator-(vec3 const &u, vec3 const &v) noexcept { return vec3{u[0] - v[0], u[1] - v[1], u[2] - v[2]}; }

constexpr auto operator-(vec3 const &u, float t) noexcept { return vec3{u[0] - t, u[1] - t, u[2] - t}; }

constexpr vec3 operator+(const vec3 &u, const vec3 &v) noexcept { return vec3{u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]}; }

constexpr auto dot(vec3 const &a, vec3 const &b) noexcept { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

constexpr auto cross(vec3 const &u, vec3 const &v) { return vec3{u.e[y] * v.e[z] - u.e[z] * v.e[y], u.e[z] * v.e[x] - u.e[x] * v.e[z], u.e[x] * v.e[y] - u.e[y] * v.e[x]}; }

constexpr auto vec3::length() const noexcept -> double { return std::sqrt(dot(*this, *this)); }

constexpr auto reflect(vec3 const &v, vec3 const &n) noexcept { return v - 2 * dot(v, n) * n; }

constexpr auto refract(vec3 const &uv, vec3 const &n, double etai_over_etat) noexcept {
        auto cos_theta = std::min(dot(-uv, n), 1.0f);
        auto r_out_perp = etai_over_etat * (uv + cos_theta * n);
        auto r_out_perallel = -std::sqrt(std::fabs(1.0 - dot(r_out_perp, r_out_perp))) * n;
        return r_out_perp + r_out_perallel;
}

inline auto generate_random_vector_in_unit_sphere() noexcept {
        for (;;) {
                auto p = vec3::random(-1, 1);
                if (dot(p, p) < 1) {
                        return p;
                }
        }
}

inline auto generate_random_vector_in_unit_disc() noexcept {
        while (1) {
                auto p = vec3{random_float(-1, 1), random_float(-1, 1), 0};
                if (dot(p, p) < 1) {
                        return p;
                }
        }
}

inline auto generate_random_unit_vector_on_hemisphere(vec3 const &normal) noexcept {
        vec3 on_unit_sphere = generate_random_vector_in_unit_sphere().normalize();
        if (dot(on_unit_sphere, normal) > 0.0) {
                return on_unit_sphere;
        } else {
                return -on_unit_sphere;
        }
}

struct Ray {
        vec3 origin, direction;
        constexpr auto at(double t) const noexcept { return origin + t * direction; }
};

inline auto write_rgba(rgba rgba) noexcept {
        auto [r, g, b, a] = rgba;
        std::puts(std::format("{} {} {}", r, g, b).c_str());
}

inline auto calculate_void_color(Ray const &ray) noexcept {
        auto a = 0.5 * (ray.direction.normalize().e[y] + 1.0);
        auto color = ((1.0 - a) * RGB{1, 1, 1} + a * RGB{0.5, 0.7, 1.0});
        // auto [r,g,b] = generate_random_vector_in_unit_sphere().e;
        // auto color = vec3{std::clamp(r, 0.03,1.0), std::clamp(g, 0.01, 1.0),
        // std::clamp(b, 0.02, 1.0)};
        return color;
}

struct Hit_Record {
        vec3 point;
        vec3 normal;
        double t;
        size_t entity_index;
        bool front_face;
};

constexpr auto calculate_face_normal(Ray const &ray, vec3 const &outward_normal) noexcept {
        auto front_face = dot(ray.direction, outward_normal) < 0;
        auto normal = front_face ? outward_normal : -outward_normal;
        struct {
                vec3 normal;
                bool front_face;
        } face_normal{normal, front_face};
        return face_normal;
};

inline auto defocus_disk_sample(vec3 const &camera_center, vec3 const &defocus_disk_u, vec3 const &defocus_disk_v) noexcept {
        auto point = generate_random_vector_in_unit_disc();
        return camera_center + (point[x] * defocus_disk_u) + (point[y] * defocus_disk_v);
}

struct Ecs {
        enum class Type { sphere } type;
        std::vector<Type> entities;
        std::vector<size_t> geometry_indices;
        struct Sphere {
                vec3 center;
                double radius;
        };
        std::vector<Sphere> spheres;
        std::vector<size_t> entity_from_sphere_index;
        std::vector<size_t> material_indices;

        constexpr auto add_sphere(Sphere sphere) noexcept {
                entities.push_back(Type::sphere);
                material_indices.push_back(0);
                spheres.push_back(sphere);
                entity_from_sphere_index.push_back(entities.size() - 1);
                geometry_indices.push_back(spheres.size() - 1);
                return entities.size() - 1;
        }

        struct Material {
                RGB albedo;
                std::size_t properties_index;
                enum class Type {
                        lambert,
                        metal,
                        dielectric,
                } type;
        };

        std::vector<Material> materials = {{{1, 0, 1}, 0, Material::Type::lambert}};
        // Accessed through properties index;
        std::vector<double> metal_fuzz_radiai;
        std::vector<double> transmision_ir;

        constexpr auto add_lambert_material(RGB albedo) noexcept {
                materials.push_back({albedo, 0, Material::Type::lambert});
                return materials.size() - 1;
        }

        constexpr auto add_transmission_material(RGB albedo, double ir) noexcept {
                materials.push_back({albedo, transmision_ir.size(), Material::Type::dielectric});
                transmision_ir.push_back(ir);
                return materials.size() - 1;
        }

        constexpr auto add_metal_material(RGB albedo, double fuzz_radius) noexcept {
                materials.push_back({albedo, metal_fuzz_radiai.size(), Material::Type::metal});
                metal_fuzz_radiai.push_back(fuzz_radius);
                return materials.size() - 1;
        }

        constexpr auto assign_material(std::size_t material_index, std::size_t index) { material_indices[index] = material_index; }

        struct Scatter_Values {
                Ray ray;
                RGB attenuation_color;
        };

        constexpr auto scatter(Ray const &ray, Hit_Record const &hit) const noexcept -> std::optional<Scatter_Values> {
                auto material = materials[material_indices[hit.entity_index]];
                auto attenuation = material.albedo;
                switch (material.type) {
                case Ecs::Material::Type::lambert: {
                        auto direction = hit.normal + generate_random_vector_in_unit_sphere();
                        auto new_ray = Ray(hit.point, direction);
                        if (direction.near_zero())
                                direction = hit.normal;

                        return Scatter_Values{new_ray, attenuation};
                }
                case Ecs::Material::Type::metal: {
                        auto reflected = reflect(ray.direction.normalize(), hit.normal);
                        return Scatter_Values{
                                .ray = Ray{hit.point, reflected + (metal_fuzz_radiai[material.properties_index] * generate_random_vector_in_unit_sphere().normalize())},
                                .attenuation_color = attenuation,
                        };
                }
                case Ecs::Material::Type::dielectric: {
                        double refraction_ratio = hit.front_face ? (1.0 / transmision_ir[material.properties_index]) : transmision_ir[material.properties_index];

                        auto unit_direction = ray.direction.normalize();
                        auto refracted = refract(unit_direction, hit.normal, refraction_ratio);

                        double cos_theta = std::min(dot(-unit_direction, hit.normal), 1.0f);
                        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

                        bool cannot_refact = refraction_ratio * sin_theta > 1.0;
                        auto direction = vec3{};

                        if (cannot_refact || reflectance(cos_theta, refraction_ratio) > random_double()) {
                                direction = reflect(unit_direction, hit.normal);
                        } else {
                                direction = refract(unit_direction, hit.normal, refraction_ratio);
                        }

                        return Scatter_Values{
                                .ray = Ray(hit.point, direction),
                                .attenuation_color = attenuation,
                        };
                }
                }
                return std::nullopt;
        }

        constexpr auto hit(std::size_t index, Ray const &ray, double ray_tmin, double ray_tmax) const noexcept -> std::optional<Hit_Record> {
                switch (entities[index]) {
                case Type::sphere: {
                        auto [center, radius] = spheres[geometry_indices[index]];
                        // auto sphere_hit = hit_sphere(center,
                        // radius, ray);
                        auto sphere_origin = ray.origin - center;
                        auto a = dot(ray.direction, ray.direction);
                        auto half_b = dot(sphere_origin, ray.direction);
                        auto c = dot(sphere_origin, sphere_origin) - (radius * radius);
                        auto discriminant = half_b * half_b - a * c;
                        if (discriminant < 0) {
                                return std::nullopt;
                        }
                        auto discriminant_square_root = std::sqrt(discriminant);

                        // auto sphere_hit = (-half_b -
                        // std::sqrt(discriminant)) / a;
                        auto root = (-half_b - discriminant_square_root) / a;
                        if (root <= ray_tmin or root >= ray_tmax) {
                                root = (-half_b + discriminant_square_root) / a;
                                if (root <= ray_tmin or root >= ray_tmax) {
                                        return std::nullopt;
                                }
                        }

                        // if (sphere_hit > 0) {
                        //   auto normal = (ray.at(sphere_hit)
                        //   - vec3{0, 0, -1}).to_unit();
                        //   return Hit_Record{sphere_origin,
                        //   normal, sphere_hit};
                        // }
                        // return vec3{1,0,0};
                        // return 0.5 * vec3{normal[0] + 1,
                        // normal[1] + 1, normal[2] + 1};
                        auto point = ray.at(root);
                        auto outward_normal = (point - center) / radius;
                        auto [normal, is_front_face] = calculate_face_normal(ray, outward_normal);

                        return Hit_Record{.point = point, .normal = normal, .t = root, .entity_index = index, .front_face = is_front_face};
                }
                default:
                        return std::nullopt;
                }
        }
        constexpr auto calculate_hit(Ray const &ray, double ray_tmin, double ray_tmax) const noexcept -> std::optional<Hit_Record> {
                // TODO: find the nearest object hit.
                auto hit_record = std::optional<Hit_Record>(std::nullopt);
                for (auto i = 0; i < entities.size(); ++i) {
                        auto current_hit = hit(i, ray, ray_tmin, ray_tmax);
                        if (current_hit) {
                                if (not hit_record) {
                                        hit_record = current_hit;
                                } else if (current_hit->t < hit_record->t) {
                                        hit_record = current_hit;
                                }
                        }
                }
                return hit_record;
        };
};

constexpr auto write_color(vec3 color, double samples_per_pixel) noexcept {
        auto scale = 1.0 / samples_per_pixel;
        auto [r, g, b] = color.e;
        r *= scale;
        g *= scale;
        b *= scale;
        // Gamma correct
        r = std::sqrt(r);
        g = std::sqrt(g);
        b = std::sqrt(b);

        auto ir = static_cast<std::uint8_t>(255 * std::clamp(r, 0.0f, 1.0f));
        auto ig = static_cast<std::uint8_t>(255 * std::clamp(g, 0.0f, 1.0f));
        auto ib = static_cast<std::uint8_t>(255 * std::clamp(b, 0.0f, 1.0f));
        write_rgba({ir, ig, ib, 0});
};

constexpr auto convert_color(RGB color, double samples_per_pixel) noexcept {
        auto scale = 1.0 / samples_per_pixel;
        auto [r, g, b] = color;
        r *= scale;
        g *= scale;
        b *= scale;
        // Gamma correct
        r = std::sqrt(r);
        g = std::sqrt(g);
        b = std::sqrt(b);

        auto ir = static_cast<std::uint8_t>(255 * std::clamp(r, 0.0f, 1.0f));
        auto ig = static_cast<std::uint8_t>(255 * std::clamp(g, 0.0f, 1.0f));
        auto ib = static_cast<std::uint8_t>(255 * std::clamp(b, 0.0f, 1.0f));
        return rgba{ir, ig, ib, 0};
};

constexpr auto calculate_hit_color(Hit_Record const &hit) noexcept {
        auto normal = hit.normal;
        auto color = 0.5 * vec3{normal[0] + 1, normal[1] + 1, normal[2] + 1};
        return color;
};

constexpr auto settup_world() noexcept {
        auto world = Ecs{};
        auto middleboi = world.add_sphere({{0, 0, -1}, 0.5});
        auto bigboi = world.add_sphere({{0, -100.5, -1}, 100});
        auto metal1 = world.add_sphere({{-1, 0, -1}, 0.75});
        auto metal2 = world.add_sphere({{1, 0, -1}, 0.25});
        auto metal3 = world.add_sphere({{1, 0.5, -1}, 0.25});
        world.materials.front().albedo = {1, 0.01, 0.01};
        world.assign_material(world.add_lambert_material({0.01, 0.1, 0.01}), bigboi);
        world.assign_material(world.add_metal_material({.5, .5, 0}, 1.0), metal1);
        world.assign_material(world.add_metal_material({.5, .5, .5}, .01), world.add_sphere({{-2, 0, -2}, 0.75}));
        world.assign_material(world.add_lambert_material({0, 0, 1}), world.add_sphere({{1.5, 0, 1.5}, 0.75}));
        world.assign_material(world.add_transmission_material({1, 1, 1}, 1.05), world.add_sphere({{.5, 0, .5}, 0.75}));
        world.assign_material(world.add_metal_material({.5, .5, .5}, .01), world.add_sphere({{-2, 3, -2}, 0.75}));
        world.assign_material(world.add_metal_material({.5, .1, .1}, .01), world.add_sphere({{4, 0, -4}, 0.75}));
        world.assign_material(world.add_metal_material({.1, .5, .1}, .01), world.add_sphere({{2, 0, -4}, 0.75}));
        world.assign_material(world.add_metal_material({.1, .1, .5}, .01), world.add_sphere({{0, 0, -4}, 0.75}));
        world.assign_material(world.add_metal_material({.5, .1, .5}, .01), world.add_sphere({{-2, 0, -4}, 0.75}));
        world.assign_material(world.add_metal_material({.3, .5, .5}, .01), world.add_sphere({{6, 0, -4}, 0.75}));
        world.assign_material(world.add_metal_material({0.01, 0.25, 0.25}, .2), metal2);
        world.assign_material(world.add_metal_material({0.01, 1, 0}, 0), metal3);
        world.assign_material(world.add_transmission_material({0.25, 0.02, 0.25}, 1.5), world.add_sphere({{.5, .1, -1}, 0.25}));
        world.assign_material(world.add_transmission_material({1, 1, 1}, 1.5), world.add_sphere({{.5, .25, -.75}, 0.25}));
        auto spheres = 10;
        auto sphere_size = 0.1f;
        auto material_index = 0;
        auto materials_material_index = 0;
        auto start = -((spheres * sphere_size * 1.5f) / 2);
        for (int i = spheres; i; --i) {
                auto offset = start + (sphere_size * 2 * i);
                world.assign_material(materials_material_index, world.add_sphere({{offset, -0.5, -0.5}, sphere_size}));
                if (materials_material_index == world.materials.size() - 1)
                        materials_material_index = 0;
                else
                        ++materials_material_index;
        }
        return world;
}

struct XY {
        uint32_t x, y, index;
};

constexpr auto generate_bruijn_sequience(std::uint32_t size) {
        auto sequence = std::vector<XY>(size);
        for (auto x = 0; x < size; ++x) {
                if (x < 1) {
                        sequence[x].x = x;
                        sequence[x].y = 0;
                        continue;
                }

                auto x1 = (sequence[x - 1].x + 0xaaaaaaab) & 0x55555555;
                sequence[x].x = x1;
                sequence[x].y = x1 << 1;
        }

        return sequence;
}

// TODO: account for images that arent square.
constexpr auto generate_z_order_mapping(std::uint64_t width, std::uint64_t height) noexcept {
        auto map = std::vector<uint32_t>(width * height);
        const auto max_size = std::max(width, height);
        auto bruijn_sequience = generate_bruijn_sequience(max_size);
        for (uint32_t x = 0; x < width; ++x) {
                for (uint32_t y = 0; y < height; ++y) {
                        auto index = bruijn_sequience[x].x | bruijn_sequience[y].y;
                        if (index >= width * height) {
                                // std::puts(std::format("index {} is outside of array\n", index).c_str());
                                continue;
                        }
                        map[x + (y * width)] = index;

                        // std::puts(std::format("x {}, y {}, index {}", x, y, index).c_str());
                }
                // std::puts("\n");
        }

        return map;
}

// TODO: figure out how to make this work with things that arent' multiples of 4
template <typename T> constexpr auto swizzle(std::vector<T> const &pixels, std::vector<uint32_t> const &map, std::uint32_t width, std::uint32_t height) noexcept {
        auto new_pixels = std::vector<T>(width * height);
        for (auto y = 0; y < height; ++y) {
                for (auto x = 0; x < width; ++x) {
                        auto index = x + (y * width);
                        auto pixel = pixels[index];
                        new_pixels[index] = pixels[map[index]];
                        // std::clog << std::format("rgb,{},{},{} x{} y{} z{}\n",pixel.r, pixel.g, pixel.b, x, y, map[index]);
                }
        }
        return new_pixels;
}

template <typename T> constexpr auto unmap_image(std::vector<T> const &pixels, std::vector<uint32_t> const &map, std::uint32_t width, std::uint32_t height) noexcept {
        auto new_pixels = std::vector<T>(width * height);
        for (auto y = 0; y < height; ++y) {
                for (auto x = 0; x < width; ++x) {
                        auto index = x + (y * width);
                        auto pixel = pixels[index];
                        new_pixels[map[index]] = pixels[index];
                        // std::clog << std::format("rgb,{},{},{} x{} y{} z{}\n",pixel.r, pixel.g, pixel.b, x, y, map[index]);
                }
        }
        return new_pixels;
}

struct Image {
        std::vector<uint8_t> r;
        std::vector<uint8_t> g;
        std::vector<uint8_t> b;
        uint64_t width, height;
};

inline auto ray_trace_stuff(uint64_t width, uint64_t height) -> Image {

        const auto image_size = width * height;

        auto image = Image{.r = std::vector<uint8_t>(image_size), .g = std::vector<uint8_t>(image_size), .b = std::vector<uint8_t>(image_size), .width = width, .height = height};

        const auto aspect_ratio = 1;
        const auto image_width = 512 * 2;
        auto image_height = static_cast<int>(image_width / aspect_ratio);
        if (image_height < 1)
                image_height = 1;
        const auto samples = 1;
        const auto bounces = 10;

        auto z_swizz = generate_z_order_mapping(image_width, image_height);

        const auto vertical_field_of_view = 20.0;
        const auto look_from = vec3{-12, 5, 12};
        const auto look_at = vec3{0, 0, -1};
        const auto vup = vec3{0, 1, 0}; // Cameras relative up direction.

        const auto distance_to_focus = (look_from - look_at).length();

        const auto defocus_angle = 0.4;
        const auto focus_distance = distance_to_focus;

        // Camera
        const auto h = std::tan(degreees_to_radians(vertical_field_of_view) / 2);
        const auto viewport_height = 2 * h * focus_distance;
        const auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);
        const auto camera_center = look_from;

        // Camera frame base vectors.
        const auto [u, v, w] = std::invoke([&] {
                const auto w = (look_from - look_at).normalize();
                const auto u = cross(vup, w).normalize();
                const auto v = cross(w, u);
                struct {
                        vec3 u, v, w;
                } uvw{
                        .u = u,
                        .v = v,
                        .w = w,
                };
                return uvw;
        });

        // Calculate the vectors across the horizontal and down the vertical
        // viewport edges.
        const auto viewport_u = viewport_width * u;
        const auto viewport_v = viewport_height * -v;

        // Calculate the horizontal and vertical delta vectors from pixel to
        // pixel.
        const auto pixel_delta_u = viewport_u / image_width;
        const auto pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        const auto viewport_upper_left = camera_center - (focus_distance * w) - viewport_u / 2 - viewport_v / 2;
        const auto first_pixel_location = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defucus disk basis vectors
        const auto defocus_radius = focus_distance * std::tan(degreees_to_radians(defocus_angle / 2.0));
        const auto defucus_disk_u = u * defocus_radius;
        const auto defucus_disk_v = v * defocus_radius;

        std::vector<rgba> pixels(image_width * image_height);
        // std::vector<RGB> samples_per_pixel(pixels.size() * samples, {1, 1,
        // 1});
        std::vector<RGB> pixel_samples(pixels.size(), {0, 0, 0});
        auto pixel_centers = std::vector<vec3>(image_size);
        for (auto y = 0; y < image_height; ++y) {
                for (auto x = 0; x < image_width; ++x) {
                        pixel_centers[x + (y * image_width)] = first_pixel_location + (x * pixel_delta_u) + (y * pixel_delta_v);
                }
        }

        pixel_centers = swizzle(pixel_centers, z_swizz, image_width, image_height);
        const auto ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample(camera_center, defucus_disk_u, defucus_disk_v);

        // auto threads = std::vector<std::thread>(std::thread::hardware_concurrency());

        // for(auto i = 0; i < std::thread::hardware_concurrency(); ++i){
        //     threads[i]
        // }

        const auto hits = std::vector<Hit_Record>();

        const auto world = settup_world();

        for (auto index = 0; index < image_height * image_width; ++index) {

                constexpr auto sample_square_size = .75;

                for (auto sample = 0; sample <= samples; ++sample) {
                        auto color = RGB{1, 1, 1};

                        auto pixel_sample_x = -(sample_square_size / 2.0) + random_double(0, sample_square_size);
                        auto pixel_sample_y = -(sample_square_size / 2.0) + random_double(0, sample_square_size);
                        auto pixel_sample_offset = (pixel_sample_x * pixel_delta_u) + (pixel_sample_y * pixel_delta_v);
                        auto pixel_sample_center = pixel_centers[index] + pixel_sample_offset;
                        auto ray_direction = pixel_sample_center - ray_origin;
                        auto has_ray = true;
                        auto is_void = false;
                        auto current_bounce_limit = bounces;
                        auto ray = Ray{ray_origin, ray_direction};
                        while (has_ray) {
                                if (not --current_bounce_limit) {
                                        color *= RGB{.001, .001, .001};
                                        break;
                                }

                                auto hit = world.calculate_hit(ray, 0.001, infinity);

                                if (hit) {
                                        auto ray_center = hit->point;
                                        auto ray_direction = hit->normal + generate_random_unit_vector_on_hemisphere(hit->normal);
                                        auto scatter = world.scatter(ray, hit.value());
                                        if (scatter) {
                                                ray = scatter->ray;
                                                color *= scatter->attenuation_color;
                                        } else {
                                                color *= {1, 0, 1};
                                        }
                                } else {
                                        color *= calculate_void_color(ray);
                                        has_ray = false;
                                }
                        }
                        // Skip over sampling
                        // the void
                        if (is_void) {
                                sample = std::midpoint(sample, samples);
                        }

                        pixel_samples[index] += color;
                        ;
                }
                //
                // for (auto y = 0; y < image_height; ++y) {
                //     std::clog << "\rScanlines remaining: " << (image_height - y) << ' ' << std::flush;
                //     for (auto x = 0; x < image_width; ++x) {
                //
                //         auto index = x + (y * image_width);
                //     }
                // }
        }

        auto value = float{0};

        for (auto y = 0; y < image_height; ++y) {
                for (auto x = 0; x < image_width; ++x) {
                        auto index = x + (y * image_width);
                        pixels[index] = convert_color(pixel_samples[index], samples);
                }
        }

        auto new_pixels = unmap_image(pixels, z_swizz, image_width, image_height);
        std::clog << "Writing...\n";
        std::puts(std::format("P3\n{} {}\n255\n", image_width, image_height).c_str());
        for (auto y = 0; y < image_height; ++y) {
                // std::clog << "\rScanlines remaining: " << (image_height - y) << ' ' << std::flush;
                for (auto x = 0; x < image_width; ++x) {
                        write_rgba(new_pixels[x + (y * image_width)]);
                }
        }
        return image;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData) {

        std::puts(std::format("validation layer :{} \n", pCallbackData->pMessage).c_str());

        return VK_FALSE;
}

template <typename T> static T load_vulkan_function(const char *name) { return reinterpret_cast<T>(glfwGetInstanceProcAddress(nullptr, name)); }

template <typename T> static T load_vulkan_function(VkInstance instance, const char *name) { return reinterpret_cast<T>(glfwGetInstanceProcAddress(instance, name)); }

// 🐈
//
int main() noexcept {
        if (not glfwInit()) {
                std::puts("Could not initialize GLFW");
                std::terminate();
        }

        const auto width = 512 * 2, height = 512 * 2;
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        const auto window = glfwCreateWindow(width, height, "raytracer", nullptr, nullptr);
        if (not window) {
                std::puts("Could get GLFW window");
                std::terminate();
        }

        // auto pixels = ray_trace_stuff(width, height);

        if (glfwVulkanSupported() == GLFW_API_UNAVAILABLE) {
                std::puts("Vulkan is not supported.\n");
        }

        auto allocator = nullptr;

        auto app_info = VkApplicationInfo{
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pNext = nullptr,
                .pApplicationName = "raytracer",
                .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                .pEngineName = "No Engine",
                .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                .apiVersion = VK_API_VERSION_1_3,
        };

        std::vector<char const *> extension_names = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

        uint32_t count;
        auto const glfw_extensions = glfwGetRequiredInstanceExtensions(&count);
        extension_names.insert(extension_names.end(), glfw_extensions, glfw_extensions + count);

        for (auto &&name : extension_names) {
                std::puts(name);
        }

        std::vector<char const *> layer_names = {"VK_LAYER_KHRONOS_validation"};
        auto instantce_info = VkInstanceCreateInfo{
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .pApplicationInfo = &app_info,
                .enabledLayerCount = static_cast<uint32_t>(layer_names.size()),
                .ppEnabledLayerNames = layer_names.data(),
                .enabledExtensionCount = static_cast<uint32_t>(extension_names.size()),
                .ppEnabledExtensionNames = extension_names.data(),
        };

        VkInstance instance;
        auto const vk_create_instance = load_vulkan_function<PFN_vkCreateInstance>("vkCreateInstance");

        if (vk_create_instance(&instantce_info, allocator, &instance) not_eq VK_SUCCESS) {
                std::puts("Unable to create vulkan instance.\n");
        }
        auto debug_messenger_info = VkDebugUtilsMessengerCreateInfoEXT{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                .pNext = nullptr,
                .flags = {},
                .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                .pfnUserCallback = debug_callback,
        };

        VkDebugUtilsMessengerEXT debug_utils_messenger;
        auto const vk_create_debug_utils_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(glfwGetInstanceProcAddress(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (not vk_create_debug_utils_messenger) {
                std::puts("unable to load debug utils messenger.\n");
        } else if (vk_create_debug_utils_messenger(instance, &debug_messenger_info, allocator, &debug_utils_messenger) not_eq VK_SUCCESS) {
                std::puts("unable to create debug utils messenger.\n");
        }

        VkSurfaceKHR surface;
        if (glfwCreateWindowSurface(instance, window, allocator, &surface) not_eq VK_SUCCESS) {
                std::puts("Unable to get surface");
                std::terminate();
        }

        uint32_t device_count = 0;
        auto const enumerate_physical_devices = load_vulkan_function<PFN_vkEnumeratePhysicalDevices>(instance, "vkEnumeratePhysicalDevices");
        ;
        if (enumerate_physical_devices(instance, &device_count, nullptr) not_eq VK_SUCCESS or device_count == 0) {
                std::puts("no physical device.");
                std::terminate();
        }
        VkPhysicalDevice physical_devices[device_count];
        enumerate_physical_devices(instance, &device_count, physical_devices);

        auto physical_device = physical_devices[0];

        auto const [graphics_index, present_index, compute_index] = std::invoke([&] {
                auto const get_physical_device_queue_family_properties = load_vulkan_function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
                uint32_t property_count = 0;
                get_physical_device_queue_family_properties(physical_device, &property_count, nullptr);
                VkQueueFamilyProperties properties[property_count];
                get_physical_device_queue_family_properties(physical_device, &property_count, properties);

                struct {
                        int32_t graphics_index;
                        int32_t present_index;
                        int32_t compute_index;
                } indices{-1, -1, -1};

                auto const get_physical_device_surface_support = load_vulkan_function<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(instance, "vkGetPhysicalDeviceSurfaceSupportKHR");
                for (auto i = 0; i < property_count; ++i) {
                        auto const &property = properties[i];
                        if (indices.graphics_index < 0 and property.queueFlags & VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT) {
                                indices.graphics_index = i;
                        }
                        if(indices.compute_index < 0 and property.queueFlags & VK_QUEUE_COMPUTE_BIT){
                                indices.compute_index = i;
                        }
                        VkBool32 surface_is_supported = VK_FALSE;
                        if (get_physical_device_surface_support(physical_device, i, surface, &surface_is_supported) not_eq VK_SUCCESS) {
                                std::puts(std::format("device surface support for queue index {} is not supported", i).c_str());
                                continue;
                        }
                        if (indices.present_index < 0 and surface_is_supported) {
                                indices.present_index = i;
                        }

                        if (indices.graphics_index >= 0 and indices.present_index >= 0 and indices.compute_index >= 0) {
                                break;
                        }
                }
                return indices;
        });

        float graphics_queue_prioraties[] = {1.0f};

        char const *device_extensions[1] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        auto const device_features = VkPhysicalDeviceFeatures{};

        auto const queue_create_infos = std::array{
                VkDeviceQueueCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = {},
                        .queueFamilyIndex = static_cast<uint32_t>(graphics_index),
                        .queueCount = 1,
                        .pQueuePriorities = graphics_queue_prioraties,
                },
        };

        auto const device_create_info = VkDeviceCreateInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .queueCreateInfoCount = queue_create_infos.size(),
                .pQueueCreateInfos = queue_create_infos.data(),
                .enabledLayerCount = static_cast<uint32_t>(layer_names.size()),
                .ppEnabledLayerNames = layer_names.data(),
                .enabledExtensionCount = 1,
                .ppEnabledExtensionNames = device_extensions,
                .pEnabledFeatures = &device_features,
        };

        auto const create_device = load_vulkan_function<PFN_vkCreateDevice>(instance, "vkCreateDevice");
        VkDevice device;
        if (create_device(physical_device, &device_create_info, allocator, &device) not_eq VK_SUCCESS) {
                std::puts("unable to create device");
                std::exit(42);
        }

        auto const get_queue = load_vulkan_function<PFN_vkGetDeviceQueue>(instance, "vkGetDeviceQueue");
        VkQueue graphics_queue;
        get_queue(device, graphics_index, 0, &graphics_queue);

        VkSurfaceCapabilitiesKHR surface_capabilities;
        if (load_vulkan_function<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")(physical_device, surface, &surface_capabilities) not_eq VK_SUCCESS) {
                std::exit(39393);
        }

        auto const get_surface_format = load_vulkan_function<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(instance, "vkGetPhysicalDeviceSurfaceFormatsKHR");
        uint32_t format_count = 0;
        if (get_surface_format(physical_device, surface, &format_count, nullptr) not_eq VK_SUCCESS) {
                std::exit(39393);
        }
        VkSurfaceFormatKHR formats[format_count];
        get_surface_format(physical_device, surface, &format_count, formats);
        auto const surface_format = formats[0];

        auto const get_surface_present_mode = load_vulkan_function<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(instance, "vkGetPhysicalDeviceSurfacePresentModesKHR");
        uint32_t present_mode_count = 0;
        if (get_surface_present_mode(physical_device, surface, &present_mode_count, nullptr) not_eq VK_SUCCESS) {
                std::exit(39393);
        }
        VkPresentModeKHR present_modes[present_mode_count];
        get_surface_present_mode(physical_device, surface, &format_count, present_modes);
        auto const surface_present_mode = present_modes[0];

        int32_t window_width, window_height;
        glfwGetFramebufferSize(window, &window_width, &window_height);
        auto min_image_extent = surface_capabilities.minImageExtent;
        auto max_image_extent = surface_capabilities.maxImageExtent;
        auto const image_extent = VkExtent2D{
                .width = std::clamp<uint32_t>(static_cast<uint32_t>(window_width), min_image_extent.width, max_image_extent.width),
                .height = std::clamp<uint32_t>(static_cast<uint32_t>(window_width), min_image_extent.height, max_image_extent.height),
        };

        uint32_t image_array_layers = 0;

        auto sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
        auto queue_family_indices = std::vector<uint32_t>();
        queue_family_indices.push_back(graphics_index);
        queue_family_indices.push_back(compute_index);
        if (graphics_index not_eq present_index) {
                queue_family_indices.push_back(present_index);
                sharing_mode = VK_SHARING_MODE_CONCURRENT;
        }

        auto const swapchain_create_info = VkSwapchainCreateInfoKHR{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .pNext = nullptr,
                .flags = {},
                .surface = surface,
                .minImageCount = surface_capabilities.minImageCount + 1,
                .imageFormat = surface_format.format,
                .imageColorSpace = surface_format.colorSpace,
                .imageExtent = image_extent,
                .imageArrayLayers = 1,
                .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .imageSharingMode = sharing_mode,
                .queueFamilyIndexCount = static_cast<uint32_t>(queue_family_indices.size()),
                .pQueueFamilyIndices = queue_family_indices.data(),
                .preTransform = surface_capabilities.currentTransform,
                .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                .presentMode = surface_present_mode,
                .clipped = VK_TRUE,
                .oldSwapchain = nullptr,
        };

        VkSwapchainKHR swapchain;
        auto const create_swapchain = load_vulkan_function<PFN_vkCreateSwapchainKHR>(instance, "vkCreateSwapchainKHR");
        if (create_swapchain(device, &swapchain_create_info, allocator, &swapchain) not_eq VK_SUCCESS) {
                std::puts("unable to create swapchain");
                std::exit(2323232);
        }

        auto const get_swapchain_images = load_vulkan_function<PFN_vkGetSwapchainImagesKHR>(instance, "vkGetSwapchainImagesKHR");
        uint32_t image_count;
        if (get_swapchain_images(device, swapchain, &image_count, nullptr) not_eq VK_SUCCESS) {
                std::exit(2424242);
        }

        auto swapchain_images = std::vector<VkImage>(image_count);
        get_swapchain_images(device, swapchain, &image_count, swapchain_images.data());

        auto swapchain_image_views = std::vector<VkImageView>(image_count);
        auto const create_image_views = load_vulkan_function<PFN_vkCreateImageView>(instance, "vkCreateImageView");
        for (auto i = 0; i < image_count; ++i) {
                auto const subresource_range = VkImageSubresourceRange{
                        .aspectMask = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                };
                auto const image_view_create_info = VkImageViewCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = {},
                        .image = swapchain_images[i],
                        .viewType = VK_IMAGE_VIEW_TYPE_2D,
                        .format = surface_format.format,
                        .components = {},
                        .subresourceRange = subresource_range,
                };
                if (create_image_views(device, &image_view_create_info, allocator, &swapchain_image_views[i]) not_eq VK_SUCCESS) {
                        std::puts("unable to create swapchain image views");
                        std::exit(52);
                }
        }

        auto const color_attachment = VkAttachmentDescription{
                .flags = {},
                .format = surface_format.format,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        auto const color_attachment_refrence = VkAttachmentReference{
                .attachment = 0,
                .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        auto const get_physical_device_format_properties = load_vulkan_function<PFN_vkGetPhysicalDeviceFormatProperties>(instance, "vkGetPhysicalDeviceFormatProperties");
        if (not get_physical_device_format_properties)
                exit(2323);

        auto depth_format = std::optional<VkFormat>(std::nullopt);
        for (auto format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT}) {
                VkFormatProperties physical_device_format_properties;
                get_physical_device_format_properties(physical_device, format, &physical_device_format_properties);
                auto features = physical_device_format_properties.optimalTilingFeatures;
                if ((features & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == features) {
                        depth_format = format;
                        break;
                }
        }
        if (not depth_format) {
                std::puts("unable to find depth format");
                exit(7894238);
        }

        auto const depth_attachment = VkAttachmentDescription{
                .flags = {},
                .format = depth_format.value(),
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
        };

        // auto const depth_attachement_refrence = VkAttachmentReference{
        //         .attachment = 1,
        //         .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
        // };

        auto const subpass = VkSubpassDescription{
                .flags = {},
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .colorAttachmentCount = 1,
                .pColorAttachments = &color_attachment_refrence,
        };

        VkPipelineStageFlags stage_mask_bits = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

        auto const subpass_dependency = VkSubpassDependency{
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = stage_mask_bits,
                .dstStageMask = stage_mask_bits,
                .srcAccessMask = {},
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .dependencyFlags = {},
        };

        auto const attachemnts = std::array{
                color_attachment,
                // depth_attachment
        };

        auto const render_pass_create_info = VkRenderPassCreateInfo{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .attachmentCount = attachemnts.size(),
                .pAttachments = attachemnts.data(),
                .subpassCount = 1,
                .pSubpasses = &subpass,
                .dependencyCount = 1,
                .pDependencies = &subpass_dependency,
        };

        VkRenderPass render_pass;
        auto const create_render_pass = load_vulkan_function<PFN_vkCreateRenderPass>(instance, "vkCreateRenderPass");
        if (not create_render_pass)
                exit(123);
        if (create_render_pass(device, &render_pass_create_info, allocator, &render_pass) not_eq VK_SUCCESS) {
                std::puts("unable to create render pass");
                std::exit(1);
        }

        auto const vk_create_shader_module = load_vulkan_function<PFN_vkCreateShaderModule>(instance, "vkCreateShaderModule");
        auto create_shader_module = [&vk_create_shader_module, &device, &allocator](std::filesystem::path path) -> VkShaderModule {
                std::puts("creating shader module");
                auto file = std::ifstream(path.string(), std::ios::ate | std::ios::binary);
                auto const fileSize = (size_t)file.tellg();
                auto buffer = std::vector<char>(fileSize);
                file.seekg(0);
                file.read(buffer.data(), fileSize);
                std::puts(std::format("loading shader {} with bytecount {}", path.string(), buffer.size()).c_str());
                auto const shader_module_create_info = VkShaderModuleCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = {},
                        .codeSize = buffer.size(),
                        .pCode = reinterpret_cast<uint32_t const *>(buffer.data()),
                };
                VkShaderModule module;
                if (vk_create_shader_module(device, &shader_module_create_info, allocator, &module)) {
                        std::puts("unable to create shader module");
                        std::exit(420);
                }
                return module;
        };


        auto const vertex_shader_module = create_shader_module("shader.vert.spv");
        auto const vertex_shader_stage_create_info = VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertex_shader_module,
                .pName = "main",
        };

        auto const fragment_shader_module = create_shader_module("shader.frag.spv");
        auto const fragment_shader_stage_create_info = VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragment_shader_module,
                .pName = "main",
        };

        auto const computer_shader_module = create_shader_module("shader.comp.spv");
        auto const compute_shader_stage_create_info = VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = fragment_shader_module,
                .pName = "main",
        };

        auto const shader_stages = std::array{
                vertex_shader_stage_create_info,
                fragment_shader_stage_create_info,
        };

        struct texture_uv {
                float u, v;
        };

        auto const vertex_texture_uv_binding_description = VkVertexInputBindingDescription{
                .binding = 1,
                .stride = sizeof(texture_uv),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        auto const texture_uv_attribute = VkVertexInputAttributeDescription{
                .location = 1,
                .binding = vertex_texture_uv_binding_description.binding,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = 0,
        };

        struct vertex_position {
                float x, y, z;
        };

        auto const vertex_position_binding_description = VkVertexInputBindingDescription{
                .binding = 0,
                .stride = sizeof(vertex_position),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        auto const vertex_position_attribute = VkVertexInputAttributeDescription{
                .location = 0,
                .binding = vertex_position_binding_description.binding,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = 0,
        };

        auto const vertex_binding_dexcriptions = std::array{
                vertex_position_binding_description,
                // vertex_texture_uv_binding_description
        };
        auto const vertex_attribute_descritions = std::array{
                vertex_position_attribute,
                // texture_uv_attribute,
        };

        auto const vertex_input_info = VkPipelineVertexInputStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .vertexBindingDescriptionCount = vertex_binding_dexcriptions.size(),
                .pVertexBindingDescriptions = vertex_binding_dexcriptions.data(),
                .vertexAttributeDescriptionCount = vertex_attribute_descritions.size(),
                .pVertexAttributeDescriptions = vertex_attribute_descritions.data(),
        };

        auto const input_assembly_info = VkPipelineInputAssemblyStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .primitiveRestartEnable = VK_FALSE,
        };

        auto const viewport = VkViewport{
                .x = 0,
                .y = 0,
                .width = static_cast<float>(image_extent.width),
                .height = static_cast<float>(image_extent.height),
                .minDepth = 0,
                .maxDepth = 1,
        };

        auto const scissor = VkRect2D{
                .offset = {0, 0},
                .extent = image_extent,
        };

        auto const viewport_state_info = VkPipelineViewportStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .viewportCount = 1,
                .pViewports = &viewport,
                .scissorCount = 1,
                .pScissors = &scissor,
        };

        auto const rasterizer = VkPipelineRasterizationStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_BACK_BIT,
                .frontFace = VK_FRONT_FACE_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
        };

        auto const multisampling = VkPipelineMultisampleStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = false,
                .minSampleShading = 1,
                .pSampleMask = nullptr,
                .alphaToCoverageEnable = VK_FALSE,
                .alphaToOneEnable = VK_FALSE,
        };

        auto const color_blend_attachment = VkPipelineColorBlendAttachmentState{
                .blendEnable = VK_FALSE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };

        auto const color_blending = VkPipelineColorBlendStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .logicOpEnable = VK_FALSE,
                .attachmentCount = 1,
                .pAttachments = &color_blend_attachment,
                .blendConstants = {0.0f},
        };

        auto const dynamic_states = std::array{
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR,
                VK_DYNAMIC_STATE_LINE_WIDTH,
        };

        auto const dynamic_state = VkPipelineDynamicStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .dynamicStateCount = dynamic_states.size(),
                .pDynamicStates = dynamic_states.data(),
        };

        auto const depth_stencil = VkPipelineDepthStencilStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
                .front = {},
                .back = {},
                .minDepthBounds = 0,
                .maxDepthBounds = 1,
        };

        auto const ubo_binding = VkDescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                .pImmutableSamplers = nullptr,
        };

        auto const sampler_binding = VkDescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .pImmutableSamplers = nullptr,
        };

        auto const descriptor_set_bindings = std::array{
                // ubo_binding,
                sampler_binding,
        };

        auto const descriptor_set_info = VkDescriptorSetLayoutCreateInfo{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .bindingCount = descriptor_set_bindings.size(),
                .pBindings = descriptor_set_bindings.data(),
        };

        VkDescriptorSetLayout descriptor_set_layout;
        auto const create_descriptor_set_layout = load_vulkan_function<PFN_vkCreateDescriptorSetLayout>(instance, "vkCreateDescriptorSetLayout");
        if (create_descriptor_set_layout(device, &descriptor_set_info, allocator, &descriptor_set_layout) not_eq VK_SUCCESS) {
                std::puts("unable to create descriptor set layout");
                std::exit(1);
        }

        auto const pipeline_layout_info = VkPipelineLayoutCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pNext = nullptr, .flags = {}, .setLayoutCount = 1, .pSetLayouts = &descriptor_set_layout};

        VkPipelineLayout pipeline_layout;
        auto const create_pipeline_layout = load_vulkan_function<PFN_vkCreatePipelineLayout>(instance, "vkCreatePipelineLayout");
        if (create_pipeline_layout(device, &pipeline_layout_info, allocator, &pipeline_layout) not_eq VK_SUCCESS) {
                std::puts("unable to create pipeline layout");
                std::exit(1);
        }

        auto const pipeline_create_info = VkGraphicsPipelineCreateInfo{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .stageCount = shader_stages.size(),
                .pStages = shader_stages.data(),
                .pVertexInputState = &vertex_input_info,
                .pInputAssemblyState = &input_assembly_info,
                .pViewportState = &viewport_state_info,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pColorBlendState = &color_blending,
                .pDynamicState = &dynamic_state,
                .layout = pipeline_layout,
                .renderPass = render_pass,
        };

        std::puts("making graphics pipeline");
        VkPipeline graphics_pipeline;
        auto const create_graphics_pipelines = load_vulkan_function<PFN_vkCreateGraphicsPipelines>(instance, "vkCreateGraphicsPipelines");
        if (create_graphics_pipelines(device, nullptr, 1, &pipeline_create_info, allocator, &graphics_pipeline) not_eq VK_SUCCESS) {
                std::puts("unable to create graphics pipelines.");
                std::exit(89888);
        }

        auto const get_physical_device_memory_properties = load_vulkan_function<PFN_vkGetPhysicalDeviceMemoryProperties>(instance, "vkGetPhysicalDeviceMemoryProperties");
        VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
        get_physical_device_memory_properties(physical_device, &physical_device_memory_properties);
        auto const find_memory_type = [&physical_device_memory_properties](uint32_t memory_bits_requirement, VkMemoryPropertyFlags properties) noexcept {
                for (uint32_t memory_type_index = 0; memory_type_index < physical_device_memory_properties.memoryTypeCount; ++memory_type_index) {
                        auto memory_properties = physical_device_memory_properties.memoryTypes[memory_type_index];
                        if (memory_bits_requirement & (1 << memory_type_index) and (memory_properties.propertyFlags & properties) == properties) {
                                return memory_type_index;
                        }
                }

                std::puts("unable to find suitable memory index.");
                std::terminate();
        };

        auto const vk_create_image = load_vulkan_function<PFN_vkCreateImage>(instance, "vkCreateImage");
        auto const get_image_memory_requirements = load_vulkan_function<PFN_vkGetImageMemoryRequirements>(instance, "vkGetImageMemoryRequirements");
        auto const vk_allocate_memory = load_vulkan_function<PFN_vkAllocateMemory>(instance, "vkAllocateMemory");
        auto const bind_image_memory = load_vulkan_function<PFN_vkBindImageMemory>(instance, "vkBindImageMemory");
        auto const create_image = [&](VkFormat format, uint32_t mip_levels, VkImageTiling tiling, VkImageUsageFlags usage) {
                auto const image_info = VkImageCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                        .imageType = VK_IMAGE_TYPE_2D,
                        .format = format,
                        .extent = VkExtent3D{width, height, 1},
                        .mipLevels = mip_levels,
                        .arrayLayers = 1,
                        .samples = VK_SAMPLE_COUNT_1_BIT,
                        .tiling = tiling,
                        .usage = usage,
                        .queueFamilyIndexCount = 0,
                        .pQueueFamilyIndices = nullptr,
                        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                };
                VkImage image;
                if (vk_create_image(device, &image_info, allocator, &image) not_eq VK_SUCCESS) {
                        std::puts("unable to create vulkan image");
                        std::exit(420);
                }
                VkMemoryRequirements image_memory_requirements;
                get_image_memory_requirements(device, image, &image_memory_requirements);

                VkMemoryAllocateInfo memory_allocate_info{
                        .allocationSize = image_memory_requirements.size,
                        // TODO: check memory type exists.
                        .memoryTypeIndex = image_memory_requirements.memoryTypeBits,
                };

                VkDeviceMemory image_memory;
                if (vk_allocate_memory(device, &memory_allocate_info, allocator, &image_memory) not_eq VK_SUCCESS) {
                        std::puts("unable to allocate memory for an image");
                        std::exit(420);
                }

                if (bind_image_memory(device, image, image_memory, 0) not_eq VK_SUCCESS) {
                        std::exit(420);
                }

                struct {
                        VkImage image;
                        VkDeviceMemory memory;
                } image_stuff{image, image_memory};
                return image_stuff;
        };

        // auto [depth_image, depth_image_memory] = create_image(depth_format.value(), 1, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

        // auto const depth_subresource_range = VkImageSubresourceRange{
        //         .aspectMask = VkImageAspectFlagBits::VK_IMAGE_ASPECT_DEPTH_BIT,
        //         .baseMipLevel = 0,
        //         .levelCount = 1,
        //         .baseArrayLayer = 0,
        //         .layerCount = 1,
        // };
        // auto const depth_image_view_create_info = VkImageViewCreateInfo{
        //         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        //         .pNext = nullptr,
        //         .flags = {},
        //         .image = depth_image,
        //         .viewType = VK_IMAGE_VIEW_TYPE_2D,
        //         .format = surface_format.format,
        //         .components = {},
        //         .subresourceRange = depth_subresource_range,
        // };
        // VkImageView depth_image_view;
        // if (create_image_views(device, &depth_image_view_create_info, allocator, &depth_image_view) not_eq VK_SUCCESS) {
        //         std::exit(520);
        // }

        auto const create_frame_buffer = load_vulkan_function<PFN_vkCreateFramebuffer>(instance, "vkCreateFramebuffer");
        auto frame_buffers = std::vector<VkFramebuffer>(image_count);
        for (auto i = 0; i < image_count; ++i) {
                auto const attachments = std::array{
                        // depth_image_view,
                        swapchain_image_views[i],
                };

                auto const creat_info = VkFramebufferCreateInfo{
                        .sType = VkStructureType::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                        .flags = {},
                        .renderPass = render_pass,
                        .attachmentCount = attachments.size(),
                        .pAttachments = attachments.data(),
                        .width = image_extent.width,
                        .height = image_extent.height,
                        .layers = 1,
                };
                if (create_frame_buffer(device, &creat_info, allocator, &frame_buffers[i]) not_eq VK_SUCCESS) {
                        exit(630);
                }
        }

        auto const create_command_pool = load_vulkan_function<PFN_vkCreateCommandPool>(instance, "vkCreateCommandPool");
        if (not create_command_pool) {
                std::exit(30);
        }
        VkCommandPool command_pool;
        auto const command_pool_create_info = VkCommandPoolCreateInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = static_cast<uint32_t>(graphics_index),
        };
        if (create_command_pool(device, &command_pool_create_info, allocator, &command_pool) not_eq VK_SUCCESS) {
                std::puts("unable to create command pool.");
                std::exit(40202);
        }

        // Buffers.
        auto const vk_create_buffer = load_vulkan_function<PFN_vkCreateBuffer>(instance, "vkCreateBuffer");
        auto const get_device_buffer_memory_requirements = load_vulkan_function<PFN_vkGetBufferMemoryRequirements>(instance, "vkGetBufferMemoryRequirements");
        auto const bind_buffer_memory = load_vulkan_function<PFN_vkBindBufferMemory>(instance, "vkBindBufferMemory");
        auto const allocate_command_buffers = load_vulkan_function<PFN_vkAllocateCommandBuffers>(instance, "vkAllocateCommandBuffers");
        auto const command_begin = load_vulkan_function<PFN_vkBeginCommandBuffer>(instance, "vkBeginCommandBuffer");
        auto const command_end = load_vulkan_function<PFN_vkEndCommandBuffer>(instance, "vkEndCommandBuffer");
        auto const command_coppy_buffer = load_vulkan_function<PFN_vkCmdCopyBuffer>(instance, "vkCmdCopyBuffer");
        auto const queue_submit = load_vulkan_function<PFN_vkQueueSubmit>(instance, "vkQueueSubmit");
        auto const device_wait_idle = load_vulkan_function<PFN_vkDeviceWaitIdle>(instance, "vkDeviceWaitIdle");
        auto const map_memory = load_vulkan_function<PFN_vkMapMemory>(instance, "vkMapMemory");
        auto const unmap_memory = load_vulkan_function<PFN_vkUnmapMemory>(instance, "vkUnmapMemory");

        auto const create_buffer = [&](VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_properties, VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *buffer_memory) noexcept {
                auto const buffer_create_info = VkBufferCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                        .size = size,
                        .usage = usage,
                        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                vk_create_buffer(device, &buffer_create_info, allocator, buffer);

                VkMemoryRequirements buffer_memory_requirements;
                get_device_buffer_memory_requirements(device, *buffer, &buffer_memory_requirements);

                auto const buffer_memory_alloc_info = VkMemoryAllocateInfo{
                        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                        .allocationSize = buffer_memory_requirements.size,
                        .memoryTypeIndex = find_memory_type(buffer_memory_requirements.memoryTypeBits, memory_properties),
                };
                vk_allocate_memory(device, &buffer_memory_alloc_info, allocator, buffer_memory);
                bind_buffer_memory(device, *buffer, *buffer_memory, 0);
        };

        auto const buffer_copy = [&](VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) {
                auto const command_buffer_allocate_info = VkCommandBufferAllocateInfo{
                        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                        .commandPool = command_pool,
                        .level = VkCommandBufferLevel::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                        .commandBufferCount = 1,
                };
                VkCommandBuffer copy_command_buffer;
                if (allocate_command_buffers(device, &command_buffer_allocate_info, &copy_command_buffer) not_eq VK_SUCCESS) {
                        std::puts("unable to allocate command buffers");
                        std::exit(420);
                }

                auto const begin_info = VkCommandBufferBeginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

                command_begin(copy_command_buffer, &begin_info);

                auto region = VkBufferCopy{.size = size};
                command_coppy_buffer(copy_command_buffer, src_buffer, dst_buffer, 1, &region);

                command_end(copy_command_buffer);

                auto submit_info = VkSubmitInfo{
                        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1,
                        .pCommandBuffers = &copy_command_buffer,
                };

                queue_submit(graphics_queue, 1, &submit_info, nullptr);
                device_wait_idle(device);
        };

        auto vertices = std::vector<vertex_position>{
                vertex_position{1, 1, 0},
                vertex_position{-1, 1, 0},
                vertex_position{1, -1, 0},
                vertex_position{-1, -1, 0},
        };
        auto vertex_buffer_size = VkDeviceSize{vertices.size() * sizeof(vertex_position)};

        VkDeviceMemory host_vertex_memory;
        VkBuffer host_vertex_buffer;
        create_buffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, vertex_buffer_size, &host_vertex_buffer, &host_vertex_memory);
        void *pvertex_buffer_memory;
        map_memory(device, host_vertex_memory, 0, vertex_buffer_size, 0, &pvertex_buffer_memory);
        std::memcpy(pvertex_buffer_memory, vertices.data(), vertex_buffer_size);
        unmap_memory(device, host_vertex_memory);
        VkDeviceMemory device_vertex_memory;
        VkBuffer device_vertex_buffer;
        create_buffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer_size, &device_vertex_buffer, &device_vertex_memory);
        buffer_copy(host_vertex_buffer, device_vertex_buffer, vertex_buffer_size);

        auto indices = std::vector<uint32_t>{
                2, 3, 0, 0, 1, 2,
        };

        auto index_buffer_size = VkDeviceSize{indices.size() * sizeof(uint32_t)};

        VkDeviceMemory host_index_memory;
        VkBuffer host_index_buffer;
        create_buffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, index_buffer_size, &host_index_buffer, &host_index_memory);
        void *pindex_buffer_memory;
        map_memory(device, host_index_memory, 0, index_buffer_size, 0, &pindex_buffer_memory);
        std::memcpy(pindex_buffer_memory, indices.data(), index_buffer_size);
        unmap_memory(device, host_index_memory);
        VkDeviceMemory device_index_memory;
        VkBuffer device_index_buffer;
        create_buffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer_size, &device_index_buffer, &device_index_memory);
        buffer_copy(host_index_buffer, device_index_buffer, index_buffer_size);

        // Command buffers.
        auto const command_buffer_allocate_info = VkCommandBufferAllocateInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = command_pool,
                .level = VkCommandBufferLevel::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = static_cast<uint32_t>(frame_buffers.size()),
        };
        if (not allocate_command_buffers)
                exit(1);
        auto command_buffers = std::vector<VkCommandBuffer>(image_count);
        if (allocate_command_buffers(device, &command_buffer_allocate_info, command_buffers.data()) not_eq VK_SUCCESS) {
                std::puts("unable to allocate command buffers");
                std::exit(420);
        }
        auto const clear_values = std::array{
                VkClearValue{.color = VkClearColorValue{.float32 = {1, 0, 1, 0}}},
                VkClearValue{.depthStencil = VkClearDepthStencilValue{.depth = 1, .stencil = 0}},
        };
        auto const viewport_scissor = VkRect2D{
                .offset = {0, 0},
                .extent = image_extent,
        };

        auto const command_begin_render_pass = load_vulkan_function<PFN_vkCmdBeginRenderPass>(instance, "vkCmdBeginRenderPass");
        auto const command_end_render_pass = load_vulkan_function<PFN_vkCmdEndRenderPass>(instance, "vkCmdEndRenderPass");
        auto const command_set_scissor = load_vulkan_function<PFN_vkCmdSetScissor>(instance, "vkCmdSetScissor");
        auto const command_set_viewport = load_vulkan_function<PFN_vkCmdSetViewport>(instance, "vkCmdSetViewport");
        auto const command_bind_pipeline = load_vulkan_function<PFN_vkCmdBindPipeline>(instance, "vkCmdBindPipeline");
        auto const command_bind_vertex_buffer = load_vulkan_function<PFN_vkCmdBindVertexBuffers>(instance, "vkCmdBindVertexBuffers");
        auto const command_bind_index_buffer = load_vulkan_function<PFN_vkCmdBindIndexBuffer>(instance, "vkCmdBindIndexBuffer");
        auto const command_bind_descriptor_sets = load_vulkan_function<PFN_vkCmdBindDescriptorSets>(instance, "vkCmdBindDescriptorSets");
        auto const command_draw = load_vulkan_function<PFN_vkCmdDraw>(instance, "vkCmdDraw");
        auto const command_draw_indexed = load_vulkan_function<PFN_vkCmdDrawIndexed>(instance, "vkCmdDrawIndexed");
        for (auto i = 0; i < image_count; ++i) {
                auto const &command_buffer = command_buffers[i];
                auto const &frame_buffer = frame_buffers[i];

                auto const command_buffer_begin_info = VkCommandBufferBeginInfo{
                        .sType = VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                };

                command_begin(command_buffer, &command_buffer_begin_info);

                auto const render_pass_begin_info = VkRenderPassBeginInfo{
                        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                        .renderPass = render_pass,
                        .framebuffer = frame_buffer,
                        .renderArea = VkRect2D{.offset{0, 0}, .extent = image_extent},
                        .clearValueCount = clear_values.size(),
                        .pClearValues = clear_values.data(),
                };
                command_begin_render_pass(command_buffer, &render_pass_begin_info, VkSubpassContents::VK_SUBPASS_CONTENTS_INLINE);

                command_set_viewport(command_buffer, 0, 1, &viewport);
                command_set_scissor(command_buffer, 0, 1, &viewport_scissor);
                command_bind_pipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

                VkDeviceSize offsets = 0;
                command_bind_vertex_buffer(command_buffer, 0, 1, &device_vertex_buffer, &offsets);
                command_bind_index_buffer(command_buffer, device_index_buffer, 0, VkIndexType::VK_INDEX_TYPE_UINT32);

                // TODO: bind descriptor sets
                // command_bind_descriptor_sets(command_buffer, VkPipelineBindPoint::VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, descriptor_set_bindings[i], 0, nullptr);
                // command_draw(command_buffer, vertices.size(), 1, 0, 0);
                command_draw_indexed(command_buffer, indices.size(), 1, 0, 0, 0);
                command_end_render_pass(command_buffer);
                command_end(command_buffer);
        }

        auto max_frames_in_fleight = 2;
        auto current_frame = 0;
        auto image_available_semaphores = std::vector<VkSemaphore>(image_count);
        auto render_finished_semaphores = std::vector<VkSemaphore>(image_count);
        auto in_flieght_fences = std::vector<VkFence>(image_count);
        auto const create_semaphore = load_vulkan_function<PFN_vkCreateSemaphore>(instance, "vkCreateSemaphore");
        auto const create_fence = load_vulkan_function<PFN_vkCreateFence>(instance, "vkCreateFence");
        for (auto i = 0; i < max_frames_in_fleight; ++i) {
                auto empty_semaphore_create_info = VkSemaphoreCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                };
                if (create_semaphore(device, &empty_semaphore_create_info, allocator, &image_available_semaphores[i]) not_eq VK_SUCCESS)
                        std::exit(3939);
                if (create_semaphore(device, &empty_semaphore_create_info, allocator, &render_finished_semaphores[i]) not_eq VK_SUCCESS)
                        std::exit(3939);
                auto fence_create_info = VkFenceCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                        .flags = VkFenceCreateFlagBits::VK_FENCE_CREATE_SIGNALED_BIT,
                };
                if (create_fence(device, &fence_create_info, allocator, &in_flieght_fences[i]) not_eq VK_SUCCESS)
                        std::exit(765987);
        }

        VkQueue present_queue;
        std::puts(std::format(" present index = {}", present_index).c_str());
        get_queue(device, present_index, 0, &present_queue);

        auto const acquire_next_image = load_vulkan_function<PFN_vkAcquireNextImageKHR>(instance, "vkAcquireNextImageKHR");
        auto const queue_present = load_vulkan_function<PFN_vkQueuePresentKHR>(instance, "vkQueuePresentKHR");
        auto const queue_wait_idle = load_vulkan_function<PFN_vkQueueWaitIdle>(instance, "vkQueueWaitIdle");
        auto const wait_for_fences = load_vulkan_function<PFN_vkWaitForFences>(instance, "vkWaitForFences");
        auto const reset_fences = load_vulkan_function<PFN_vkResetFences>(instance, "vkResetFences");

        while (not glfwWindowShouldClose(window)) {
                glfwPollEvents();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));

                std::clog << "waiting for fence " << current_frame << std::endl;
                device_wait_idle(device);
                if (wait_for_fences(device, 1, &in_flieght_fences[current_frame], VK_TRUE, UINT64_MAX) not_eq VK_SUCCESS) {
                        std::puts("unable to wait for frame fence");
                }
                if (reset_fences(device, 1, &in_flieght_fences[current_frame]) not_eq VK_SUCCESS) {
                        std::puts("unable to reset fence");
                }

                // TODO update uniform buffer.

                uint32_t swapchain_image_index;
                if (acquire_next_image(device, swapchain, UINT64_MAX, image_available_semaphores[current_frame], in_flieght_fences[current_frame], &swapchain_image_index) not_eq VK_SUCCESS) {
                        std::puts("unable to aquire next swapchian image index.");
                }

                VkPipelineStageFlags wait_dst_stage_mask = VkPipelineStageFlagBits::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                auto const submit_info = VkSubmitInfo{
                        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .waitSemaphoreCount = 1,
                        .pWaitSemaphores = &image_available_semaphores[current_frame],
                        .pWaitDstStageMask = &wait_dst_stage_mask,
                        .commandBufferCount = 1,
                        .pCommandBuffers = &command_buffers[swapchain_image_index],
                        .signalSemaphoreCount = 1,
                        .pSignalSemaphores = &render_finished_semaphores[current_frame],
                };

                queue_submit(graphics_queue, 1, &submit_info, in_flieght_fences[current_frame]);

                auto const present_info = VkPresentInfoKHR{
                        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                        .pNext = nullptr,
                        .waitSemaphoreCount = 0,
                        .pWaitSemaphores = nullptr,
                        .swapchainCount = 1,
                        .pSwapchains = &swapchain,
                        .pImageIndices = &swapchain_image_index,
                };
                if (queue_present(present_queue, &present_info) not_eq VK_SUCCESS) {
                        std::puts("unable to present");
                }

                if (queue_wait_idle(present_queue) not_eq VK_SUCCESS) {
                        std::puts("could not wait for some reason");
                }

                current_frame = (current_frame + 1) % max_frames_in_fleight;
                while (std::getchar() not_eq '\n')
                        ;
        }

        // auto vk_destroy_instance = load_vulkan_function<PFN_vkDestroyInstance>(instance, "vkDestroyInstance");
        // vk_destroy_instance(instance, allocator);
}
