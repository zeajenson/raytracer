#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <ostream>
#include <random>
#include <thread>
#include <vector>
#define GLFW_INCLUDE_VULKAN
#include<GLFW/glfw3.h>


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
    constexpr auto to_unit() const noexcept -> vec3;
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

constexpr auto vec3::to_unit() const noexcept -> vec3 { return (*this) / this->length(); }

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
    vec3 on_unit_sphere = generate_random_vector_in_unit_sphere().to_unit();
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
    auto a = 0.5 * (ray.direction.to_unit().e[y] + 1.0);
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
            auto reflected = reflect(ray.direction.to_unit(), hit.normal);
            return Scatter_Values{
                .ray = Ray{hit.point, reflected + (metal_fuzz_radiai[material.properties_index] * generate_random_vector_in_unit_sphere().to_unit())},
                .attenuation_color = attenuation,
            };
        }
        case Ecs::Material::Type::dielectric: {
            double refraction_ratio = hit.front_face ? (1.0 / transmision_ir[material.properties_index]) : transmision_ir[material.properties_index];

            auto unit_direction = ray.direction.to_unit();
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

    struct Image{
        std::vector<uint8_t> r;
        std::vector<uint8_t> g;
        std::vector<uint8_t> b;
        uint64_t width,height;
    };

inline auto ray_trace_stuff(uint64_t width, uint64_t height) -> Image{

    const auto image_size = width * height;

    
    auto image = Image{
        .r = std::vector<uint8_t>(image_size),
        .g = std::vector<uint8_t>(image_size),
        .b = std::vector<uint8_t>(image_size),
        .width = width,
        .height = height
    };

    const auto aspect_ratio = 1;
    const auto image_width = 512*2;
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
        const auto w = (look_from - look_at).to_unit();
        const auto u = cross(vup, w).to_unit();
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

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::puts(std::format("validation layer :{} \n", pCallbackData->pMessage).c_str());

    return VK_FALSE;
}

template<typename T>
T load_vulkan_function(VkInstance instance, const char * name){
    return reinterpret_cast<T>(glfwGetInstanceProcAddress(instance, name));
}

// 🐈
//
int main() noexcept{
    if(not glfwInit()){
        std::puts("Could not initialize GLFW");
        std::terminate();
    }
    
    const auto width = 512, height = 512;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    const auto window = glfwCreateWindow(width,height,"raytracer",nullptr, nullptr);
    if(not window){
        std::puts("Could get GLFW window");
        std::terminate();
    }

    // auto pixels = ray_trace_stuff(width, height);

    if(glfwVulkanSupported() == GLFW_API_UNAVAILABLE){
        std::puts("Vulkan is not supported.\n");
    }

    {

        auto allocator = nullptr; 

        auto app_info = VkApplicationInfo{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "raytracer",
            .applicationVersion = VK_MAKE_VERSION(1,0,0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1,0,0),
            .apiVersion = VK_API_VERSION_1_3,
        };

        std::vector<char const*> extension_names = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME}; 
        std::vector<char const*> layer_names = {"VK_LAYER_KHRONOS_validation"}; 
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
        auto vk_create_instance = load_vulkan_function<PFN_vkCreateInstance>(nullptr, "vkCreateInstance");
        if(vk_create_instance(&instantce_info, allocator, &instance) not_eq VK_SUCCESS){
            std::puts("Unable to create vulkan instance.\n");
        }
        auto debug_messenger_info = VkDebugUtilsMessengerCreateInfoEXT {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = nullptr,
            .flags = {},
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT 
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT 
            | VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debug_callback,
        };

        VkDebugUtilsMessengerEXT debug_utils_messenger;

        auto vk_create_debug_utils_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(glfwGetInstanceProcAddress(instance, "vkCreateDebugUtilsMessengerEXT"));
        if(not vk_create_debug_utils_messenger){
            std::puts("unable to load debug utils messenger.\n");
        }else if(vk_create_debug_utils_messenger(instance, &debug_messenger_info, allocator, &debug_utils_messenger) not_eq VK_SUCCESS){
            std::puts("unable to create debug utils messenger.\n");
        }

        auto vk_destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(glfwGetInstanceProcAddress(instance, "vkDestroyInstance"));
        vk_destroy_instance(instance, allocator);
    }


    while (not glfwWindowShouldClose(window)){
        glfwPollEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
