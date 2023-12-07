#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

enum { x, y, z };

auto dev = std::random_device();
std::mt19937 generator(dev());
inline double random_double(double min = 0,
                            double max = RAND_MAX + 1.0) noexcept {
  std::uniform_real_distribution<double> dist(min, max);
  return dist(generator);
}

struct vec3 {
  double e[3];
  constexpr auto operator[](std::size_t index) const noexcept {
    return e[index];
  }
  constexpr auto operator-() const noexcept {
    return vec3{-e[0], -e[1], -e[2]};
  }
  constexpr auto operator+=(vec3 const &v) noexcept {
    e[x] += v[x];
    e[y] += v[y];
    e[z] += v[z];
  }
  constexpr auto operator*=(vec3 const &v) noexcept {
    e[x] *= v[x];
    e[y] *= v[y];
    e[z] *= v[z];
  }
  constexpr auto operator*=(double const &t) noexcept {
    e[x] *= t;
    e[y] *= t;
    e[z] *= t;
  }
  constexpr auto length() const noexcept -> double;
  constexpr auto to_unit() const noexcept -> vec3;
  static inline auto random(double min = 0, double max = RAND_MAX + 1.0) {
    return vec3{random_double(min, max), random_double(min, max),
                random_double(min, max)};
  }
  constexpr auto near_zero()const noexcept{
    constexpr auto s = 1e-8;
    return (std::fabs(e[x]) < s && std::fabs(e[y]) < s && std::fabs(e[z]) < s);
  }
};

struct LAB {
  float l, a, b;
};
struct rgba {
  std::uint8_t r, g, b, a;
};

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

constexpr double degreees_to_radians(double degrees) noexcept {
  return degrees * pi / 180.0;
}

constexpr auto operator*(vec3 const &u, vec3 const &v) noexcept {
  return vec3{{
      v.e[0] * u.e[0],
      v.e[1] * u.e[1],
      v.e[2] * u.e[2],
  }};
}

constexpr auto operator*(vec3 const &v, double t) noexcept {
  return vec3{{
      v.e[0] * t,
      v.e[1] * t,
      v.e[2] * t,
  }};
}

constexpr auto operator*(double t, vec3 const &v) noexcept { return v * t; }

constexpr auto operator/(vec3 v, double t) noexcept { return v * (1 / t); }

constexpr auto vec3::to_unit() const noexcept -> vec3 {
  return (*this) / this->length();
}

constexpr auto operator-(vec3 const &u, vec3 const &v) noexcept {
  return vec3{u[0] - v[0], u[1] - v[1], u[2] - v[2]};
}

constexpr auto operator-(vec3 const &u, double t) noexcept {
  return vec3{u[0] - t, u[1] - t, u[2] - t};
}

constexpr vec3 operator+(const vec3 &u, const vec3 &v) noexcept {
  return vec3{u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]};
}

constexpr auto dot(vec3 const &a, vec3 const &b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
constexpr auto vec3::length() const noexcept -> double {
  return std::sqrt(dot(*this, *this));
}

constexpr auto reflect(vec3 const & v, vec3 const & n) noexcept{
  return v - 2*dot(v,n)*n;
}

inline auto generate_random_vector_in_unit_sphere() noexcept {
  for (;;) {
    auto p = vec3::random(-1, 1);
    if (dot(p, p) < 1) {
      return p;
    }
  }
}

inline auto
generate_random_unit_vector_on_hemisphere(vec3 const &normal) noexcept {
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

inline auto calculate_void_color(Ray const &ray) {
  auto a = 0.5 * (ray.direction.to_unit().e[y] + 1.0);
  auto color = ((1.0 - a) * vec3{1, 1, 1} + a * vec3{0.5, 0.7, 1.0});
  // auto [r,g,b] = generate_random_vector_in_unit_sphere().e;
  // auto color = vec3{std::clamp(r, 0.03,1.0), std::clamp(g, 0.01, 1.0), std::clamp(b, 0.02, 1.0)};
  return color;
}

struct Hit_Record {
  vec3 point;
  vec3 normal;
  double t;
  size_t entity_index;
  bool front_face;
};

constexpr auto calculate_face_normal(Ray const &ray,
                                     vec3 const &outward_normal) noexcept {
  auto front_face = dot(ray.direction, outward_normal) < 0;
  auto normal = front_face ? outward_normal : -outward_normal;
  struct {
    vec3 normal;
    bool front_face;
  } face_normal{normal, front_face};
  return face_normal;
};

int main() {
  // Image
  auto aspect_ratio = 16.0 / 9.0;
  auto image_width = 700;
  auto image_height = static_cast<int>(image_width / aspect_ratio);
  if (image_height < 1)
    image_height = 1;
  auto samples = 20;
  auto bounces = 5;

  // Camera
  auto focal_length = 1.0;
  auto viewport_height = 2.0;
  auto viewport_width =
      viewport_height * (static_cast<double>(image_width) / image_height);
  auto camera_center = vec3{0, 0, 0};

  // Calculate the vectors across the horizontal and down the vertical viewport
  // edges.
  auto viewport_u = vec3{viewport_width, 0, 0};
  auto viewport_v = vec3{0, -viewport_height, 0};

  // Calculate the horizontal and vertical delta vectors from pixel to pixel.
  auto pixel_delta_u = viewport_u / image_width;
  auto pixel_delta_v = viewport_v / image_height;

  // Calculate the location of the upper left pixel.
  auto viewport_upper_left = camera_center - vec3{0, 0, focal_length} -
                             viewport_u / 2 - viewport_v / 2;
  auto first_pixel_location =
      viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

  // Settupe object arrays

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
      vec3 albedo;
      enum class Type {
        lambert,
        metal,
      } type;
    };

    std::vector<Material> materials = {{{1,0,1}, Material::Type::lambert}};

    constexpr auto apply_material(std::size_t material_index,
                                  std::size_t index) noexcept {
      material_indices[index] = material_index;
    }

    constexpr auto apply_material(Material material,
                                  std::size_t index) noexcept {
      material_indices[index] = materials.size();
      materials.push_back(material);
    }

    struct Scatter_Values{
      Ray ray;
      vec3 attenuation_color;
    };

    constexpr auto scatter(Ray const &ray,
                           Hit_Record const &hit) const noexcept -> std::optional<Scatter_Values> {
      auto material = materials[material_indices[hit.entity_index]];
      auto attenuation = material.albedo;
      switch (material.type) {
      case Ecs::Material::Type::lambert: {
        auto direction = hit.normal + generate_random_vector_in_unit_sphere();
        auto new_ray = Ray(hit.point, direction);
          if(direction.near_zero()){
            direction = hit.normal;
          }

        return Scatter_Values{
            new_ray,
            attenuation
          };
      }
      case Ecs::Material::Type::metal: {
          auto reflected = reflect(ray.direction.to_unit(), hit.normal);
          return Scatter_Values{
            .ray = Ray{hit.point, reflected},
            .attenuation_color = attenuation,
          };
      }
      }
      return std::nullopt;
    }

    constexpr auto hit(std::size_t index, Ray const &ray, double ray_tmin,
                       double ray_tmax) const noexcept
        -> std::optional<Hit_Record> {
      switch (entities[index]) {
      case Type::sphere: {
        auto [center, radius] = spheres[geometry_indices[index]];
        // auto sphere_hit = hit_sphere(center, radius, ray);
        auto sphere_origin = ray.origin - center;
        auto a = dot(ray.direction, ray.direction);
        auto half_b = dot(sphere_origin, ray.direction);
        auto c = dot(sphere_origin, sphere_origin) - (radius * radius);
        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0) {
          return std::nullopt;
        }
        auto discriminant_square_root = std::sqrt(discriminant);

        // auto sphere_hit = (-half_b - std::sqrt(discriminant)) / a;
        auto root = (-half_b - discriminant_square_root) / a;
        if (root <= ray_tmin or root >= ray_tmax) {
          root = (-half_b + discriminant_square_root) / a;
          if (root <= ray_tmin or root >= ray_tmax) {
            return std::nullopt;
          }
        }

        // if (sphere_hit > 0) {
        //   auto normal = (ray.at(sphere_hit) - vec3{0, 0, -1}).to_unit();
        //   return Hit_Record{sphere_origin, normal, sphere_hit};
        // }
        // return vec3{1,0,0};
        // return 0.5 * vec3{normal[0] + 1, normal[1] + 1, normal[2] + 1};
        auto point = ray.at(root);
        auto outward_normal = (point - center) / radius;
        auto [normal, is_front_face] =
            calculate_face_normal(ray, outward_normal);

        return Hit_Record{.point = point,
                          .normal = normal,
                          .t = root,
                          .entity_index = index,
                          .front_face = is_front_face};
      }
      default:
        return std::nullopt;
      }
    }
    constexpr auto calculate_hit(Ray const &ray, double ray_tmin,
                                 double ray_tmax) const noexcept
        -> std::optional<Hit_Record> {
      // TODO: find the nearest object hit.
      auto hit_record = std::optional<Hit_Record>(std::nullopt);
      for (auto i = 0; i < entities.size(); ++i) {
        auto current_hit = hit(i, ray, ray_tmin, ray_tmax);
        if(current_hit){
          if(not hit_record){
            hit_record = current_hit;
          }else if (current_hit->t < hit_record->t){
            hit_record = current_hit;
          }
        }
      }
      return hit_record;
    };
  } world;

  // main sphere
  auto middleboi = world.add_sphere({{0, 0, -1}, 0.5});
  auto bigboi = world.add_sphere({{0, -100.5, -1}, 100});
  auto metal1 = world.add_sphere({{-1, 0, -1}, 0.75});
  auto metal2 = world.add_sphere({{1, 0, -1}, 0.25});
  auto metal3 = world.add_sphere({{1, 0.5, -1}, 0.25});
  world.materials.front().albedo = {1,0.01,0.01};
  world.apply_material({{0.01,0.1,0.01}, Ecs::Material::Type::lambert}, bigboi);
  world.apply_material({{.5,.5,0},Ecs::Material::Type::metal}, metal1);
  world.apply_material({{0,0.25,0.25},Ecs::Material::Type::metal}, metal2);
  world.apply_material({{0,1,0},Ecs::Material::Type::metal}, metal3);
  auto spheres = 10;
  auto sphere_size = 0.1;
  auto material_index = 0;
  auto materials_material_index = 0;
  constexpr auto materials_count = 4;
  std::size_t materilas[materials_count] = {0,2,4,1}; 
  auto start = -((spheres * sphere_size * 1.5)/2);
  for(int i = spheres; i ; --i){
    auto offset = start + (sphere_size * 1.5 * i); 
    world.apply_material(materilas[materials_material_index], world.add_sphere({{offset, -0.5, -0.5}, sphere_size}));
    if(materials_material_index == materials_count -1) materials_material_index = 0; 
    else ++materials_material_index;
  }

  auto calculate_hit_color = [](Hit_Record const &hit) noexcept {
    auto normal = hit.normal;
    auto color = 0.5 * vec3{normal[0] + 1, normal[1] + 1, normal[2] + 1};
    return color;
  };

  auto convert_color = [](vec3 color, double samples_per_pixel) noexcept{
    auto scale = 1.0 / samples_per_pixel;
    auto [r, g, b] = color.e;
    r *= scale;
    g *= scale;
    b *= scale;
    // Gamma correct
    r = std::sqrt(r);
    g = std::sqrt(g);
    b = std::sqrt(b);

    auto ir = static_cast<std::uint8_t>(255 * std::clamp(r, 0.0, 1.0));
    auto ig = static_cast<std::uint8_t>(255 * std::clamp(g, 0.0, 1.0));
    auto ib = static_cast<std::uint8_t>(255 * std::clamp(b, 0.0, 1.0));
    return rgba{ir,ig,ib,0};
  };

  auto write_color = [](vec3 color, double samples_per_pixel) noexcept {
    auto scale = 1.0 / samples_per_pixel;
    auto [r, g, b] = color.e;
    r *= scale;
    g *= scale;
    b *= scale;
    // Gamma correct
    r = std::sqrt(r);
    g = std::sqrt(g);
    b = std::sqrt(b);

    auto ir = static_cast<std::uint8_t>(255 * std::clamp(r, 0.0, 1.0));
    auto ig = static_cast<std::uint8_t>(255 * std::clamp(g, 0.0, 1.0));
    auto ib = static_cast<std::uint8_t>(255 * std::clamp(b, 0.0, 1.0));
    write_rgba({ir, ig, ib, 0});
  };

  std::vector<rgba>pixels(image_width*image_height);
  std::vector<vec3>samples_per_pixel(pixels.size() * samples, {1,1,1});

  for (auto y = 0; y < image_height; ++y) {
    std::clog << "\rScanlines remaining: " << (image_height - y) << ' '
              << std::flush;
    for (auto x = 0; x < image_width; ++x) {

      auto pixel_center =
          first_pixel_location + (x * pixel_delta_u) + (y * pixel_delta_v);
      auto ray_center = camera_center;
      auto pixel_color = vec3{0, 0, 0};

      constexpr auto sample_square_size = .75;

      for (auto sample = 0; sample <= samples; ++sample) {
        auto pixel_sample_x =
            -(sample_square_size / 2.0) + random_double(0, sample_square_size);
        auto pixel_sample_y =
            -(sample_square_size / 2.0) + random_double(0, sample_square_size);
        auto pixel_sample_offset =
            (pixel_sample_x * pixel_delta_u) + (pixel_sample_y * pixel_delta_v);
        auto pixel_sample_center = pixel_center + pixel_sample_offset;
        auto ray_direction = pixel_sample_center - camera_center;
        auto has_ray = true;
        auto is_void = false;
        auto & color = samples_per_pixel[sample + (samples * (x + (y * image_width)))];
        // auto color = vec3{1,1,1};
        auto current_bounce_limit = bounces;
        auto ray = Ray{ray_center, ray_direction};
        while (has_ray) {
          if (not --current_bounce_limit) {
            color *= vec3{.001, .001, .001};
            break;
          }

          auto hit = world.calculate_hit(ray, 0.001, infinity);

          if (hit) {
            auto ray_center = hit->point;
            auto ray_direction =
                hit->normal +
                generate_random_unit_vector_on_hemisphere(hit->normal);
            auto scatter = world.scatter(ray, hit.value());
            if(scatter){
              ray = scatter->ray;
              color *= scatter->attenuation_color;
            }
            else{
              color *= {1,0,1};
            }
            //has_ray = false;
            // ray = Ray{ray_center, ray_direction};
            // color *= 0.5 * (calculate_hit_color(hit.value()).to_unit());
          } else {
            color *= calculate_void_color(ray);
            has_ray = false;
            //undefined behavior in the code is causing this to not work.
            // is_void = true;
            // if(current_bounce_limit == bounces -1){
            //   is_void = true;
            //   color *= vec3{1,0,1}; //calculate_void_color(ray);
            //   break;
            // }
          }
        }
        //Skip over sampling the void
        if(is_void){
          sample = std::midpoint(sample, samples);
        }

        // pixel_color += color;
      }

      // pixels[x + (y*image_width)] = convert_color(pixel_color, samples);
      // write_color(pixel_color, samples);
    }
  }

  for (auto y = 0; y < image_height; ++y) {
    for (auto x = 0; x < image_width; ++x) {
      vec3 color = {0,0,0};
      for (auto sample = 0; sample <= samples; ++sample) {
        color += samples_per_pixel[sample + (samples * (x + (y * image_width)))];
      }
      pixels[x + (y * image_width)] = convert_color(color, samples);
    }
  }

  std::clog << "Writing...";
  std::puts(std::format("P3\n{} {}\n255\n", image_width, image_height).c_str());
  for (auto y = 0; y < image_height; ++y) {
    std::clog << "\rScanlines remaining: " << (image_height - y) << ' ' << std::flush;
    for (auto x = 0; x < image_width; ++x) {
      write_rgba(pixels[x + (y*image_width)]);
    }
  }
}
