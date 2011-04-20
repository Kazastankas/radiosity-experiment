#ifndef RADIOSITY_H
#define RADIOSITY_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <vector>

namespace radiosity {

struct Camera
{
    float3 pos;
    float3 dir;
    float3 up;
    // vertical field of view, in radians.
    float fov;
    // width / height
    float aspect_ratio;
};

// A plane is determined by a corner of origin, two vectors, and
// ranges dictating a rectangle (if any ranges exist). Each plane has a color.
// They now also have reflectance and natural emission. Lights are just planes
// with emissive properties.
struct Plane
{
    float3 corner_pos;
    float3 x_vec;
    float3 y_vec;
    float3 color;
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float emission;
    float reflectance;
    float energy;
};

// Contains all state, perhaps also the data we calculate.
struct Scene
{
    std::vector<Plane> planes;
    std::vector<Plane> patches;
};

bool initialize_radiosity(Scene* scene);
void render_image(uint8_t* color, size_t width, size_t height,
                  const Scene* scene, const Camera* camera);
void trace_ray(uint8_t* color, size_t x, size_t y, float3 pos, float3 dir,
               const Scene* scene, size_t height);
}

#endif
