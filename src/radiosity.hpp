#ifndef RADIOSITY_H
#define RADIOSITY_H

#include "kdtree.hpp"

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
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float3 color;
    float emission;
    float energy;

    Box bound;
};

// Contains all state, perhaps also the data we calculate.
struct Scene
{
    std::vector<Plane> patches;
    KDTree tree;
};

bool calc_radiosity(Scene* scene, float3* matrix, size_t dim);
float form_factor(Plane *p1, Plane *p2);
void solve_radiosity(float3 *M, float3 *b, float3 *sol_0, float3 *sol_1, size_t dim);
}

#endif
