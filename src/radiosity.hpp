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
// ranges dictating a rectangle (if any ranges exist).
struct Plane
{
    float3 corner_pos;
    float3 x_vec;
    float3 y_vec;
    float x_min;
    float x_max;
    float y_min;
    float y_max;
};

// A light is determined by a position and maybe a color? xyz -> rgb.
struct Light
{
    float3 pos;
    float3 color;
};

// Contains all state, perhaps also the data we calculate.
struct Scene
{
    std::vector<Plane> objs;
    std::vector<Light> lights;
};

bool initialize_radiosity(Scene* scene);
void render_image(uint8_t* color, size_t width, size_t height,
                  const Scene* scene, const Camera* camera);

}

