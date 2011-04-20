
#include "radiosity.hpp"
#include "cutil_math.hpp"
#include <stdio.h>
#include <limits>

namespace radiosity {

#define PI 3.14159265358979f

float visible(Scene* scene, size_t src, size_t dst) 
{

  float3 pos = scene->patches[src].corner_pos;
  float3 dir = scene->patches[dst].corner_pos -
               scene->patches[src].corner_pos;
  float hit_time = length(dir);
  dir = normalize(dir);

  // Check versus everything else, see if something else gets hit first.
  for (size_t i = 0; i < scene->patches.size(); i++) {
    if (i == src || i == dst) continue;

    float3 p0 = scene->patches[i].corner_pos;
    float3 p1 = p0 + scene->patches[i].x_vec;
    float3 p2 = p0 + scene->patches[i].y_vec;
    
    // column vectors
    float3 c1 = -dir;
    float3 c2 = p1 - p0;
    float3 c3 = p2 - p0;
    float det = dot(c1, cross(c2, c3));
    
    // inverted row vectors
    float3 r1 = cross(c2, c3) / det;
    float3 r2 = cross(c3, c1) / det;
    float3 r3 = cross(c1, c2) / det;
    
    // coefficients
    float t_hit = dot(r1, pos - p0);
    float u_hit = dot(r2, pos - p0);
    float v_hit = dot(r3, pos - p0);
    
    // We've hit something else first, abort!
    if (t_hit > 0 && t_hit < hit_time &&
        u_hit >= scene->patches[i].x_min && u_hit <= scene->patches[i].x_max &&
        v_hit >= scene->patches[i].y_min && v_hit <= scene->patches[i].y_max)
    {
      return false;
    }
  }

  return true;
}


bool initialize_radiosity(Scene* scene, float* matrix, size_t* matrix_dim)
{
  size_t dim = scene->patches.size();
  *matrix_dim = dim;
  
  matrix = new float[dim * dim];

  return calc_radiosity(scene, matrix, dim);
}

bool calc_radiosity(Scene* scene, float* matrix, size_t dim)
{
  for (size_t x = 0; x < dim; x++) {
    if (x % 32 == 0) {
      printf("Starting row %d\n", x);
    }
    for (size_t y = 0; y < x; y++) {
      bool v = visible(scene, x, y);
      if (!v) continue;

      float ff = form_factor(&scene->patches[y], &scene->patches[x]);
      matrix[y * dim + x] = -ff * scene->patches[x].reflectance;
      matrix[x * dim + y] = -ff * scene->patches[y].reflectance;
    }
    matrix[x * dim + x] = 1.0f;
  }
  return true;
}

//Calculate the form factor between two planes
float form_factor(Plane *p1, Plane *p2)
{
	float3 p1_norm = cross(p1->x_vec, p1->y_vec);
	float3 p2_norm = cross(p2->x_vec, p2->y_vec);
	float a1 = length(p1_norm);
	float a2 = length(p2_norm);

	float3 btwn = p1->corner_pos - p2->corner_pos;
	float  dist = length(btwn);

	btwn    = normalize(btwn);
	p1_norm = normalize(p1_norm);
	p2_norm = normalize(p2_norm);

	float dTheta = dot(btwn, p1_norm) * dot(btwn, p2_norm);
	// since we effectively divide by a1 at the end, only take on a2
  float dArea  = a2;
	float ff = dTheta * dArea / (dist * dist * PI);

	return ff;
}

}
