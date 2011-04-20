
#include "radiosity.hpp"
#include "cutil_math.hpp"
#include <stdio.h>
#include <limits>

namespace radiosity {

#define PI 3.14159265358979f


bool initialize_radiosity(Scene* scene)
{
  return true;
  // TODO implement
};

void render_image(uint8_t* color, size_t width, size_t height,
                  const Scene* scene, const Camera* camera)
{
  float3 v, w, u;
  float l, r, b, t, n;
  float fovrad = camera->fov;
  
  v = camera->up;
  w = -camera->dir;
  u = cross(v, w);
  
  n = 1;
  t = tan(fovrad / 2.0f) * n;
  b = -t;
  l = b * camera->aspect_ratio;
  r = -l;
 
  for (size_t i = 0; i < width; i++) {
    for (size_t j = 0; j < height; j++) {
      float u_s = l + (r - l) * i / width;
      float v_s = b + (t - b) * j / height;
      float w_s = -n;
      
      float3 vray = normalize(u_s * u + v_s * v + w_s * w);
      trace_ray(color, i, j, camera->pos, vray, scene, height);
    }
  }
}

void trace_ray(uint8_t* color, size_t x, size_t y, float3 pos, float3 dir,
               const Scene* scene, size_t height)
{
  size_t idx = 4 * (y * height + x);
  float hit_time = std::numeric_limits<float>::infinity();
  
  // reset
  color[idx] = color[idx + 1] = color[idx + 2] = color[idx + 3] = 0;

  // Find the closest thing hit.
  for (size_t i = 0; i < scene->patches.size(); i++) {
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
    
    if (t_hit > 0 && t_hit < hit_time &&
        u_hit >= scene->patches[i].x_min && u_hit <= scene->patches[i].x_max &&
        v_hit >= scene->patches[i].y_min && v_hit <= scene->patches[i].y_max)
    {
      hit_time = t_hit;
      color[idx] = scene->patches[i].color.x * 255;
      color[idx + 1] = scene->patches[i].color.y * 255;
      color[idx + 2] = scene->patches[i].color.z * 255;
      color[idx + 3] = 255;
    }
  }
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
	float dArea  = a1*a2;
	float ff = dTheta * dArea / (dist * dist * PI);

	return ff;
}

__host__ __device__
void jacobi(size_t ii, float *x_0, float *x_1, float *M, float* b)
{
	float acc = 0;
	size_t dim = 10; //TODO: matrix dimensions

	for(size_t jj = 0; jj < dim; jj++)
	{
		acc += M[ii*dim + jj] * x_0[jj];
	}

	x_1[ii] = (b[ii] - acc) / M[ii*dim + ii];
}

__global__
void jacobi_GPU(float *x_0, float *x_1, float *M, float *b)
{
	size_t ii = blockIdx.x * blockDim.x + threadIdx.x;

	//Check index
	size_t dim = 10;
	if(ii >= dim)
		return;

	jacobi(ii, x_0, x_1, M, b);
}

void jacobi_CPU(float *x_0, float *x_1, float *M, float *b)
{
	size_t dim = 10; //TODO: matrix dimensions
	for(size_t ii = 0; ii < dim; ii++)
	{
		jacobi(ii, x_0, x_1, M, b);
	}
}

__host__
void solveRadiosity()
{
	size_t iters = 5;
	float *M;
	float *x_0;
	float *x_1;
	float *b;

	for(size_t ii = 0; ii < iters; ii++)
	{
		jacobi_CPU(x_0, x_1, M, b);
		jacobi_CPU(x_1, x_0, M, b);
		//jacobi_GPU<<<>>>(x_0, x_1, M, b);
		//jacobi_GPU<<<>>>(x_1, x_0, M, b);
	}
}

}
