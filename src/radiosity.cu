
#include "radiosity.hpp"
#include "scene.hpp"
#include "cutil_math.hpp"
#include <stdio.h>

#define IS_LIGHT(x)		(x < (2.0f / SPLIT_UNIT) * (2.0f / SPLIT_UNIT))

namespace radiosity {

#define PI 3.14159265358979f

bool visible(Scene* scene, size_t src, size_t dst) 
{
  float3 pos = scene->patches[src].corner_pos;
  float3 dir = scene->patches[dst].corner_pos -
               scene->patches[src].corner_pos;
  float hit_time = length(dir);
  dir = normalize(dir);
  float new_hit = scene->tree.intersect(pos, dir, EPSILON, INFTY, NULL);

  if (new_hit < hit_time) {
    //printf("occlusion\n");
    return false;
  }

  //printf("%f vs %f\n", hit_time, new_hit);
  return true;
}

void update_radiosity(Scene *scene, float3 *matrix, size_t dim)
{
  //Populate initial state
  float3 *energies = new float3[dim];
  float3 *sol_0    = new float3[dim];
  float3 *sol_1    = new float3[dim];
  for(size_t ii = 0; ii < dim; ii++)
  {
    energies[ii] = make_float3(scene->patches[ii].emission);
    sol_0[ii] = energies[ii];
    sol_1[ii] = energies[ii];
  }
  
  //Solve, then populate textures
  solve_radiosity(matrix, energies, sol_0, sol_1, dim);

  //Populate patches with solved colors
  for(size_t x = 0; x < dim; x++)
  {
    scene->patches[x].color = sol_1[x] * scene->patches[x].reflectivity;
  }

  delete energies;
  delete sol_0;
  delete sol_1;
}

void update_light(Scene* scene, float3* matrix, size_t dim)
{
  // Move light slowly:
  static float dir = 1.0f;
  for (size_t x = 0; x < dim; x++) {
	  if (!IS_LIGHT(x)) break;
    scene->patches[x].corner_pos.z -= dir * 0.1f;
  }
  
  if (fabs(scene->patches[0].corner_pos.z - 5.0f) > 5.0f) dir = -dir;

  // Populate energy-transfer matrix
  for (size_t x = 0; x < dim; x++) {
    if (!IS_LIGHT(x)) continue;

    for (size_t y = x+1; y < dim; y++) {
      bool v = visible(scene, x, y);
      if (!v) {
        matrix[y * dim + x] = matrix[x * dim + y] = make_float3(0.0f);
        continue;
      }

      float ff = form_factor(&scene->patches[y], &scene->patches[x]);
      matrix[y * dim + x] = -ff * scene->patches[y].reflectivity;
      matrix[x * dim + y] = -ff * scene->patches[x].reflectivity;

    }
  }

  update_radiosity(scene, matrix, dim);
}


bool calc_radiosity(Scene* scene, float3* matrix, size_t dim)
{
  //Populate energy-transfer matrix
  for (size_t x = 0; x < dim; x++) {
    if (x % 32 == 0) {
      printf("Starting row %d\n", x);
    }
    for (size_t y = 0; y < x; y++) {
      bool v = visible(scene, x, y);
      if (!v) {
        matrix[y * dim + x] = matrix[x * dim + y] = make_float3(0.0f);
        continue;
      }

      float ff = form_factor(&scene->patches[y], &scene->patches[x]);
      matrix[y * dim + x] = -ff * scene->patches[y].reflectivity;
      matrix[x * dim + y] = -ff * scene->patches[x].reflectivity;
    }
    matrix[x * dim + x] = make_float3(1.0f);
  }

  update_radiosity(scene, matrix, dim);
  return true;
}

//Calculate the form factor between two planes
float form_factor(Plane *p1, Plane *p2)
{
	float3 p1_norm = cross(p1->x_vec, p1->y_vec);
	float3 p2_norm = cross(p2->x_vec, p2->y_vec);
	float a1 = length(p1_norm);
	float a2 = length(p2_norm);

	float3 btwn = (p1->corner_pos + 0.5 * p1->x_vec + 0.5 * p1->y_vec) -
				  (p2->corner_pos + 0.5 * p2->x_vec + 0.5 * p2->y_vec);
	float  dist = length(btwn);

	btwn    = normalize(btwn);
	p1_norm = normalize(p1_norm);
	p2_norm = normalize(p2_norm);

	float dTheta = dot(btwn, p1_norm) * dot(btwn, p2_norm);
	// since we effectively divide by a1 at the end, only take on a2
  float dArea  = a2;
	float ff = dTheta * dArea / (dist * dist * PI);

  return (ff > 0) ? ff : -ff;
}

__host__ __device__
void jacobi(size_t ii, float3 *x_0, float3 *x_1, float3 *M, float3* b, size_t dim)
{
	float3 acc = make_float3(0.0f);

	for(size_t jj = 0; jj < dim; jj++)
	{
    if (ii == jj) continue;
		acc += M[ii*dim + jj] * x_0[jj];
	}

	x_1[ii] = b[ii] - acc;  // (b[ii]- acc) / M[ii*dim + ii];
}

__global__
void jacobi_GPU(float3 *x_0, float3 *x_1, float3 *M, float3 *b, size_t dim)
{
	size_t ii = blockIdx.x * blockDim.x + threadIdx.x;

	//Check index
	if(ii >= dim)
		return;

	jacobi(ii, x_0, x_1, M, b, dim);
}

void jacobi_CPU(float3 *x_0, float3 *x_1, float3 *M, float3 *b, size_t dim)
{
	for(size_t ii = 0; ii < dim; ii++)
	{
		jacobi(ii, x_0, x_1, M, b, dim);
	}
}

__host__
void solve_radiosity(float3 *M, float3 *b, float3 *sol_0, float3 *sol_1, size_t dim)
{
	size_t iters = 4;

	size_t threadsPerBlock = 256;
	size_t threads = dim;
	size_t blocks  = threads / threadsPerBlock;
	blocks += ((threads % threadsPerBlock) > 0) ? 1 : 0;

	//Copy data to GPU:
	float3 *Mg, *bg, *sol_0g, *sol_1g;
	cudaMalloc((void **) &Mg, dim * dim * sizeof(float3));
	cudaMalloc((void **) &bg, dim * sizeof(float3));
	cudaMalloc((void **) &sol_0g, dim * sizeof(float3));
	cudaMalloc((void **) &sol_1g, dim * sizeof(float3));

	cudaMemcpy(Mg, M, dim*dim * sizeof(float3),     cudaMemcpyHostToDevice);
	cudaMemcpy(bg, b, dim * sizeof(float3),         cudaMemcpyHostToDevice);
	cudaMemcpy(sol_0g, sol_0, dim * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(sol_1g, sol_1, dim * sizeof(float3), cudaMemcpyHostToDevice);

	for(size_t ii = 0; ii < iters; ii++)
	{
		//jacobi_CPU(sol_0, sol_1, M, b, dim);
		//jacobi_CPU(sol_1, sol_0, M, b, dim);
		jacobi_GPU<<<blocks, threadsPerBlock>>>(sol_0g, sol_1g, Mg, bg, dim);
		jacobi_GPU<<<blocks, threadsPerBlock>>>(sol_1g, sol_0g, Mg, bg, dim);
	}

	//Copy data back to CPU:
	cudaMemcpy(sol_1, sol_1g, dim * sizeof(float3), cudaMemcpyDeviceToHost);

	cudaFree(Mg);
	cudaFree(bg);
	cudaFree(sol_0g);
	cudaFree(sol_1g);
}

}
