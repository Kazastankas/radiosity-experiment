
#include "radiosity.hpp"
#include "cutil_math.hpp"
#include <stdio.h>

namespace radiosity {

#define PI 3.14159265358979f

__host__ __device__
bool visible(Plane* patches, size_t src, size_t dst, size_t dim) 
{

  float3 pos = patches[src].corner_pos;
  float3 dir = patches[dst].corner_pos -
               patches[src].corner_pos;
  float hit_time = length(dir);
  dir = normalize(dir);

  // Check versus everything else, see if something else gets hit first.
  for (size_t i = 0; i < dim; i++) {
    if (i == src || i == dst) continue;

    float3 p0 = patches[i].corner_pos;
    float3 p1 = p0 + patches[i].x_vec;
    float3 p2 = p0 + patches[i].y_vec;
    
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
        u_hit >= patches[i].x_min && u_hit <= patches[i].x_max &&
        v_hit >= patches[i].y_min && v_hit <= patches[i].y_max)
    {
      return false;
    }
  }

  return true;
}

__global__
void build_matrix(Plane *patches, float3 *M, size_t dim)
{
	size_t xx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yy = blockIdx.y * blockDim.y + threadIdx.y;

	if(xx >= dim || yy >= dim)
		return;

	if(xx == yy)
	{
    	M[xx * dim + xx] = make_float3(1.0f);
	}
	else
	{
      bool v   = visible(patches, xx, yy, dim);
      float ff = form_factor(&patches[yy], &patches[xx]);

      M[yy * dim + xx] = v * -ff * patches[yy].color;
      M[xx * dim + yy] = v * -ff * patches[xx].color;
	}
}

bool calc_radiosity(Scene* scene, float3* matrix, size_t dim)
{
  float3 *Mg, *bg, *sol_0g, *sol_1g;
	cudaMalloc((void **) &Mg, dim * dim * sizeof(float3));
	cudaMalloc((void **) &bg, dim * sizeof(float3));
	cudaMalloc((void **) &sol_0g, dim * sizeof(float3));
	cudaMalloc((void **) &sol_1g, dim * sizeof(float3));

  //Copy scene data
  Plane *patches, *temp;
	cudaMalloc((void **) &patches, dim * sizeof(Plane));
  temp = new Plane[dim];
  for(size_t ii = 0; ii < dim; ii++)
  {
	temp[ii] = scene->patches[ii];
  }
  cudaMemcpy(patches, temp, dim * sizeof(Plane), cudaMemcpyHostToDevice);

  //Populate energy-transfer matrix
	size_t threadsPerBlock = 256;
	size_t threads = dim;
	size_t blocks  = threads / threadsPerBlock;
	blocks += ((threads % threadsPerBlock) > 0) ? 1 : 0;

  cudaMemcpy(Mg, matrix, dim*dim * sizeof(float3),     cudaMemcpyHostToDevice);
  build_matrix<<<blocks, threads>>>(patches, Mg, dim);


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

	cudaMemcpy(bg,     energies, dim * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(sol_0g, sol_0,    dim * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(sol_1g, sol_1,    dim * sizeof(float3), cudaMemcpyHostToDevice);
  
  //Solve, then populate textures
  solve_radiosity(Mg, bg, sol_0g, sol_1g, dim);
  cudaMemcpy(sol_1g, sol_1, dim * sizeof(float3), cudaMemcpyDeviceToHost);
  for(size_t x = 0; x < dim; x++)
  {
    scene->patches[x].color = sol_1[x] * scene->patches[x].color;
  }

  return true;
}

//Calculate the form factor between two planes
__host__ __device__
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
	size_t iters = 100;

	size_t threadsPerBlock = 256;
	size_t threads = dim;
	size_t blocks  = threads / threadsPerBlock;
	blocks += ((threads % threadsPerBlock) > 0) ? 1 : 0;

	for(size_t ii = 0; ii < iters; ii++)
	{
		//jacobi_CPU(sol_0, sol_1, M, b, dim);
		//jacobi_CPU(sol_1, sol_0, M, b, dim);
		jacobi_GPU<<<blocks, threadsPerBlock>>>(sol_0, sol_1, M, b, dim);
		jacobi_GPU<<<blocks, threadsPerBlock>>>(sol_1, sol_0, M, b, dim);
	}
}

}
