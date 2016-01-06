/*   kernel.cu
 *	Author: Jitesh
 *	Created on 25/July/2015
 * 	Edited by Tapan Goel on 29th December 2015
 * 	
 */

#include <cuda_runtime.h>
#include <curand.h>
#include "Swarm.h"

inline __device__ __host__ float doPeriodic(float x, float L){		//clamps value of x within period [0,L]
     x = fmodf(x, L);		//fmodf returns floating point remainder of x/L.
     x += L;							// this second fmodf is required because some implementations..
     x = fmodf(x, L);	// ..of fmodf use negative remainders in principle range
     return x; 
}

inline __device__ __host__ float distCorr(float D, float world)
     /*if (D > (world / 2.0)){
      *		D = world - D;
     }
     else if (D < -(world / 2.0)){
	  D = world + D;
     }
     return D;*/
     D = abs(D);
     D = min(D, world - D);
     return D;
}

inline __device__ __host__ float calcDist(float2 a, float2 b, float L){   //to calculate distance between vectors a and b.
     float dx, dy, dist;
     dx = a.x - b.x;
     dy = a.y - b.y;
     dx = distCorr(dx, L);
     dy = distCorr(dy, L);
     dist = powf((powf(dx, 2) 
     powf(dy, 2)), 0.5);
     return dist;
}

__global__ void alignmentKernel (Particle *d_particles, int nParticles, float L, int pid, float2 * d_sumdir, float * d_c, float *d_dist){
     
     int cid = threadIdx.x + blockIdx.x * blockDim.x;	//inspecting each particle
     if (cid >= nParticles)
	  return;
     //cid = max(cid, 0);			//checking idx to be between 0 and maximum number of particles
     //cid = min(cid, nParticles - 1); 		
     //distance
     d_dist[cid] = calcDist(d_particles[pid].coord, d_particles[cid].coord, L);			
     //alignment
     if (d_dist[cid] <= Particle::Rs/* && cid != pid*/){
	  d_sumdir[pid] += d_particles[cid].dir;
	  d_c[pid] = d_c[pid] + 1.0;
     }
     //__syncthreads();
}

__device__ __host__ void alignmentFunction (Particle *d_particles, int nParticles, float L, int pid, float2 * d_sumdir, float * d_c, float *d_dist){
     for (int cid = 0; cid < nParticles; cid++){
	  //distance
	  d_dist[cid] = calcDist(d_particles[pid].coord, d_particles[cid].coord, L);			
	  //calculate total number of particles in the vicinity and sum their directions
	  if (d_dist[cid] <= Particle::Rs/* && cid != pid*/){
	       d_sumdir[pid].x += d_particles[cid].dir.x;
	       d_sumdir[pid].y += d_particles[cid].dir.y;
	       d_c[pid] = d_c[pid] + 1.0;
	  }
     }
}

__global__ void updateKernel (Particle *d_particles, int nParticles, float L, float2 *d_sumdir, float *d_c, float *d_dist, float* randArray){
     float w = 1.0;
     
     int idx = threadIdx.x + blockIdx.x * blockDim.x;
     idx = max(idx, 0);			//checking idx to be between 0 and maximum number of particles
     idx = min(idx, nParticles);
     d_sumdir[idx].x = 0.0; d_sumdir[idx].y = 0.0;
     d_c[idx] = 0.0;
     //call alignment kernel/function to calculate average direction of particles in vicinity
     //alignmentKernel <<<(nParticles / 256) + 1, 256 >>>(d_particles, nParticles, L, idx, d_sumdir, d_c, d_dist);
     alignmentFunction (d_particles, nParticles, L, idx, d_sumdir, d_c, d_dist);
     __syncthreads();
     //alignment (update direction with respect to average direction of particles in vicinity)
     d_particles[idx].dir.x = (w * d_particles[idx].dir.x + d_sumdir[idx].x) / (d_c[idx] + w);
     d_particles[idx].dir.y = (w * d_particles[idx].dir.y + d_sumdir[idx].y) / (d_c[idx] + w);
     //calculate theta
     d_particles[idx].theta = atan2(d_particles[idx].dir.y, d_particles[idx].dir.x);
     //Adding noise to theta
     d_particles[idx].theta = d_particles[idx].theta + (d_particles[idx].eta * ((randArray[idx] * 2.0) - 1) / 2.0);
     //calculate directions from theta
     d_particles[idx].dir.x = cos(d_particles[idx].theta);
     d_particles[idx].dir.y = sin(d_particles[idx].theta);
     //updating velocity of particles
     d_particles[idx].vel = d_particles[idx].dir * Particle::speed;
     __syncthreads();
     //updating coordinates of particles
     d_particles[idx].coord.x += d_particles[idx].vel.x;
     d_particles[idx].coord.y += d_particles[idx].vel.y;
     __syncthreads();
     //implementing periodic boundary
     d_particles[idx].coord.x = doPeriodic(d_particles[idx].coord.x, L);		
     d_particles[idx].coord.y = doPeriodic(d_particles[idx].coord.y, L);		
}





//-----------------------------------------------------------------------------------------------//
//----------------------------Random number generation protocols---------------------------------//
//-----------------------------------------------------------------------------------------------//

/*
 * launchRandInit() is called in main function and it calls the init_stuff function. 
 * It calls the init_stuff CUDA kernel which actually does the initialization of seeds d_state -
 * the inbuilt CUDA system for random number generation.
 * 
 * The make_rand function is called by the update the randArray for every updateKernel launch - the randArray contains random
 * angles for each particle.
 * 
 */

//function to seed states
__global__ void init_stuff (curandState* state, unsigned long seed){
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     curand_init(seed, idx, 0, &state[idx]);
}
//function to generate random numbers using seed states and copy them to randArray
__global__ void make_rand (curandState* state, float* randArray){
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     randArray[idx] = curand_uniform(&state[idx]);
}

void Swarm::launchRandInit(unsigned long t){
     init_stuff <<<(nParticles / 256) + 1, 256>>> (d_state, t);
}

//----------------------------------------------------------------------------------------------//
//--------------------------End of Random Number Generation protocols---------------------------//
//----------------------------------------------------------------------------------------------//





void Swarm::update(){
     make_rand <<<(nParticles / 256) + 1, 256>>> (d_state, randArray);
     updateKernel <<<(nParticles / 256) + 1, 256>>> (d_particles_old,d_particles_new, nParticles, L, d_c, d_dist, randArray);
}
