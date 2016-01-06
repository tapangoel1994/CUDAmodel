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
     dist = powf( (powf(dx, 2) + powf(dy, 2)), 0.5);
     return dist;
}



__global__ void updateKernel (Particle *d_particles_old, Particle *d_particles_new, int nParticles, float L, float *d_c, float *d_dist, float* randArray){

     
     float distance;
     float2 sumdir;
     float avgtheta;
     
     int idx_focal = threadIdx.x + blockIdx.x * blockDim.x;
     idx_focal = max(idx_focal, 0);			//checking idx to be between 0 and maximum number of particles
     idx_focal = min(idx_focal, nParticles);
     
     
     d_c[idx_focal] = 0.0;
     sumdir.x = 0;
     sumdir.y = 0;
    
     
     for(int cid = 0; cid < nParticles; cid++)
     {
	  distance = calcDist(d_particles_old[idx_focal].pos, d_particles_old[cid].pos, L);
	  
	  if(distance <= d_particles_old[idx_focal].R_s)
	  {
	       sumdir.x += d_particles_old[cid].dir.x;
	       sumdir.y += d_particles_old[cid].dir.y;
	       d_c[idx_focal] += 1;
	  }
	  
     }
     
     avgtheta = atan2(sumdir.y,sumdir.x);
     
     avgtheta += d_particles_old[idx_focal].noise * ( (randArray[idx_focal] * 2.0) - 1) / 2.0;
     
     d_particle_new[idx_focal].dir.x = cos(avgtheta);
     d_particle_new[idx_focal].dir.y = sin(avgtheta);
     
     d_particle_new[idx_focal].R_s = d_particle_old[idx_focal].R_s ;
     d_particle_new[idx_focal].s = d_particle_old[idx_focal].s;
     d_particle_new[idx_focal].noise = d_particle_old[idx_focal].noise;
     
     d_particle_new[idx_focal].pos.x = d_particles_old[idx_focal].pos.x + d_particle_new[idx_focal].s*d_particle_new[idx_focal].dir.x;
     d_particle_new[idx_focal].pos.y = d_particles_old[idx_focal].pos.y + d_particle_new[idx_focal].s*d_particle_new[idx_focal].dir.y;
     
     d_particles_new[idx_focal].pos.x = doPeriodic(d_particles_new[idx_focal].pos.x, L);		
     d_particles_new[idx_focal].pos.y = doPeriodic(d_particles_new[idx_focal].pos.y, L);
     
     
     __syncthreads();
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
