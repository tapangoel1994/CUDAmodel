/* Class declaration for swarm class
 *   
 *   Written by : Tapan Goel
 *   Date : 29th December 2015
 * 
 Description : Class called Swarm
 Has following attributes : 
 Number of Particles 		: nParticles - integer
 Storage space required 	: memorysize - integer
 Length of arena        	: L - float
 Array of Particles on host 	: *h_particles - pointer to array of particles
 Array of Particles on device 	: *d_particles_old - pointer to array of particles at time t-1
				: *d_particles_new - pointer to array of particles at time t
 Array of random angles		: *randArray - float array
 Order Parameter 		: orderParam - float
 Particle statedState		: *d_state  - array of curandState
 Functions: 
 
 Constructor			:Swarm(n, systemsize) - Takes number of particles and systemsize as arguments and assigns them to nParticles
				   and L. Allocates spaces for particles on host and initializes the size variable.
 Initializer			:init(*noise) - Takes maximum value of eta for each particle as a parameter and calls initialize function for all particles
 
 Allocator			:allocate() - allocates memory for GPU variables on CUDA. returns 0 if allocation does not happen for some variable.
 
 Copy to CUDA			:cudaCopy() - Copies all particles from system to GPU.
 
 Updater			:update() - {defined in the file kernel.cu} Puts random numbers in randArray using d_state as seed.
					     Launches updatekernel as defined in kernel.cu 
				  
 Random number generator	:launchRandInit(unsigned long t) - assigns random values to d_state using time as seed.
 
 Order parameter calculation	:calcOrderparam() - returns order parameter.
 
Retrieve data from device	:cudaBackCopy() - copies all particles from GPU to host.
 
Destructor			: ~Swarm() - frees up memory on GPU and the host. 
 */

#ifndef SWARM_H_
#define SWARM_H_

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "Particle.h"
#include <curand_kernel.h>
using namespace std;

class Swarm{
     
public:
     int nParticles;	
     int memorysize;
     float L;
     Particle *h_particles;
     Particle *d_particles_old;
     Particle *d_particle_new;
     float *d_c, *d_dist, *randArray;
     float orderParam;
     curandState *d_state;
     
public:
     Swarm(int n, float systemsize);
     void init(float *noise);
     int allocate();
     int cudaCopy();
     int update();
     void launchRandInit(unsigned long t);
     float calcOrderparam();
     int cudaBackCopy();
     const Particle const *returnParticles(){ return h_particles; };
     ~Swarm();
     
};

#endif