/* Class declaration for swarm class
 *   
 *   Written by : Tapan Goel
 *   Date : 29th December 2015
 * 
 * Description : Class called Swarm
 * Has following attributes : 
 * Number of Particles 		: nParticles - integer
 * Storage space required 	: size - integer
 * Length of arena        	: L - float
 * Array of Particles on host 	: *h_particles - pointer to array of particles
 * Array of Particles on device : *d_particles - pointer to array of particles
 * Array of random angles	: *randArray - float array of random numbers generated from Unif(0,1)
 * Order Parameter 		: orderParam - float
 * Particle statedState		: *d_state  - array of curandState
 * 
 * Functions: 
 * 
 * Constructor			:Swarm(n, systemsize) - Takes number of particles and systemsize as arguments and assigns them to nParticles
 *				   and L. Allocates spaces for particles on host and initializes the size variable.
 * Initializer			:init(noise) - Takes maximum value of eta as a parameter and calls initialize function for all particles
 * 
 * Allocator			:allocate() - allocates memory for GPU variables on CUDA. returns 0 if allocation does not happen for some variable.
 * 
 * Copy to CUDA			:cudaCopy() - Copies all particles from system to GPU.
 * 
 * Updater			:update() - {defined in the file kernel.cu} Puts random numbers in randArray using d_state as seed.
 *				  Launches updatekernel as defined in kernel.cu 
 *				  
 * Random number generator	:launchRandInit(unsigned long t) - assigns random values to d_state using time as seed.
 * 
 * Order parameter calculation	:calcOrderparam() - returns order parameter.
 * 
 * Retrieve data from device	:cudaBackCopy() - copies all particles from GPU to host.
 * 
 * Destructor			: ~Swarm() - frees up memory on GPU and the host. 
 */


#include "Swarm.h"
#include <cuda_runtime.h>

Swarm::Swarm(int n, float systemsize){
     L = systemsize;	
     nParticles = n;
     h_particles = new Particle[nParticles];
     
     size = sizeof(Particle) * nParticles;
}

void Swarm::init(float *eta){
     for (int i = 0; i < nParticles; i++){
	  h_particles[i].initialize(L,eta[i]);
	  
     }
}

int Swarm::allocate(){
     if (cudaMalloc(&d_particles_old, size) != cudaSuccess){
	  cout << "unable to allocate memory on device" << endl;
	  delete []h_particles;
	  return 0;
     }
     
     if (cudaMalloc(&d_particles_new, size) != cudaSuccess){
	  cout << "unable to allocate memory on device" << endl;
	  delete []h_particles;
	  cudaFree(d_particles_old);
	  
	  return 0;
     }
//      if (cudaMalloc(&d_sumdir, (sizeof(float2) * nParticles)) != cudaSuccess){
// 	  cout << "unable to allocate memory on device" << endl;
// 	  delete []h_particles;
// 	  cudaFree(d_particles);
// 	  return 0;
//      }
     if (cudaMalloc(&d_c, (sizeof(float) * nParticles)) != cudaSuccess){
	  cout << "unable to allocate memory on device" << endl;
	  delete []h_particles;
	  cudaFree(d_particles_old);
	  cudaFree(d_particles_new);
//	  cudaFree(d_sumdir);
	  return 0;
     }
     if (cudaMalloc(&d_dist, (sizeof(float) * nParticles)) != cudaSuccess){
	  cout << "unable to allocate memory on device" << endl;
	  delete []h_particles;
	  cudaFree(d_particles_old);
	  cudaFree(d_particles_new);
	  cudaFree(d_c);
//	  cudaFree(d_sumdir);
	  return 0;
     }
     if (cudaMalloc(&d_state, nParticles * sizeof(curandState)) != cudaSuccess){
	  cout << "unable to allocate memory on device" << endl;
	  delete []h_particles;
	  cudaFree(d_particles_old);
	  cudaFree(d_particles_new);
//	  cudaFree(d_sumdir);
	  cudaFree(d_c);
	  cudaFree(d_dist);
	  return 0;
     }
     if (cudaMalloc(&randArray, nParticles * sizeof(float)) != cudaSuccess){
	  cout << "unable to allocate memory on device" << endl;
	  delete []h_particles;
	  cudaFree(d_particles_old);
	  cudaFree(d_particles_new);
	  cudaFree(d_c);
//	  cudaFree(d_sumdir);
	  cudaFree(d_dist);
	  cudaFree(d_state);
	  return 0;
     }
}

int Swarm::cudaCopy(){
     checkCudaErrors(cudaMemcpy(d_particles_old, h_particles, size, cudaMemcpyHostToDevice));
}


int Swarm::cudaBackCopy(){
     checkCudaErrors(cudaMemcpy(h_particles, d_particles_new, size, cudaMemcpyDeviceToHost));
     //checkCudaErrors(cudaMemcpy(h_sumdir, d_sumdir, (sizeof(float2) * nParticles), cudaMemcpyDeviceToHost));
     //checkCudaErrors(cudaMemcpy(h_c, d_c (sizeof(float) * nParticles), cudaMemcpyDeviceToHost));	
     //cout << h_particles[0].coord.x << "\t" << h_particles[0].coord.y << "\n";
}

float Swarm::calcOrderparam(){
     float sumxspeed = 0,sumyspeed = 0;
     
     for (int i = 0; i < nParticles; i++) {
	  
	  sumxspeed += h_particle[i].dir.x;
	  sumyspeed += h_particle[i].dir.y;
     }
     orderParam = (pow((pow(sumxspeed,2) + pow(sumyspeed,2)),0.5) / nParticles);
     return orderParam;
}

Swarm::~Swarm(){
     delete []h_particles;
     cudaFree(d_particles);
   //  cudaFree(d_sumdir);
     cudaFree(d_c);
     cudaFree(d_dist);
     cudaFree(d_state);
     cudaFree(randArray);
     
}