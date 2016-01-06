/* Class declaration for particle class
 *   
 *   Written by : Tapan Goel
 *   Date : 16th December 2015
 * 
 Description : Class called Particle
 Has following attributes : 
 Position of particle 	: pos - 2-D vector
 Orientiation direction : dir - 2-D vector
 Speed 			: s - float
 Interaction radius 	: R_s - float
 Functions: 
 Constructor : Empty
 Initializer : initialize - gives initial values to all variables. Takes the length of the system and noise range as arguments.		   
 Destructor
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "../utils/cuda_vector_math.cuh"
#include<stdio.h>

class Particle {
     
public : 
     float2 pos,dir;
     float s, R_s,noise;
     
public : 
     Particle();
     void initialize(float systemsize, float eta);
     ~Particle();
     
};


#endif