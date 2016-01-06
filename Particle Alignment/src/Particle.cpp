/* Class function definition for particle class
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
 Initializer : initialize - gives initial values to all variables. Takes the length of the system and noise range as argument.
 Destructor
 
 The definitions of the class functions are given
 */


#include "Particle.h"
#define PI 3.14

Particle::Particle(){}

Particle::~Particle(){}

void Particle::initialize(float systemsize, float eta)
{
     float theta;
     
     pos.x = (1.0 * rand() / RAND_MAX) * systemsize;
     pos.y = (1.0 * rand() / RAND_MAX) * systemsize;
     
     theta = (2.0 * rand() / RAND_MAX) * PI;
     dir.x = cos(theta);
     dir.y = sin(theta);
     
     R_s = 1;
     s = 0.003;
     
     noise = eta;
     
}