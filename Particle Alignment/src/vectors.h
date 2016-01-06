/* Structure and functions related to implementing 2-D vectors	
     
     Written by : Tapan Goel
     Date : 15th December 2015

Description : Structure called float2 which emulates a 2-D vector with components x and y. 
	      Functions declared: 
	      Norm
	      Normalize
	      Scale
	      Negate

	      Addition 
	      Subtraction
	      Dot Product
	      Cross Product
	      Distance
	      Distance along periodic boundary
*/

#ifndef VECTORS_H_
#define VECTORS_H_

#include <stdlib.h>

struct float2{

     float x;
     float y;
};

float norm(struct float2 v1);

struct float2 normalize(struct float2 v1);

struct float2 scale(struct float2 v1, float a);

struct float2 _(struct float2 v1);

struct float2 add(struct float2 v1, struct float2 v2);

struct float2 subtract(struct float2 v1, struct float2 v2);

float dot(struct float2 v1, struct float2 v2);

float cross(struct float2 v1,struct float2 v2);

float distance(struct float2 v1,struct float2 v2);

float distance_periodic(struct float2 v1, struct float2 v2, float L);


#endif
