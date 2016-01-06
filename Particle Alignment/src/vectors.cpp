/* Structure and functions related to implementing 2-D vectors	
     
     Written by : Tapan Goel
     Date : 15th December 2015

Description : Structure called float2 which emulates a 2-D vector with components x and y. 
	      Functions : 
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


#include <math.h>
#include "vectors.h"

float norm(struct float2 v1)
{

     return( sqrt(v1.x*v1.x + v1.y*v1.y) );
     
}




struct float2 subtract(struct float2 v1, struct float2 v2)
{
     struct float2 v3;
     v3.x = v1.x - v2.x;
     v3.y = v1.y - v2.y;
     
     return v3;  
}

struct float2 add(struct float2 v1, struct float2 v2)
{
     struct float2 v3;
     v3.x = v1.x + v2.x;
     v3.y = v1.y + v2.y;
     
     return v3;  
}

struct float2 scale(struct float2 v1, float a)
{
     struct float2 v2;
     v2.x = v1.x*a;
     v2.y = v1.y*a;
     
     return v2;
}

float distance(struct float2 v1,struct float2 v2)
{
     struct float2 v3;
     
     v3 = subtract(v1,v2);

     return(norm(v3));
}

float distance_periodic(struct float2 v1, struct float2 v2, float L)
{
     float x,dx,dy;
     
     dx = fabs(v1.x - v2.x);
     dy = fabs(v1.y - v2.y);
     if(dx > L/2) dx = L- dx;
     if(dy > L/2) dy = L- dy;
     x = (dx*dx) + (dy*dy);
     x = sqrt(x);
     return x;
     
}

struct float2 normalize(struct float2 v1)
{
     float norm;
     
     norm = sqrt( v1.x*v1.x + v1.y*v1.y );
     if(norm)
     {
	  v1.x = v1.x/norm;
	  v1.y = v1.y/norm;
     }
     
     
     return v1;
}

float dot(struct float2 v1, struct float2 v2)
{
     float x;
     
     x= v1.x*v2.x + v1.y*v2.y;
     
     return x;
}


float cross(struct float2 v1,struct float2 v2)
{
     float x;
     
     x = v1.x*v2.y - v2.x*v1.y;
     
     return x;
}

struct float2 _(struct float2 v1)
{
     struct float2 v2;
     v2.x = -v1.x;
     v2.y = -v1.y;
     
     return v2;
}

	      