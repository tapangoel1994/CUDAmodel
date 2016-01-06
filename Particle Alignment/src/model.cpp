/*   main.cpp
 * Author: Tapan Goel
 * Created on 31st December 2015
 */

#include <iostream>
#include "Swarm.h"

using namespace std;

int main ()

{
     srand(time(NULL));
     
     time_t t;
     
     time(&t);
     
     int N,systemsize;		//input N and systemsize
     float *noise,OP;
     long int Total_Time,ntimesteps;
     
     
     noise = new float[N];
     
     for(int i = 0; i < N;i++)
     {
	  noise[i] = 1; // Enter value of noise for ith particle.
     }
     
     
     Swarm swarm(N,systemsize);
    
     swarm.init(noise);
     swarm.allocate();
     swarm.launchRandInit((unsigned long) t);
     
     swarm.cudaCopy();
     
     for(ntimesteps =0; ntimesteps < Total_Time;ntimesteps++)
     {
	  swarm.update();
	  swarm.cudaBackCopy();
	  OP = swarm.calcOrderparam();
	  printf("\n Orderparameter = %f",OP);
	  
	  swarm.cudaCopy();
	  
     }
     
     
     return 0;
}