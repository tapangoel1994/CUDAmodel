#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<unistd.h>
using namespace std;

#define PI 3.14159

struct particle{
     
     float x;
     float y;
     float theta;
};



void main()
{
     
     struct particle *d_fish[N],*d_host;
     
     initialize()
     
     
}





float simulate(float x[2][N],float y[2][N],float theta[2][N],long int T, FILE *fp); //Runs simulation
void initialize(float x[2][N],float y[2][N],float theta[2][N]); // Gives random initial values to position and direction of each particle
void update_pos(float x[2][N],float y[2][N],float theta[2][N]); // Updates position of each particle
void update_vel(float x[2][N],float y[2][N],float theta[2][N]); // Updates velocity of each particle
float distance(float x1,float x2,float y1,float y2); // calculates distance along torus between two points
float arctan(float sin,float cos); //generates the angle between [-PI PI] for given sin and cos values
float limit(float x); // limits angle to [-PI PI] for some angle x
void swap(float x[2][N],float y[2][N],float theta[2][N]); // x[0][i] = x[1][i] and so for other variables
float Orderparameter(float theta[2][N]); 
float *correlation(float x[2][N],float y[2][N],float theta[2][N],float l,float delta_l);


void timeseriesplot();
void plot(FILE *fp);
void write_to_file(float x[2][N],float y[2][N]);



int main()
{
     float x[2][N],y[2][N],theta[2][N];
     int n=1;
     float v_a[n],mean,stdev;
     float density;
     char name[100];
     long int T = 5000;
     FILE *data;
     
     
     density = 5.00;
     
     L = sqrt(N/density);
     
     for(eta = 0;eta <= 7;eta+= .25)
     {
	  sprintf(name,"Correlations/Density_%.2fNoise_%.2f.csv",density,eta);
	  printf("%s\n",name);
	  data = fopen(name,"w");
	  
	  v_a[0] = simulate(x,y,theta,T,data); 
	  
	  fclose(data);
	  
     }
     
     
     return 0;
}


float simulate(float x[2][N],float y[2][N],float theta[2][N],long int T,FILE *fp)
{
     
     // FILE *gnupipe;
     //  gnupipe = popen("gnuplot -persistent","w");
     
     
     initialize(x,y,theta);   
     long int t;
     int i;
     for(t=0;t<T;t++)
     {
	  update_pos(x,y,theta);
	  update_vel(x,y,theta);
	  
	  // fprintf(fp,"%ld\t%f\n",t,Orderparameter(theta));
	  // write_to_file(x,y);
	  
	  //plot(gnupipe);
	  
	  swap(x,y,theta);
     }
     
     // pclose(gnupipe);
     
     for(i = 0;i < (N-1);i++)
     {
	  fprintf(fp,"x%d,y%d,theta%d,",i,i,i); 
	  
     }
     fprintf(fp,"x%d,y%d,theta%d\n",i,i,i);
     
     for(t=0;t<500;t++)
     {
	  update_pos(x,y,theta);
	  update_vel(x,y,theta);
	  
	  for(i = 0; i < (N-1); i++)
	  {
	       fprintf(fp,"%.2f,%.2f,%.2f,",x[1][i],y[1][i],theta[1][i]);
	       
	  }
	  
	  fprintf(fp,"%.2f,%.2f,%.2f\n",x[1][i],y[1][i],theta[1][i]);
	  
	  
	  swap(x,y,theta);
     }
     
     
     return (Orderparameter(theta));
}


void initialize(float x[2][N],float y[2][N],float theta[2][N])
{
     srand(time(NULL));
     
     for(int i=0;i<N;i++)
     {
	  x[0][i]= ( (float)rand()/(float)RAND_MAX )*L;
	  y[0][i]= ( ((float)rand())/((float)RAND_MAX) )*L;
	  theta[0][i] = -PI + ( ((float)rand())/ ((float)RAND_MAX) )*2*PI;   
     }
}

void update_pos(float x[2][N],float y[2][N],float theta[2][N])
{
     for(int i=0;i<N;i++)
     {
	  x[1][i] = x[0][i] + v*cos(theta[0][i])*delta_t;
	  y[1][i] = y[0][i] + v*sin(theta[0][i])*delta_t;
	  
	  if(x[1][i] > L)
	  {
	       x[1][i] -= L;
	  }
	  
	  if(x[1][i] < 0)
	  {
	       x[1][i] += L;
	  }
	  
	  if(y[1][i] > L)
	  {
	       y[1][i]-= L;
	  }
	  
	  if(y[1][i] < 0)
	  {
	       y[1][i] += L;
	  }
	  
     }  
}
void update_vel(float x[2][N],float y[2][N],float theta[2][N])
{
     for(int i=0; i<N;i++)
     {
	  float sumsin;
	  float sumcos;
	  float count;
	  float d;
	  sumsin = 0;
	  sumcos = 0;
	  count = 0;
	  
	  for(int j =0 ; j < N; j++)
	  {
	       d = distance(x[0][i],x[0][j],y[0][i],y[0][j]);
	       
	       if(d < r)
	       {
		    sumsin += sin( theta[0][j]);
		    sumcos += cos( theta[0][j]);
		    count += 1;
	       }
	  }
	  
	  float sinavg;
	  float cosavg;
	  float thetaavg;
	  float delta_theta; 
	  
	  if(count == 0) {
	       count = 1;
	       sumsin = sin(theta[0][i]);
	       sumcos = cos(theta[0][i]);
	  }
	  
	  sinavg = sumsin/count;
	  cosavg = sumcos/count;
	  thetaavg = arctan(sinavg,cosavg); 
	  delta_theta = eta*( (float)rand()/(float)RAND_MAX ) - eta/2;
	  
	  theta[1][i] = thetaavg + delta_theta;
	  
	  theta[1][i] = limit(theta[1][i]);
     }
}

void swap(float x[2][N],float y[2][N],float theta[2][N])
{
     for(int i=0;i<N;i++)
     {
	  x[0][i] = x[1][i];
	  y[0][i] = y[1][i];
	  theta[0][i] = theta[1][i]; 
     }
}

float Orderparameter(float theta[2][N])
{
     float sumcos,sumsin;
     sumcos =0;
     sumsin =0;
     
     for(int i=0;i<N;i++)
     {
	  sumsin += sin(theta[1][i]);
	  sumcos += cos(theta[1][i]);
     }
     
     return((sqrt(sumsin*sumsin + sumcos*sumcos))/N);
}

float distance(float x1,float x2,float y1,float y2)
{
     float dx,dy;
     dx = fabs(x1-x2);
     dy = fabs(y1-y2);
     
     if(dx > L/2)
	  dx = L-dx;
     if( dy > L/2)
	  dy = L-dy;
     
     return( sqrt(dx*dx + dy*dy) );
}

float arctan(float sin,float cos)
{
     if(sin > 0  && cos > 0) return(atan(sin/cos));
     if(sin < 0  && cos > 0) return(atan(sin/cos));
     if(sin > 0  && cos < 0) return(PI + atan(sin/cos));
     if(sin < 0  && cos < 0) return(-PI + atan(sin/cos));
     
     if(sin == 0 && cos > 0) return(0);
     if(sin == 0 && cos < 0) return(PI);
     if(sin > 0 && cos == 0) return(PI/2);
     if(sin < 0 && cos == 0) return(-PI/2);
     
}

float limit(float x)
{
     if(x <= PI && x >= -PI) return(x);
     
     while(!(x <=PI && x >= -PI))
     {
	  
	  if(x > PI && x <= 2*PI) x = x - 2*PI;
	  else if(x < -PI && x >= -2*PI) x = x + 2*PI;
	  else
	  {
	       if(x > 2*PI)
	       {
		    while(x >= 2*PI) x = x - 2*PI;
	       }
	       
	       if(x < 0)
	       {
		    while(x <= 0) x = x + 2*PI;
	       }
	  }  
     }
     
     return x;
     
}
