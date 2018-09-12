#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define NUM pow(2,15)

typedef struct point {
	float x;
	float y;
}POINT;

typedef struct distance {
	float da;
	float db;
	float dc;
}DISTANCE;

DISTANCE calculate_euclidean(POINT,POINT,POINT,POINT);

__global__ void calculate_trail(DISTANCE *d, POINT *point1, POINT *point2, POINT *point3, POINT *result) {
	POINT ex,ey,temp;
    float d12,i,j,x,y,a,b,c,dp;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	d12 = powf(powf(point2->x-point1->x,2) + powf(point2->y-point1->y,2),0.5);
    ex.x = (point2->x-point1->x)/d12;
	ex.y = (point2->y-point1->y)/d12;
    i = ex.x * (point3->x-point1->x) + ex.y * (point3->y-point1->y);
    temp.x = point3->x-point1->x-i*ex.x;
	temp.y = point3->y-point1->y-i*ex.y;
    ey.x = temp.x / powf(powf(temp.x,2)+powf(temp.y,2),0.5);
	ey.y = temp.y / powf(powf(temp.x,2)+powf(temp.y,2),0.5);
    j = ey.x * (point3->x-point1->x) + ey.y * (point3->y-point1->y);
	a = powf(d[index].da,2);
	b = powf(d[index].db,2);
	c = powf(d[index].dc,2);
	dp = powf(d12,2);
    x = ( a - b + dp)/ (2 * d12);
    y = (a - c + powf(i,2) + pow(j,2))/(2*j) - i*x/j;
	d[index].da = point1->x+ x*ex.x + y*ey.x;
	d[index].db = point1->y+ x*ex.y + y*ey.y;

	// Wait till all the positions are calculated
	__syncthreads();
	if(index%4 == 0) {
		int temp_index = index/4;
		float x_sum = 0.0, y_sum = 0.0;

		// Averages 4 positions
		for(int i=0;i<4;i++) {
			x_sum += d[index+i].da;
			y_sum += d[index+i].db;
		}
		result[temp_index].x = x_sum/4.0;
	    result[temp_index].y = y_sum/4.0;
	}
}

int main()
{
	float dx = 0.5, dy =0.5;
	POINT a,b,c;
	POINT *p = (POINT*) malloc(sizeof(POINT)*NUM);
	POINT *result_p = (POINT*) malloc(sizeof(POINT)*NUM);
	DISTANCE *d =(DISTANCE *) malloc(sizeof(DISTANCE)*NUM);
	POINT *point1, *point2, *point3;
	DISTANCE *cuda_d;
	POINT *cuda_p;
	struct timeval t1, t2;

	int U=3/2,V=128;

	// Allocate memory on GPU
	cudaMalloc(&cuda_d, sizeof(DISTANCE)*NUM);
	cudaMalloc((void **)&cuda_p, sizeof(POINT)*NUM);
	cudaMalloc((void **)&point1, sizeof(POINT));
	cudaMalloc((void **)&point2, sizeof(POINT));
	cudaMalloc((void **)&point3, sizeof(POINT));

	a.x = 4.0;
	a.y = 4.0;
	b.x = 9.0;
	b.y = 7.0;
	c.x = 9.0;
	c.y = 1.0;
	p[0].x = 2.5;
	p[0].y = 1.0;
	d[0] = calculate_euclidean(p[0],a,b,c);

	// Generate sequence of positions by adding delta
	for(int i=1;i<NUM;i++) {
		p[i].x = p[i-1].x + dx;
		p[i].y = p[i-1].y + dy;
		d[i] = calculate_euclidean(p[i],a,b,c);
	}

	printf("\n\nResult from self-verification :\n");
	for(int i=0;i<NUM;i++) {
		printf("\n%.2f %.2f",p[i].x,p[i].y);
	}

	// Copy data to GPU memory
	cudaMemcpy(cuda_d, d, sizeof(DISTANCE)*NUM, cudaMemcpyHostToDevice);
	cudaMemcpy(point1, &a, sizeof(POINT), cudaMemcpyHostToDevice);
	cudaMemcpy(point2, &b, sizeof(POINT), cudaMemcpyHostToDevice);
	cudaMemcpy(point3, &c, sizeof(POINT), cudaMemcpyHostToDevice);

    gettimeofday(&t1, 0);

	// Calling Device Function
	calculate_trail<<<U,V>>>(cuda_d,point1,point2,point3,cuda_p);

    gettimeofday(&t2, 0);

	cudaMemcpy(result_p, cuda_p, sizeof(POINT)*NUM, cudaMemcpyDeviceToHost);

	// Calculate time elapsed
	double time = (1000000.0*(t2.tv_sec - t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

	printf("\n\nResult from GPU :\n");
	for(int i=0;i<(U*V)/4;i++) {
		printf("\n%.2f %.2f",result_p[i].x,result_p[i].y);
	}

	printf("\n\nTime elapsed : %3.3f ms",time);

	printf("\n");

	// Free memory
	free(p);
	free(result_p);
	free(d);

	cudaFree(cuda_d);
	cudaFree(cuda_p);
	cudaFree(point1);
	cudaFree(point2);
	cudaFree(point3);

	return 0;
}

// Function to calculate distance between 2 points
DISTANCE calculate_euclidean(POINT p, POINT a, POINT b, POINT c) {
	DISTANCE d;
	d.da = sqrt(pow((a.x-p.x),2)+pow((a.y-p.y),2));
	d.db = sqrt(pow((b.x-p.x),2)+pow((b.y-p.y),2));
	d.dc = sqrt(pow((c.x-p.x),2)+pow((c.y-p.y),2));
	return d;
}
