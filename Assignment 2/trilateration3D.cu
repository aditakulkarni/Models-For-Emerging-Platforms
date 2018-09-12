#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

typedef struct point {
	float x;
	float y;
	float z;
}POINT;

typedef struct distance {
	float da;
	float db;
	float dc;
}DISTANCE;

DISTANCE calculate_euclidean(POINT,POINT,POINT,POINT);

int NUM = pow(2,15);

__device__ POINT subtract(POINT *a, POINT *b) {
	POINT c;
	c.x = a->x - b->x;	c.y = a->y - b->y;	c.z = a->z - b->z;
	return c;
}

__device__ POINT add(POINT *a, POINT *b) {
	POINT c;
	c.x = a->x + b->x;	c.y = a->y + b->y;	c.z = a->z + b->z;
	return c;
}
	
__device__ POINT divide(POINT *a, float b) {
	POINT c;
	c.x = a->x / b;	c.y = a->y / b;	c.z = a->z / b;
	return c;
}

__device__ POINT multiply(POINT *a, float b) {
	POINT c;
	c.x = a->x * b;	c.y = a->y * b;	c.z = a->z * b;
	return c;
}

__device__ POINT cross(POINT *a, POINT *b) {
		POINT c;
		c.x =a->y * b->z - a->z * b->y;		c.y =a->z * b->x - a->x * b->z;		c.z = a->x * b->y - a->y * b->x;
		return c;
}

__global__ void calculate_trail(DISTANCE *d, POINT *point1, POINT *point2, POINT *point3, POINT *result)
{
	POINT ex, ey, ez,a,location;
	float i, j, d1, x, y, z, pa, pb, pc;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

 	POINT sub = subtract(point2, point1);
	ex = divide(&sub, sqrtf(powf(sub.x,2) + powf(sub.y,2) + powf(sub.z,2)));	
	POINT subP3P1 = subtract(point3, point1);
	i = ex.x * subP3P1.x + ex.y * subP3P1.y + ex.z * subP3P1.z;
	POINT mul = multiply(&ex, i);
	a = subtract(&subP3P1, &mul);
	ey = divide(&a, sqrtf(powf(a.x,2) + powf(a.y,2) + powf(a.z,2)));
	ez = cross(&ex, &ey);
	sub = subtract(point2, point1);
	d1 = sqrtf(powf(sub.x,2) + powf(sub.y,2) + powf(sub.z,2));
	pa = powf(d[index].da,2);
	pb = powf(d[index].db,2);
	pc = powf(d[index].dc,2);
	j = ey.x * subP3P1.x + ey.y * subP3P1.y + ey.z * subP3P1.z;	
	x = ( pa - pb + powf(d1,2)) / (2 * d1);
	y = ( pa - pc + powf(i,2) + powf(j,2)) / (2 * j) - (i / j) * x;		
	z = sqrtf(powf(d[index].da,2) - powf(x,2) - powf(y,2));
	POINT m1 = multiply(&ex, x);
	POINT m2 = multiply(&ey, y);
	mul = add(&m1, &m2);
	a = add(point1, &mul);
	mul = multiply(&ez, z);
	location = subtract(&a, &mul);
	d[index].da = location.x;
	d[index].db = location.y;
	d[index].dc = location.z;

	// Wait till all the positions are calculated
	__syncthreads();
	if(index%4 == 0) {
		int temp_index = index/4;
		float x_sum = 0.0, y_sum = 0.0, z_sum = 0.0;

		// Averages 4 positions
		for(int i=0;i<4;i++) {
			x_sum += d[index+i].da;
			y_sum += d[index+i].db;
			z_sum += d[index+i].dc;
		}
		result[temp_index].x = x_sum/4.0;
	    result[temp_index].y = y_sum/4.0;
	    result[temp_index].z = z_sum/4.0;
	}

}

int main()
{
	float dx = 0.5, dy =0.5, dz = 0.5;
	POINT a,b,c;
	POINT *p = (POINT*) malloc(sizeof(POINT)*NUM);
	POINT *result_p = (POINT*) malloc(sizeof(POINT)*NUM);
	DISTANCE *d =(DISTANCE *) malloc(sizeof(DISTANCE)*NUM);
	POINT *point1, *point2, *point3;
	DISTANCE *cuda_d;
	POINT *cuda_p;
	struct timeval t1, t2;

	int U=3,V=128/2;

	// Allocate memory on GPU
	cudaMalloc(&cuda_d, sizeof(DISTANCE)*NUM);
	cudaMalloc((void **)&cuda_p, sizeof(POINT)*NUM);
	cudaMalloc((void **)&point1, sizeof(POINT));
	cudaMalloc((void **)&point2, sizeof(POINT));
	cudaMalloc((void **)&point3, sizeof(POINT));

	a.x = 4.0;
	a.y = 4.0;
	a.z = 1.0;
	b.x = 9.0;
	b.y = 7.0;
	b.z = 2.0;
	c.x = 9.0;
	c.y = 1.0;
	c.z = 3.0;
	p[0].x = 2.5;
	p[0].y = 1.0;
	p[0].z = 1.5;
	d[0] = calculate_euclidean(p[0],a,b,c);

	// Generate sequence of positions by adding delta 
	for(int i=1;i<NUM;i++) {
		p[i].x = p[i-1].x + dx;
		p[i].y = p[i-1].y + dy;
		p[i].z = p[i-1].z + dz;
		d[i] = calculate_euclidean(p[i],a,b,c);
	}

	printf("\n\nResult from self-verification :\n");
	for(int i=0;i<NUM;i++) {
		printf("\n%.2f %.2f %.2f",p[i].x,p[i].y,p[i].z);
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
		printf("\n%.2f %.2f %.2f",result_p[i].x,result_p[i].y,result_p[i].z);
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
	d.da = sqrt(pow((a.x-p.x),2)+pow((a.y-p.y),2)+pow((a.z-p.z),2));
	d.db = sqrt(pow((b.x-p.x),2)+pow((b.y-p.y),2)+pow((b.z-p.z),2));
	d.dc = sqrt(pow((c.x-p.x),2)+pow((c.y-p.y),2)+pow((c.z-p.z),2));
	return d;
}
