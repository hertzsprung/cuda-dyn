#include <cstdio> 

#define NUMERIC_TYPE float
#define C(x) x
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr, "%s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

__constant__ int xsz;
__constant__ NUMERIC_TYPE end_time = C(1000.0);
__managed__ bool finished = false;
__managed__ NUMERIC_TYPE t = C(0.0);
__device__ NUMERIC_TYPE next_snapshot_time = C(100.0);

__global__ void update_flow_variables(NUMERIC_TYPE* U) 
{ 
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;

	for (int i=global_i+1; i<xsz; i+=blockDim.x*gridDim.x)
	{
		U[i] += C(0.25);
	}
} 

__global__ void simulate(NUMERIC_TYPE* U) 
{ 
	NUMERIC_TYPE dt = C(0.1);
	printf("DEV next_snapshot_time=%f\n", next_snapshot_time);

	while (t < end_time && t < next_snapshot_time)
	{
    	update_flow_variables<<<64, 256>>>(U);
		if (cudaDeviceSynchronize() != cudaSuccess) return;

		t += dt;

		if (t >= end_time)
		{
			finished = true;
		}
	}

	if (t >= next_snapshot_time)
	{
		next_snapshot_time += C(100.0);
		// TODO
	}
} 

int main(int argc, char *argv[]) 
{ 
	const int xsz_H = 2 << 24; // ~16M elements
	const size_t U_size = xsz_H*sizeof(NUMERIC_TYPE);

	checkCudaErrors(cudaMemcpyToSymbol(xsz, &xsz_H, sizeof(int)));

	NUMERIC_TYPE* snapshot_H;
	checkCudaErrors(cudaMallocHost(&snapshot_H, U_size));
	memset(snapshot_H, 0, U_size);

	NUMERIC_TYPE* current_solution_D;
	checkCudaErrors(cudaMalloc(&current_solution_D, U_size));
	checkCudaErrors(cudaMemcpy(current_solution_D, snapshot_H, U_size, cudaMemcpyHostToDevice));

	printf("HOST initialised\n");

	while (!finished)
	{
		simulate<<<1,1>>>(current_solution_D); 
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(snapshot_H, current_solution_D, U_size, cudaMemcpyDeviceToHost));
		printf("HOST t=%f\n", t);
	}

	checkCudaErrors(cudaFreeHost(snapshot_H));
	checkCudaErrors(cudaFree(current_solution_D));

    return 0; 
}
