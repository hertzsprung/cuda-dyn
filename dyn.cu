#include <cstdio> 
#include <future>

#define SNAPSHOT_INCREMENT 200.0f
#define SAMPLE_BUF_SIZE 1024

#define NUMERIC_TYPE float
#define C(x) x##f

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr, "%s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

typedef struct Samples
{
	NUMERIC_TYPE* time;
	NUMERIC_TYPE* value;
	int buf_size;
} Samples;

void allocate_D(Samples& buf, const int buf_size)
{
	buf.buf_size = buf_size;
	checkCudaErrors(cudaMalloc(&(buf.time), buf_size*sizeof(NUMERIC_TYPE)));
	checkCudaErrors(cudaMalloc(&(buf.value), buf_size*sizeof(NUMERIC_TYPE)));
}

void allocate_H(Samples& buf, const int buf_size)
{
	buf.buf_size = buf_size;
	checkCudaErrors(cudaMallocHost(&(buf.time), buf_size*sizeof(NUMERIC_TYPE)));
	checkCudaErrors(cudaMallocHost(&(buf.value), buf_size*sizeof(NUMERIC_TYPE)));
}

void copy_device_to_host(Samples& buf_H, Samples& buf_D, const int entries)
{
	checkCudaErrors(cudaMemcpy(buf_H.time, buf_D.time, entries*sizeof(NUMERIC_TYPE), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(buf_H.value, buf_D.value, entries*sizeof(NUMERIC_TYPE), cudaMemcpyDeviceToHost));
}

void deallocate_D(Samples& buf)
{
	checkCudaErrors(cudaFree(buf.time));
	checkCudaErrors(cudaFree(buf.value));
}

void deallocate_H(Samples& buf)
{
	checkCudaErrors(cudaFreeHost(buf.time));
	checkCudaErrors(cudaFreeHost(buf.value));
}

__constant__ int xsz;
__constant__ NUMERIC_TYPE end_time = C(1000.0);
__managed__ bool finished = false;
__managed__ bool write_snapshot = false;
__managed__ NUMERIC_TYPE t = C(0.0);
__device__ NUMERIC_TYPE next_snapshot_time = SNAPSHOT_INCREMENT;
Samples buf_D;
Samples buf_H;
__managed__ int sample_idx = 0;

void write(NUMERIC_TYPE* snapshot, const int xsz, NUMERIC_TYPE t)
{
	char filename[256];
	sprintf(filename, "results/%.3f.dat", t);
	FILE* file = fopen(filename, "wb");
	for (int i=0; i<xsz; i++)
	{
		fprintf(file, "%.1f\t", snapshot[i]);
		if (i % 80 == 79) fprintf(file, "\n");
	}
	fclose(file);
}

void write(FILE* file, Samples& buf, const int entries)
{
	for (int i=0; i<entries; i++)
	{
		fprintf(file, "%.2f %.2f\n", buf.time[i], buf.value[i]);
	}
}

__global__ void update_flow_variables(NUMERIC_TYPE* U) 
{ 
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;

	for (int i=global_i; i<xsz; i+=blockDim.x*gridDim.x)
	{
		NUMERIC_TYPE u = U[i];

		for (int l=0; l<10; l++)
		{
			u = u*u;
			u = sqrt(u);
		}

		U[i] = u + C(0.1);
	}
} 

__global__ void simulate(NUMERIC_TYPE* U, Samples buf) 
{ 
	NUMERIC_TYPE dt = C(0.1);
	write_snapshot = false;
	sample_idx = 0;

	while (t < end_time && t < next_snapshot_time && sample_idx < SAMPLE_BUF_SIZE)
	{
    	update_flow_variables<<<256, 256>>>(U);
		if (cudaDeviceSynchronize() != cudaSuccess) return;

		t += dt;
		buf.time[sample_idx] = t;
		buf.value[sample_idx] = U[0];
		sample_idx++;

		if (t >= end_time)
		{
			write_snapshot = true;
			finished = true;
		}
	}

	if (t >= next_snapshot_time)
	{
		write_snapshot = true;
		next_snapshot_time += SNAPSHOT_INCREMENT;
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

	allocate_D(buf_D, SAMPLE_BUF_SIZE);
	allocate_H(buf_H, SAMPLE_BUF_SIZE);
	FILE* samples_file = fopen("results/samples.dat", "wb");

	auto snapshot_future = std::async(std::launch::async, []{});
	auto samples_future = std::async(std::launch::async, []{});
	printf("HOST initialised\n");

	while (!finished)
	{
		simulate<<<1,1>>>(current_solution_D, buf_D); 
		samples_future.wait();
		snapshot_future.wait();
		checkCudaErrors(cudaDeviceSynchronize());
		printf("HOST memcpy from device to host\n");
		checkCudaErrors(cudaMemcpy(snapshot_H, current_solution_D, U_size, cudaMemcpyDeviceToHost));
		copy_device_to_host(buf_H, buf_D, sample_idx);

		printf("HOST t=%f\n", t);

		printf("HOST write samples async\n");
		const int entries = sample_idx;
		samples_future = std::async(std::launch::async, [&]{write(samples_file, buf_H, entries);});

		if (write_snapshot)
		{
			printf("HOST write snapshot async\n");
			const int t_H = t;
			snapshot_future = std::async(std::launch::async, [&]{write(snapshot_H, xsz_H, t_H);});
		}
	}

	samples_future.wait();
	snapshot_future.wait();
	checkCudaErrors(cudaFreeHost(snapshot_H));
	checkCudaErrors(cudaFree(current_solution_D));
	deallocate_D(buf_D);
	deallocate_H(buf_H);
	fclose(samples_file);

    return 0; 
}
