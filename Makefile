dyn: dyn.cu
	nvcc -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_37,code=compute_37 -rdc=true $< -o $@ -lcudadevrt
