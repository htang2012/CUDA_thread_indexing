////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// This sample illustrates the usage of CUDA events for both GPU timing and
// overlapping CPU and GPU execution.  Events are inserted into a stream
// of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
// perform computations while GPU is executing (including DMA memcopies
// between the host and device).  CPU can query CUDA events to determine
// whether GPU has completed tasks.
//

// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>

__global__ void unique_idx_1d_grid_1d_block(int *input)
{
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	printf("gid=%d, value=%d\n",gid, input[gid]); 

}

__global__  void  unique_idx_1d_grid_2d_block(int *input)
{
      int gid = threadIdx.x + blockIdx.x * blockDim.x * blockDim.y
	         + blockDim.x *threadIdx.y;
      
      printf("gid=%d, value=%d\n",gid, input[gid]); 

}

__global__  void  unique_idx_1d_grid_3d_block(int *input)
{
      int gid = threadIdx.x + blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
	        threadIdx.z * blockDim.y * blockDim.x + 
		threadIdx.y * blockDim.x; 
      
      printf("gid=%d, value=%d\n",gid, input[gid]); 

}



__global__  void  unique_idx_2d_grid_1d_block(int *input)
{
      int blockId = blockIdx.y * gridDim.x + blockIdx.x;
      int gid = blockId  * blockDim.x + threadIdx.x;
      
      printf("gid=%d, value=%d\n",gid, input[gid]); 
}


__global__  void  unique_idx_2d_grid_2d_block(int *input)
{
      int blockId = blockIdx.x  + blockIdx.y * gridDim.x;

      int gid = blockId  * (blockDim.x * blockDim.y) 
	        + threadIdx.y * blockDim.x + threadIdx.x;
      
      printf("gid=%d, value=%d\n",gid, input[gid]); 
}


__global__  void  unique_idx_2d_grid_3d_block(int *input)
{
	int blockId = blockIdx.x  + blockIdx.y * gridDim.x;
	int gid = blockId *( blockDim.x * blockDim.y * blockDim.z)
		 + threadIdx.z * (blockDim.x * blockDim.y)
		 + threadIdx.y * blockDim.x  + threadIdx.x;
        printf("gid=%d, value=%d\n",gid, input[gid]);
}




int main(void)
{
    int array_size = 256;
    int array_byte_size = sizeof(int) * array_size;
    int *h_data = (int *) malloc(array_byte_size);
    for (int i = 0; i< array_size; i++)
    {   
	 h_data[i] = (i+1) * 10;
         printf("%d ", h_data[i]);
    }
    printf("\n\n");
    int *d_data;
    cudaMalloc((void **)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);
   
#if 1 
    printf("\n1D grid of 1D block:\n");
    dim3 block11(4);
    dim3 grid11(2);
    unique_idx_1d_grid_1d_block<<<grid11,block11 >>>(d_data);
    printf("\n\n");
    cudaDeviceSynchronize();
#endif

#if 1
    printf("\n1D grid of 2D block:\n");
    dim3 block12(4,2);
    dim3 grid12(2);
    unique_idx_1d_grid_2d_block<<<grid12,block12>>>(d_data);
    printf("\n\n");
    cudaDeviceSynchronize();
#endif

#if 1
    printf("\n1D grid of 3D block:\n");
    dim3 block13(8,4,2);
    dim3 grid13(2);
    unique_idx_1d_grid_3d_block<<<grid13,block13>>>(d_data);
    printf("\n\n");
    cudaDeviceSynchronize();
#endif

#if 1
    printf("\n2D grid of 1D block:\n");
    dim3 block21(4);
    dim3 grid21(4,4);
    unique_idx_2d_grid_1d_block<<<grid21,block21>>>(d_data);
    printf("\n\n");
    cudaDeviceSynchronize();
#endif


 #if 1
    printf("\n2D grid of 2D block:\n");
    dim3 block22(4,4);
    dim3 grid22(4,4);
    unique_idx_2d_grid_2d_block<<<grid22,block22>>>(d_data);
    printf("\n\n");
    cudaDeviceSynchronize();
#endif

   
 #if 1
    printf("\n2D grid of 3D block:\n");
    dim3 block23(2,2,2);
    dim3 grid23(4,4);
    unique_idx_2d_grid_3d_block<<<grid23,block23>>>(d_data);
    printf("\n\n");
    cudaDeviceSynchronize();
#endif



    cudaDeviceReset();
    cudaFree(d_data);
    free(h_data); 
    return 0;
}
