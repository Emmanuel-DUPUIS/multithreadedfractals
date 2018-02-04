#include <stdio.h>
#include <vector_types.h>

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "HostComputation.h"

#define FLOATING_TYPE float

__global__ void cuComputeAndFillImageForBlock(uchar4 *dst, uint16_t imageW, uint16_t imageH, float pixelXYWidth, float deltaX, float deltaY, uint16_t nbIterations)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if(ix < imageW && iy < imageH)
  {
    FLOATING_TYPE cx = deltaX + ix * pixelXYWidth;
    FLOATING_TYPE cy = deltaY - iy * pixelXYWidth ;
    
    FLOATING_TYPE x = cx;
    FLOATING_TYPE y = cy;
    FLOATING_TYPE x2 = x*x, y2 = y*y;

    for (uint_fast16_t i = 1; i < nbIterations; i++)
    {
      y  = 2*x*y+cy;
      x  = x2-y2+cx;

      x2 = x*x;
      y2 = y*y;

      if (x2+y2 > 4)
      {
        FLOATING_TYPE dist = 1.4426954 * log(log(x*x + y*y) / 1.38629436111);
        FLOATING_TYPE f = ((i + 1) % 12 - dist) / 12.0;

        dst[imageW * iy + ix].x = (unsigned char)(10.0 * f);
        dst[imageW * iy + ix].y = (unsigned char)(20.0 * f);
        dst[imageW * iy + ix].z = (unsigned char)(255.0 * f);
        dst[imageW * iy + ix].w = 255;
        return;
      }
    }
   
    *(reinterpret_cast<uint32_t*>(&dst[imageW * iy + ix])) = 0xFF643264; // (100,50,100,255)
  }
}

__global__ void cuComputeConvergenceForBlock(unsigned char *dst, uint16_t imageW, uint16_t imageH, float pixelXYWidth, float deltaX, float deltaY, uint16_t nbIterations)
{
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if (ix < imageW && iy < imageH)
  {
    float cx = deltaX + ix * pixelXYWidth;
    float cy = deltaY - iy * pixelXYWidth;

    float x = cx;
    float y = cy;
    float x2 = x*x, y2 = y*y;

    for (uint_fast16_t i = 1; i < nbIterations; i++)
    {
      y = 2 * x*y + cy;
      x = x2 - y2 + cx;

      x2 = x*x;
      y2 = y*y;

      if (x2 + y2 > 4)
      {
        dst[imageW * iy + ix] = i;
        return;
      }
    }

    dst[imageW * iy + ix] = nbIterations;
  }
}

__global__ void cuComputeAndFillImageForBlock2(uchar4 *dst, uint16_t imageW, uint16_t imageH, float pixelXYWidth, float deltaX, float deltaY, uint16_t nbIterations, uint32_t numBlocks, uint32_t gridWidth)
{
  for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks; blockIndex += gridDim.x)
  {
    unsigned int blockX = blockIndex % gridWidth;
    unsigned int blockY = blockIndex / gridWidth;

    // process this block
    const int ix = blockDim.x * blockX + threadIdx.x;
    const int iy = blockDim.y * blockY + threadIdx.y;

    if (ix < imageW && iy < imageH);
    {
      float cx = deltaX + ix * pixelXYWidth;
      float cy = deltaY - iy * pixelXYWidth;

      float x = cx;
      float y = cy;
      float x2 = x*x, y2 = y*y;

      uint_fast16_t i = 1;
      for (; i < nbIterations; i++)
      {
        y = 2*x*y + cy;
        x = x2 - y2 + cx;

        x2 = x*x;
        y2 = y*y;

        if (x2 + y2 > 4)
        {
          float dist = 1.4426954 * log(log(x*x + y*y) / 1.38629436111);
          float f = ((i + 1) % 12 - dist) / 12.0;
    
          dst[imageW * iy + ix].x = (unsigned char)(10.0 * f);
          dst[imageW * iy + ix].y = (unsigned char)(20.0 * f);
          dst[imageW * iy + ix].z = (unsigned char)(255.0 * f);
          dst[imageW * iy + ix].w = 255;
          break;
        }
      }

      if (i == nbIterations)
        *(reinterpret_cast<uint32_t*>(&dst[imageW * iy + ix])) = 0xFF643264; // (100,50,100,255)
    }
  }
}


void cuComputeAndFillImageForGrid(uchar4 *dst, const uint16_t imageW, const uint16_t imageH, float pixelXYWidth, float windowCenterX, float windowCenterY, uint16_t nbIterations, float& oElapsedTime)
{
  // Creates events for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start, NULL));

  float deltaX = windowCenterX - .5 * (float)imageW * pixelXYWidth;
  float deltaY = windowCenterY + .5 * (float)imageH * pixelXYWidth;

  uint32_t mode = 2;
  if (mode == 1)
  {
    dim3 threadsPerBlock(8, 8);
    dim3 nbBlocks((unsigned int)((float)(imageW+threadsPerBlock.x-1) / threadsPerBlock.x), (unsigned int)((float)(imageH+threadsPerBlock.y-1) / threadsPerBlock.y));
    cuComputeAndFillImageForBlock<<<nbBlocks,threadsPerBlock>>>(dst, imageW, imageH, pixelXYWidth, deltaX, deltaY, nbIterations);
  }
  else if (mode == 2)
  {
    dim3 threads(16, 16);
    dim3 grid((unsigned int)((float)(imageW+ threads.x-1) / threads.x), (unsigned int)((float)(imageH+threads.y-1) / threads.y));
    cuComputeAndFillImageForBlock2<<<5,threads>>>(dst, imageW, imageH, pixelXYWidth, deltaX, deltaY, nbIterations, grid.x *grid.y, grid.x);
  }
  else
  {
    dim3 threadsPerBlock(8, 8);
    dim3 nbBlocks((imageW) / threadsPerBlock.x, (imageH) / threadsPerBlock.y);
    cuComputeConvergenceForBlock<<<nbBlocks, threadsPerBlock>>>(reinterpret_cast<unsigned char*>(dst), imageW, imageH, pixelXYWidth, deltaX, deltaY, nbIterations);
  }

  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));

  checkCudaErrors(cudaEventElapsedTime(&oElapsedTime, start, stop));
}


