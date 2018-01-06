#include <stdio.h>
#include <vector_types.h>

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "HostComputation.h"
#include "ComplexSequence.h"

__global__ void cuComputeAndFillImageForBlock(uchar4 *dst, uint16_t imageW, uint16_t imageH, double pixelXYWidth, double deltaX, double deltaY, uint16_t nbIterations)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

  if(ix < imageW && iy < imageH);
  {
    double cx = deltaX + ix * pixelXYWidth;
    double cy = deltaY - iy * pixelXYWidth ;
    
    double x = cx;
    double y = cy;
    
    for (uint_fast16_t i = 1; i < nbIterations; i++)
    {
      double u = y*y;
      y = 2.*x*y + cy;
      x = x*x - u + cx;
      //x2 = x*x;
      //y2 = y*y;
      //double radius2 = x2 + y2;
      
      if (x*x + y*y > 4.)
      {
        //dist = log(log(radius) / log(4.)) / log(2.);
        double dist = 1.4426954 * log(log(x*x + y*y) / 1.38629436111);
        //dist = (log(log(radius)) - log(2.)) / log(2.0);
        //dx = 2.*dx*x + 1;
        //dy = 2.*dy*y;
        //double r = sqrt(radius);
        //dist = r*log(r) / sqrt(dx*dx+dy*dy);

        double f = ((i + 1) % 12 - dist) / 12.0;
        //double rest = remquo((double)i+1, 12., int* quo)
        dst[imageW * iy + ix].x = (unsigned char)(10.0 * f);
        dst[imageW * iy + ix].y = (unsigned char)(20.0 * f);
        dst[imageW * iy + ix].z = (unsigned char)(255.0 * f);
        dst[imageW * iy + ix].w = 255;
        return;
      }
    }
   
    dst[imageW * iy + ix].x = 100;
    dst[imageW * iy + ix].y = 50;
    dst[imageW * iy + ix].z = 100;
    dst[imageW * iy + ix].z = 255;
    /**/
  }
}

void cuComputeAndFillImageForGrid(uchar4 *dst, const uint16_t imageW, const uint16_t imageH, double pixelXYWidth, double windowCenterX, double windowCenterY, uint16_t nbIterations, float& oElapsedTime)
{
  // Creates events for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start, NULL));

  double deltaX = windowCenterX - .5 * (double)imageW * pixelXYWidth;
  double deltaY = windowCenterY + .5 * (double)imageH * pixelXYWidth;

  dim3 threadsPerBlock(8, 8);
	dim3 nbBlocks((imageW)/ threadsPerBlock.x, (imageH)/ threadsPerBlock.y);
  cuComputeAndFillImageForBlock<<<nbBlocks, threadsPerBlock>>>(dst, imageW, imageH, pixelXYWidth, deltaX, deltaY, nbIterations);

  /*cudaDeviceSynchronize();

  exit(1);*/

  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));

  checkCudaErrors(cudaEventElapsedTime(&oElapsedTime, start, stop));

	//getLastCudaError("Kernel execution failed.\n");
}

