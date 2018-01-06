#ifndef __JuliaCuda_h__
#define __JuliaCuda_h__

#include <vector_types.h>
#include "Complex.h"

extern "C" void cuComputeAndFillImageForGrid(uchar4 *dst, uint16_t imageW, uint16_t imageH, double pixelXYWidth, double windowCenterX, double windowCenterY, uint16_t nbIterations, float& oElapsedTime);

__CudaCallable__ uint16_t MandelbrotGPU(uint16_t w, uint16_t h, uint16_t imageW, uint16_t imageH, double pixelXYWidth, double windowCenterX, double windowCenterY, uint16_t nbIterations, double& dist, double& ax, double& ay)
{
  dist = ax = ay = .0;

  double cx = windowCenterX + (w - .5 * imageW) * pixelXYWidth;
  double cy = windowCenterY + (-h + .5 * imageH) * pixelXYWidth;

  double x = cx;
  double y = cy;

  uint16_t index = 0;
  for (uint16_t i = 1; i < 100; i++)
  {
    double u = x*x - y*y;
    double v = 2 * x*y;
    x = u + cx;
    y = v + cy;
    double radius2 = x*x + y*y;
    if (radius2 > 4.)
    {
      dist = 1.4426954 * log(log(radius2) / 1.38629436111);
      return i;
    }
  }
  ax = x; ay = y;
  return index;
}

#endif
