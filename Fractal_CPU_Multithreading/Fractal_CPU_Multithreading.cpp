#include <cstdint>
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <Windows.h>

#include "../../ExS/ExsMtS/includes/ExSMtS_TaskDispatcher.h"

#define __CompileProfiler
#include "../../ExS/ExsSys/includes/ExSSys_Profiler.h"

struct ConsumerOpProfiling
{
  BYTE _Core;
  std::chrono::time_point<std::chrono::high_resolution_clock> _StartTime;
  std::chrono::time_point<std::chrono::high_resolution_clock> _EndTime;
  uint32_t _Size;
};

struct ConsumerProfiling
{
  BYTE _CoreAtLaunch;
  std::chrono::time_point<std::chrono::high_resolution_clock> _StartTime;
  std::chrono::time_point<std::chrono::high_resolution_clock> _EndTime;
  ConsumerOpProfiling _Operations[256];
  uint16_t _NbOperations;
};

ConsumerProfiling gProfiles[8];

void convergency(uint32_t* image, uint16_t iImageWidth, uint16_t iImageHeight, uint16_t iWPixel, uint16_t iHPixel, double iXCenter, double iYCenter, double iPixelWidth, uint16_t nbIterations)
{
  double cx = iXCenter + (iWPixel - .5 * iImageWidth) * iPixelWidth;
  double cy = iYCenter + (-iHPixel + .5 * iImageHeight) * iPixelWidth;

  double x = cx;
  double y = cy;

  double x2 = x*x;
  double y2 = y*y;

  for (uint_fast16_t n = 1; n < nbIterations; n++)
  {
    y = 2.*x*y + cy;
    x = x2 - y2 + cx;
    x2 = x*x;
    y2 = y*y;
    double radius2 = x2 + y2;

    if (radius2 > 4.)
    {
      double dist = 1.4426954 * log(log(radius2) / 1.38629436111);

      double f = ((n + 1) % 12 - dist) / 12.0;
      uint32_t color = (uint32_t)(10.0 * f);
      color |= (uint32_t)(20.0 * f) << 8;
      color |= (uint32_t)(255.0 * f) << 16;
      color |= 255 << 24;
      image[iHPixel*iImageWidth + iWPixel] = color;
      return;
    }
  }

  uint32_t color = 100;
  color |= 50 << 8;
  color |= 100 << 16;
  color |= 255 << 24;
  image[iHPixel*iImageWidth + iWPixel] = color;
}

void ComputeThread(std::chrono::time_point<std::chrono::high_resolution_clock> tStartG, uint16_t affinity, uint32_t* image, uint16_t width, uint16_t height, uint32_t start, uint32_t end, double pixelSize, double cx, double cy, uint16_t nbIterations)
{
  using namespace std::chrono_literals;
  auto tStart = std::chrono::high_resolution_clock::now();

  PROCESSOR_NUMBER pn;
  GetCurrentProcessorNumberEx(&pn);
  while (pn.Number != affinity)
  {
    std::this_thread::sleep_for(10us);
    GetCurrentProcessorNumberEx(&pn);
  }

  uint16_t w = start % width;
  uint16_t h = start / width;
  uint32_t nb = end - start;
  for (uint_fast32_t i = 0; i < nb; i++)
  {
    convergency(image, width, height, w, h, cx, cy, pixelSize, nbIterations);
    w++;
    if (w == width)
    {
      w = 0;
      h++;
    }
  }

  auto tEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff1 = tStart - tStartG;
  std::chrono::duration<double> diff2 = tEnd - tStart;
  std::stringstream msg;
  msg << "Time on core " << (int)pn.Number << " : start=+" << diff1.count()*1000 << "ms, duration=" << diff2.count()*1000 << " ms\n";
  std::cout << msg.str();
}

struct SubImage
{
  uint16_t wIndex;
  uint16_t hIndex;
  uint16_t width;
  uint16_t height;
};

void fillQueue(ExS::MtS::TaskDispatcher<SubImage>& queue, uint16_t threadIndex, uint16_t width, uint16_t height)
{
  uint16_t dim = 64;
  uint16_t nbW =(uint16_t)(width / dim);
  uint16_t nbH = (uint16_t)(height / dim);
  uint16_t wLastDim = (uint16_t)(width % dim);
  uint16_t hLastDim = (uint16_t)(height % dim);
  for (uint16_t w=0; w<=nbW; w++)
    for (uint16_t h = 0; h <= nbH; h++)
    {
      uint16_t wDim = dim;
      uint16_t hDim = dim;
      if (w == nbW)
        wDim = wLastDim;
      if (h == nbH)
        hDim = hLastDim;

      if (wDim && hDim)
      {
        SubImage image = { (uint16_t)(w*dim), (uint16_t)(h*dim), wDim, hDim };
        queue.send(image);
      }
    }
}

void consumeQueue(SubImage& subImage, uint16_t threadIndex, std::chrono::time_point<std::chrono::high_resolution_clock> tStartG,uint32_t* image, uint16_t width, uint16_t height, double pixelSize, double cx, double cy, uint16_t nbIterations)
{
    for (uint_fast16_t i = 0; i < subImage.width; i++)
      for (uint_fast16_t j = 0; j < subImage.height; j++)
        convergency(image, width, height, subImage.wIndex+i, subImage.hIndex+j, cx, cy, pixelSize, nbIterations);
}


void ComputeOnCpu(std::string& mode, uint32_t* image, uint16_t width, uint16_t height, double pixelSize, double cx, double cy, uint16_t nbIterations)
{
  if (mode == "single")
  {
    for (uint_fast16_t i = 0; i < width; i++)
      for (uint_fast16_t j = 0; j < height; j++)
        convergency(image, width, height, i, j, cx, cy, pixelSize, nbIterations);
  }
  else if (mode == "multiFix")
  {
    uint16_t nbCores = std::thread::hardware_concurrency();

    std::thread** threads = new std::thread*[nbCores];

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    uint32_t totalSize = width*height;
    uint32_t packetSize = (uint32_t)(totalSize / nbCores);
    for (uint16_t t = 0; t < nbCores; t++)
    {
      if (t != std::thread::hardware_concurrency() - 1)
        threads[t] = new std::thread(ComputeThread, start, t, image, width, height, t*packetSize, (t + 1)*packetSize, pixelSize, cx, cy, nbIterations);
      else
        threads[t] = new std::thread(ComputeThread, start, t, image, width, height, t*packetSize, totalSize, pixelSize, cx, cy, nbIterations);
      auto handle = threads[t]->native_handle();
      SetThreadAffinityMask(handle, (DWORD_PTR)1 << t);
    }

    for (uint16_t t = 0; t < nbCores; t++)
      threads[t]->join();
  }
  else if (mode == "multiQueue")
  {
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    
    ExS::MtS::TaskDispatcher<SubImage> queue(4096);

    queue.setProducers(1, std::bind(fillQueue, std::placeholders::_1, std::placeholders::_2, width, height));
    queue.setConsumers(8, std::bind(consumeQueue, std::placeholders::_1, std::placeholders::_2, start, image, width, height, pixelSize, cx, cy, nbIterations), {});

    queue.launch(true, true);
  }
}


int main()
{
  uint16_t w = 1800;
  uint16_t h = 1400;

  double pixelSize = 3.0 / 800;
  double cx = -2.;
  double cy = .0;

  uint32_t* image = new uint32_t[w*h];

  std::string mode = "multiQueue"; // single multiFix
  
 __Profiler(start());

  ComputeOnCpu(mode, image, w, h, pixelSize, cx, cy, 500);

  __Profiler(end());
  __Profiler(displayReport());
}

