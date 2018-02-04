//==============================================================================
//  (c) Emmanuel DUPUIS 2016, emmanuel.dupuis@undecentum.com, MIT Licence
//==============================================================================

#include <cstdint>
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <Windows.h>

void saveAsBmp(const char* iFileName, uint32_t iWidth, uint32_t iHeight, void* iRawData);

#include "../../ExS/ExsMtS/includes/ExSMtS_TaskDispatcher.h"

static uint32_t gIter = 0;

#define FLOATING_TYPE double

void convergency(uint32_t* image, uint32_t iImageWidth, uint32_t iImageHeight, uint32_t iWPixel, uint32_t iHPixel, FLOATING_TYPE iCx, FLOATING_TYPE iCy, uint32_t iNbIterations)
{
  FLOATING_TYPE x = iCx;
  FLOATING_TYPE y = iCy;

  FLOATING_TYPE x2 = x*x;
  FLOATING_TYPE y2 = y*y;

  for (uint_fast16_t n = 1; n < iNbIterations; n++)
  {
    y = (FLOATING_TYPE)2.0*x*y + iCy;
    x = x2-y2 + iCx;
    x2 = x*x;
    y2 = y*y;
    FLOATING_TYPE radius2 = x2 + y2;
        
    if (radius2 > 4.0)
    {
      gIter++;
      FLOATING_TYPE dist = (FLOATING_TYPE)1.4426954 * log(log(radius2) / (FLOATING_TYPE)1.38629436111);
      FLOATING_TYPE f = abs((n + 1) % 12 - dist) / (FLOATING_TYPE)12.0;

      uint32_t color = (uint32_t)(10.0 * f);
      color |= (uint32_t)(20.0 * f) << 8;
      color |= (uint32_t)(255.0 * f) << 16;
      color |= 255 << 24;
      image[iHPixel*iImageWidth + iWPixel] = color;
      return;
    }
  }

  image[iHPixel*iImageWidth + iWPixel] = 0xFF643264; // (100,50,100,255)
}

void ComputeThread(std::chrono::time_point<std::chrono::high_resolution_clock> tStartG, uint32_t affinity, uint32_t* image, uint32_t width, uint32_t height, uint32_t start, uint32_t end, double pixelSize, double cx, double cy, uint32_t nbIterations)
{
  using namespace std::chrono_literals;

  PROCESSOR_NUMBER pn;
  GetCurrentProcessorNumberEx(&pn);
  while (affinity != -1 && pn.Number != affinity)
  {
    std::this_thread::sleep_for(10us);
    GetCurrentProcessorNumberEx(&pn);
  }

  uint32_t w = start % width;
  uint32_t h = start / width;
  uint32_t nb = end - start;
  double hw = 0.5*width;
  double hh = 0.5*height;

  for (uint_fast32_t i = 0; i < nb; i++)
  {
    convergency(image, width, height, w, h, (FLOATING_TYPE)(cx+(w-hw)*pixelSize), (FLOATING_TYPE)(cy+ (hh-h)*pixelSize), nbIterations);
    w++;
    if (w == width)
    {
      w = 0;
      h++;
    }
  }
}

struct SubImage
{
  uint32_t wIndex;
  uint32_t hIndex;
  uint32_t width;
  uint32_t height;
};

void requestToDispatcher(ExS::MtS::TaskDispatcher<SubImage>& dispatcher, uint32_t threadIndex, uint32_t width, uint32_t height, uint32_t sidePixelDim)
{
  uint32_t nbW =(uint32_t)((width+sidePixelDim-1) / sidePixelDim);
  uint32_t nbH = (uint32_t)((height + sidePixelDim - 1) / sidePixelDim);
  uint32_t wLastDim = width - sidePixelDim*(nbW-1);
  uint32_t hLastDim = height - sidePixelDim*(nbH-1);
  for (uint32_t w=0; w < nbW; w++)
    for (uint32_t h = 0; h < nbH; h++)
    {
      uint32_t wDim = sidePixelDim;
      uint32_t hDim = sidePixelDim;
      if (w == nbW-1)
        wDim = wLastDim;
      if (h == nbH-1)
        hDim = hLastDim;

      if (wDim && hDim)
      {
        SubImage image = { (uint32_t)(w*sidePixelDim), (uint32_t)(h*sidePixelDim), wDim, hDim };
        dispatcher.send(image);
      }
    }
}

void respondToDispatcher(SubImage& subImage, uint32_t threadIndex, std::chrono::time_point<std::chrono::high_resolution_clock> tStartG,uint32_t* image, uint32_t width, uint32_t height, double pixelSize, double cx, double cy, uint32_t nbIterations)
{
  double hw = 0.5*width;
  double hh = 0.5*height;
  for (uint_fast16_t i = 0; i < subImage.width; i++)
      for (uint_fast16_t j = 0; j < subImage.height; j++)
      {
        convergency(image, width, height, subImage.wIndex + i, subImage.hIndex + j, (FLOATING_TYPE)(cx + (subImage.wIndex + i - hw)*pixelSize), (FLOATING_TYPE)(cy + (hh - subImage.hIndex - j)*pixelSize), nbIterations);
      }
}


void ComputeOnCpu(std::string& mode, uint32_t* image, uint32_t width, uint32_t height, double pixelSize, double cx, double cy, uint32_t nbIterations, uint32_t sidePixelDim)
{
  if (mode == "sequential")
  {
    double hw = 0.5*width;
    double hh = 0.5*height;
    for (uint_fast16_t w = 0; w < width; w++)
      for (uint_fast16_t h = 0; h < height; h++)
        convergency(image, width, height, w, h, (float)(cx + (w - hw)*pixelSize), (float)(cy + (hh - h)*pixelSize), nbIterations);
  }
  else if (mode == "multiFix")
  {
    uint32_t nbCores = std::thread::hardware_concurrency();

    std::thread** threads = new std::thread*[nbCores];

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    uint32_t totalSize = width*height;
    uint32_t packetSize = (uint32_t)(totalSize / nbCores);
    for (uint32_t t = 0; t < nbCores; t++)
    {
      // No affinity, leaves the OS assigning 
      if (t != std::thread::hardware_concurrency() - 1)
        threads[t] = new std::thread(ComputeThread, start, -1/*t*/, image, width, height, t*packetSize, (t + 1)*packetSize, pixelSize, cx, cy, nbIterations);
      else
        threads[t] = new std::thread(ComputeThread, start, -1/*t*/, image, width, height, t*packetSize, totalSize, pixelSize, cx, cy, nbIterations);
      auto handle = threads[t]->native_handle();
      //SetThreadAffinityMask(handle, (DWORD_PTR)1 << t);
    }

    for (uint32_t t = 0; t < nbCores; t++)
      threads[t]->join();
  }
  else if (mode == "multiDispatcher1")
  {
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    
    ExS::MtS::TaskDispatcher<SubImage> dispatcher(4096);

    dispatcher.setCustomers(1, std::bind(requestToDispatcher, std::placeholders::_1, std::placeholders::_2, width, height, sidePixelDim));
    // No affinity, leaves the OS assigning 
    //std::vector<int16_t> cores = { 0 };
    dispatcher.setSuppliers(8, std::bind(respondToDispatcher, std::placeholders::_1, std::placeholders::_2, start, image, width, height, pixelSize, cx, cy, nbIterations)/*, &cores*/);

    dispatcher.launch(true, true);
  }
  else if (mode == "multiDispatcher2")
  {
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    ExS::MtS::TaskDispatcher<SubImage> dispatcher(4096);

    dispatcher.setCustomers(1, std::bind(requestToDispatcher, std::placeholders::_1, std::placeholders::_2, width, height, sidePixelDim));
    dispatcher.setSuppliers(4, std::bind(respondToDispatcher, std::placeholders::_1, std::placeholders::_2, start, image, width, height, pixelSize, cx, cy, nbIterations));

    dispatcher.launch(true, true);
  }
}


int main()
{
  uint32_t w = 800; // 790;
  uint32_t h = 600; // 590;
  double pixelSize = 0.00405063294; // 1.0 / 800;
  double cx = -0.5;// -1.5;
  double cy = .0;

  uint32_t* image = new uint32_t[w*h];
 
  /*for (uint32_t sidePixelDim : {200}) //, 100, 50, 40, 25, 20, 10, 8, 5, 4, 2, 1})
  {
    using namespace std::chrono_literals;
    auto tStart = std::chrono::high_resolution_clock::now();

    constexpr uint32_t nbIters = 250;
    for (uint32_t iter = 0; iter < nbIters; iter++)
    {
      ComputeOnCpu(std::string("sequential"), image, w, h, pixelSize, cx, cy, 200, sidePixelDim);
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = tEnd - tStart;
    std::stringstream msg;
    msg << sidePixelDim << "=" << (1000 * diff.count() / nbIters) << "ms\n";
    std::cout << msg.str();
  }*/

  for (int y = 0; y < 5; y++)
  {
    for (auto mode : { "sequential", "multiFix", "multiDispatcher1" })
    {
      using namespace std::chrono_literals;
      auto tStart = std::chrono::high_resolution_clock::now();

      constexpr uint32_t nbIters = 250;
      for (uint32_t iter = 0; iter < nbIters; iter++)
      {
        ComputeOnCpu(std::string(mode), image, w, h, pixelSize, cx, cy, 200, 20);
      }

      auto tEnd = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = tEnd - tStart;
      std::stringstream msg;
      msg << mode << "=" << (1000 * diff.count() / nbIters) << "ms\n";
      std::cout << msg.str();
    }
  }

  saveAsBmp(std::string("mandelbrot_cpu.bmp").c_str(), w, h, image);
}

void saveAsBmp(const char* iFileName, uint32_t iWidth, uint32_t iHeight, void* iRawData)
{
  FILE* fpW = nullptr;
  if (!fopen_s(&fpW, iFileName, "wb"))
  {
    uint32_t zero = 0, size1 = 14 + 124 + 4 * iWidth*iHeight, offset = 14 + 124, size2 = 124;

    // BMP header
    fwrite((void*)"BM", 1, 2, fpW);
    fwrite((unsigned char*)&size1, 4, 1, fpW);
    fwrite((void*)&zero, 4, 1, fpW);
    fwrite((void*)&offset, sizeof(offset), 1, fpW);
    
    fwrite((void*)&size2, sizeof(size2), 1, fpW);
    fwrite((void*)&iWidth, sizeof(iWidth), 1, fpW);
    fwrite((void*)&iHeight, sizeof(iHeight), 1, fpW);
    // Color plane
    uint16_t t16 = 1;
    fwrite((void*)&t16, sizeof(t16), 1, fpW);
    // Number of bits per pixel
    t16 = 32;
    fwrite((void*)&t16, sizeof(t16), 1, fpW);
    // Compression mode BI_BITFIELDS
    uint32_t t32 = 3;
    fwrite((void*)&t32, sizeof(t32), 1, fpW);
    // Size of raw data
    t32 = 4 * iWidth*iHeight;
    fwrite((void*)&t32, sizeof(t32), 1, fpW);
    // Print resolution
    t32 = 2835;
    fwrite((void*)&t32, sizeof(t32), 1, fpW);
    fwrite((void*)&t32, sizeof(t32), 1, fpW);
    // Palette
    fwrite((void*)&zero, 4, 1, fpW);
    fwrite((void*)&zero, 4, 1, fpW);
    // Masks
    t32 = 0x000000FF;
    fwrite((void*)&t32, sizeof(t32), 1, fpW);
    t32 = 0x0000FF00;
    fwrite((void*)&t32, sizeof(t32), 1, fpW);
    t32 = 0x00FF0000;
    fwrite((void*)&t32, sizeof(t32), 1, fpW);
    t32 = 0xFF000000;
    fwrite((void*)&t32, sizeof(t32), 1, fpW);
    t32 = 0x73524742;
    fwrite((void*)&t32, sizeof(t32), 1, fpW);
    for (uint16_t n = 0; n<48 / 4; n++)
      fwrite((void*)&zero, 4, 1, fpW);
    t32 = 0X02;
    fwrite((void*)&t32, sizeof(t32), 1, fpW);
    for (uint16_t n = 0; n<12 / 4; n++)
      fwrite((void*)&zero, 4, 1, fpW);

    fwrite((void*)iRawData, sizeof(uint32_t), iWidth*iHeight, fpW);
    fclose(fpW);
  }
}