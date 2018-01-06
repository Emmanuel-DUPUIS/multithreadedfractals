#include <cstdint>
#include <nvfunctional>
#include <vector_types.h>
#include <cassert>

#include "CudaIntegration.h"
#include "Complex.h"

#if !defined(__CUDA_ARCH__)
#include <cmath>
using namespace std;
#endif

struct BBox
{
  bool   _set;
  double _Xmin;
  double _Xmax;
  double _Ymin;
  double _Ymax;

  __CudaCallable__ explicit BBox() { _set = false;  }

  __CudaCallable__ void addPoint(double x, double y)
  {
    if (_set)
    {
      if (x < _Xmin)
        _Xmin = x;
      else if (x > _Xmax)
        _Xmax = x;
      if (y < _Ymin)
        _Ymin = y;
      else if (y > _Ymax)
        _Ymax = y;
    }
    else
    {
      _Xmin = _Xmax = x;
      _Ymin = _Ymax = y;
      _set = true;
    }
  }

  __CudaCallable__ void reset() { _set = false; }
  __CudaCallable__ double diagonal() { assert(_set); return sqrt((_Ymax - _Ymin)*(_Ymax - _Ymin) + (_Xmax - _Xmin)*(_Xmax - _Xmin)); }
  __CudaCallable__ double area() { assert(_set); return (_Ymax - _Ymin)*(_Xmax - _Xmin); }
};

__CudaCallable__ bool z2Mandelbrot(Complex& ioZ, const Complex& iZ0, double iModulus2Limit, double* iRatio2, double* iRatio);

class ComplexSequence
{
private:
  bool                                      _Compute;
  Complex                                   _Init;
  Complex                                   _DerivateInit;
  /*nvstd::function<bool(Complex&)>                   _EvaluateFct;*/
  nvstd::function<uchar4(const ComplexSequence&)>   _ColorFct;
  uint16_t                                  _NbIterations;
  uint16_t                                  _NbPerformedIterations;
  BBox                                      _SequenceResultsBBox;
  bool                                      _SetBBox;
  double                                    _EstimatedDistanceToSet;
  double                                    _Delta;
  Complex                                   _Attractor;

public:

  __CudaCallable__ explicit ComplexSequence(uint16_t iNbIterations, const Complex& iInit, const Complex& iDerivateInit/*, nvstd::function<bool(Complex&)> iEvaluateFct*/, nvstd::function<uchar4(const ComplexSequence&)> iColorFct)
  : _SequenceResultsBBox()
  {
    _NbIterations = iNbIterations; _Init = iInit; _DerivateInit = iDerivateInit; /*_EvaluateFct = iEvaluateFct;*/ _ColorFct = iColorFct; _Compute = _SetBBox = false;
  }

  __CudaCallable__ ~ComplexSequence() {}

  ComplexSequence(ComplexSequence&) = delete;
  ComplexSequence(ComplexSequence&&) = delete;
  ComplexSequence& operator=(ComplexSequence&) = delete;
  ComplexSequence& operator=(ComplexSequence&&) = delete;

  __CudaCallable__ void reset(const Complex& iInit) { _Init = iInit; if (_SetBBox) _SequenceResultsBBox.reset(); }
  __CudaCallable__ void setBBox(bool iCompute) { _SetBBox = iCompute; }

  __CudaCallable__ bool           isBounded() const { return _NbPerformedIterations == _NbIterations; }
  __CudaCallable__ uint16_t       getIterationNb() const { return _NbPerformedIterations; }
  __CudaCallable__ double         getEstimatedDistanceToSet() const { return _EstimatedDistanceToSet; }
  __CudaCallable__ const Complex& getInit() const { return _Init; }
  __CudaCallable__ const Complex& getAttractor() const { return _Attractor; }

  __CudaCallable__ void compute()
  {
    _Compute = true;

    Complex z  = _Init;
    Complex dZ = _DerivateInit;

    double x = _Init.real();
    double y = _Init.imaginary();
    double dist = 0;

    uint32_t index = 0;
    for (uint32_t i = 1; i < 100; i++)
    {
      double u = x*x - y*y;
      double v = 2 * x*y;
      x = u + _Init.real();
      y = v + _Init.imaginary();
      double radius = x*x + y*y;
      if (radius > 4)
      {
        _NbPerformedIterations = i;
        return;
      }
    }
    _NbPerformedIterations = 100;
    /*for (_NbPerformedIterations = 0; _NbPerformedIterations < _NbIterations; _NbPerformedIterations++)
    {
      if (_SetBBox)
        _SequenceResultsBBox.addPoint(z.real(), z.imaginary());

      Complex sZ = z;
      //if (!_EvaluateFct(z))
      if (!z2Mandelbrot(z, _Init, 4., nullptr, nullptr))
      {
        double zModulus = z.modulus();
        //_EstimatedDistanceToSet = log(log(zModulus) / log(4.0)) / log(2.0);
        dZ = (z*dZ*2.) + Complex(1., .0);
        _EstimatedDistanceToSet = zModulus*log(zModulus) / dZ.modulus();
        break;
      }

      if (_NbPerformedIterations == _NbIterations - 1)
      {
        _Delta     = sZ.distanceTo(z);
        _Attractor = z;
      }
    }*/
  }

  __CudaCallable__ uchar4 getColor() const
  {
    return _ColorFct(*this);
  }
};

__CudaCallable__ bool z2Mandelbrot(Complex& ioZ, const Complex& iZ0, double iModulus2Limit, double* iRatio2, double* iRatio)
{
  double x = ioZ.real();
  double y = ioZ.imaginary();
  double x2 = x*x;
  double y2 = y*y;

  double modulus2 = x2 + y2;
  if (modulus2 > iModulus2Limit)
    return false;
  
  double u  = x2 - y2;
  double v  = 2*x*y;

  double x0 = iZ0.real();
  double y0 = iZ0.imaginary();
  ioZ = { (iRatio2?*iRatio2*u:u) + (iRatio? *iRatio*x0:x0), (iRatio2?*iRatio2*v:v) + (iRatio?*iRatio*y0:y0) };

  return true;
}


//===============================================================================================
//                                            COLORS
//===============================================================================================


__CudaCallable__ uchar4 flatColor(const ComplexSequence& sequence)
{
  uchar4 color;

  if (sequence.isBounded())
  {
    color.x = (unsigned char)(255);
    color.y = (unsigned char)(50);
    color.z = (unsigned char)(180);
  }
  else
  {
    color.x = (unsigned char)(5);
    color.y = (unsigned char)(10);
    color.z = (unsigned char)(127);
  }
  color.w = 0;
  return color;
}

__CudaCallable__ uchar4 color1(const ComplexSequence& sequence)
{
  uchar4 color;

  if (sequence.isBounded())
  {
    Complex init      = sequence.getInit();
    Complex attractor = sequence.getAttractor();

    double dx = abs(init.real() - attractor.real());
    double dy = abs(init.imaginary() - attractor.imaginary());
    double f  = 1. / (1. + 100.*(dx*dx+dy*dy));
    if (f > 1.)
      f = 1.;
    double fC = 1. - f;
    double fM = f*255.0;

    color.x = (unsigned char)(dx*500.*fC + fM);
    color.y = (unsigned char)(dy*500.*fC + fM);
    color.z = (unsigned char)((0.0001 + dx) / (0.0001 + dy)*300.*fC + fM);
  }
  else
  {
    double f = ((sequence.getIterationNb() + 1) % 13 - sequence.getEstimatedDistanceToSet()) / 12.;
    color.x = (unsigned char)(10.0 * f);
    color.y = (unsigned char)(20.0 * f);
    color.z = (unsigned char)(255.0 * f);
  }
  color.w = 0;
  return color;
}

    /*
    double convergenceS = 0.04;
    double convergenceI = 0.0001;

    unsigned char aS, bS, cS;
    if (convergence > convergenceI)
    {
    //double f = bbox.diagonal() / 2.;
    //if (f > 1.)
    //  f = 1.;
    //double fC = 1. - f;
    //double fM = f*255.0;
    //aS = (unsigned char)(50.0*fC + fM);
    //bS = (unsigned char)(180.0*fC + fM);
    //cS = (unsigned char)(80.0*fC + fM);
    double f1 = 1. / (1. + abs(.5 - 1.*(bbox._Xmax - bbox._Xmin)) / convergence);
    if (f1 > 1.)
    f1 = 1.;
    double f1C = 1. - f1;
    double f1M = f1*255.0;
    double f2 = 1. / (1. + abs(.5 - 1.*(bbox._Ymax - bbox._Ymin)) / convergence);
    if (f2 > 1.)
    f2 = 1.;
    double f2C = 1. - f2;
    double f2M = f2*255.0;
    double f3 = 1.0 / (1.0 + 1.*(.0001 + bbox._Ymax - bbox._Ymin)/(.0001 + bbox._Xmax - bbox._Xmin) / convergence);
    if (f3 > 1.)
    f3 = 1.;
    double f3C = 1. - f3;
    double f3M = f3*255.0;
    aS = (unsigned char)(f1M);
    bS = (unsigned char)(f2M);
    cS = (unsigned char)(f3M);
    }
    if (convergence > convergenceS)
    {
    dst[indice].x = aS;
    dst[indice].y = bS;
    dst[indice].z = cS;
    }
    else
    {
    unsigned char aI, bI, cI;
    double f = 1. / (1. + 100.*((cx - ax)*(cx - ax)+ (cy - ay)*(cy - ay)));
    if (f > 1.)
    f = 1.;
    double fC = 1. - f;
    double fM = f*255.0;
    aI = (unsigned char)(abs(cx - ax)*500.*fC + fM);
    bI = (unsigned char)(abs(cy - ay)*500.*fC + fM);
    cI = (unsigned char)((0.0001 + abs(cx - ax)) / (0.0001 + abs(cy - ay))*300.*fC + fM);
    if (convergence < convergenceI)
    {
    dst[indice].x = aI;
    dst[indice].y = bI;
    dst[indice].z = cI;
    }
    else
    {
    double ratio = log(1.+199.*(convergence - convergenceI) / (convergenceS - convergenceI))/log(200.);
    double ratioC = 1. - ratio;
    dst[indice].x = ratio*aS + ratioC*aI;
    dst[indice].y = ratio*bS + ratioC*bI;
    dst[indice].z = ratio*cS + ratioC*cI;
    }
    }*/

    /*double f = 1.0 / (1.0 + 1.*dist);
    double fM = f*255.0;
    dst[imageW * iy + ix].x = (unsigned char)(50.0*(1 - f) + fM);
    dst[imageW * iy + ix].y = (unsigned char)(180.0*(1 - f) + fM);
    dst[imageW * iy + ix].z = (unsigned char)(80.0*(1 - f) + fM);*/
    /*double f = 255.0 / ( 50.0 + 205.0*dist/4.0);
    double c=50.0*f;
    if (c > 255.0)
    dst[imageW * iy + ix].x = 255;
    else
    dst[imageW * iy + ix].x = (unsigned char)c;
    c = 180.0*f;
    if (c > 255.0)
    dst[imageW * iy + ix].y = 255;
    else
    dst[imageW * iy + ix].y = (unsigned char)c;
    c = 80.0*f;
    if (c > 255.0)
    dst[imageW * iy + ix].z = 255;
    else
    dst[imageW * iy + ix].z = (unsigned char)c;*/



/*
// 500 plus précis, notamment pour la reproduction de l'ensemble global à l'identique
#define NB_ITERS 100
__CudaCallable__ uint32_t JuliaCompute(double x, double y, double cx, double cy, double& dist, double& ax, double& ay, double& convergence, BBox* bbox = nullptr, uint16_t* step = nullptr, double* stepX = nullptr, double* stepY = nullptr)
{
  dist = .0;
  ax = cx;
  ay = cy;
	for (uint32_t i = 1; i < NB_ITERS+1; i++)
	{
    double x2 = x*x;
    double y2 = y*y; 
    double u = x2 - y2;
    double v = 2*x*y;
    //double u = 0.75*x * (x2 - 3.*y2) - .95 *( x2 - y2);
		//double v = 0.75*y * (3.*x2 - y2) -.95 *(2*x*y);
    if (step)// && i != 1)
    {
      double coef = *step / 100.;
      x = u + coef * cx;
      y = v + coef * cy;
      x = coef * u + cx;
      y = coef * v + cy;
    }
    else
    {
      x = u + cx;
      y = v + cy;
    }
    if (step && *step == i)
    {
      *stepX = x;
      *stepY = y;
    }

    if (bbox)
    {
      if (bbox->_set)
      {
        if (x < bbox->_Xmin)
          bbox->_Xmin = x;
        else if (x > bbox->_Xmax)
          bbox->_Xmax = x;
        if (y < bbox->_Ymin)
          bbox->_Ymin = y;
        else if (y > bbox->_Ymax)
          bbox->_Ymax = y;
      }
      else
      {
        bbox->_Xmin = bbox->_Xmax = x;
        bbox->_Ymin = bbox->_Ymax = y;
        bbox->_set = true;
      }
    }

		double radius = x2 + y2;
		if (radius > 4)
		{
			dist = log(log(sqrt(radius)) / log(4.0)) / log(2.0);
			return i;
		}

    if (i == NB_ITERS - 2)
    {
      ax = x;
      ay = y;
    }
    if (i == NB_ITERS-1)
      convergence = sqrt((x - ax)*(x - ax) + (y - ay)*(y - ay));
	}
  
  ax = x; ay = y;
	return 0;
}
*/


/*
const int ix = blockDim.x * blockIdx.x + threadIdx.x;
const int iy = blockDim.y * blockIdx.y + threadIdx.y;

if (ix < imageW && iy < imageH)
{
double dist, ax, ay, convergence;
double x = ((double)ix - .5 * (double)imageW) * pixelXYWidth + windowCenterX;
double y = (.5 * (double)imageH - (double)iy) * pixelXYWidth + windowCenterY;

BBox bbox;
double stepX, stepY;
uint32_t iter = JuliaCompute(x, y, cx, cy, dist, ax, ay, convergence, &bbox, &step, &stepX, &stepY);

if (iter == 0)
{
dst[imageW * iy + ix].x = (unsigned char)((int)(ax * 1000) % 255);
dst[imageW * iy + ix].y = (unsigned char)((int)(ay * 1000) % 255);
dst[imageW * iy + ix].z = (unsigned char)((int)(ax*ay * 1000000) % 255);
dst[imageW * iy + ix].w = 0;
}
else
{
uchar4 color;
double f = ((iter + 1) % 12 - dist) / 12.0;
color.x = (unsigned char)(10.0 * f);
color.y = (unsigned char)(20.0 * f);
color.z = (unsigned char)(255.0 * f);

/*if (iter == 1)
{
double f = 1./(1.-6*dist);
color.x = (unsigned char)(10.0 * f);
color.y = (unsigned char)(20.0 * f);
color.z = (unsigned char)(255.0 * f);
}
else
{
double f1 = cos(1*((iter + 1) - dist));
double f2 = sin(1*(((iter + 1) - dist)));
color.x = (unsigned char)(10.0*f1*f1+250.0*f2*f2);
color.y = (unsigned char)(20.0 * f1*f1 + 240.0*f2*f2);
color.z = (unsigned char)(255.0 * (f1*f1 + f2*f2));
}*/
//			color.w = 0;
//			dst[imageW * iy + ix] = color;
//		}

/*double ratio = 1.;
double d;
if ((d=fabs(x)) < ratio*pixelXYWidth || (d = fabs(y)) < ratio*pixelXYWidth || (d=fabs(x*x + y*y - 1)) < ratio*pixelXYWidth)
{
double t = ratio* pixelXYWidth*d;
dst[imageW * iy + ix].x = (unsigned char)(255*(1.0-t)+ dst[imageW * iy + ix].x*t);
dst[imageW * iy + ix].y = (unsigned char)(255 * (1.0 - t) + dst[imageW * iy + ix].y*t);
dst[imageW * iy + ix].z = (unsigned char)(255 * (1.0 - t) + dst[imageW * iy + ix].z*t);
}*/
//	}