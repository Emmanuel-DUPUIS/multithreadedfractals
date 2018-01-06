#include <set>
#include <map>
#include <ctime>

#include <thread>
#include <chrono>

#include <helper_gl.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>


// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "HostComputation.h"

#define FRAME_PERIOD   10   // milliseconds
#define NB_ZOOM_STEPS  200

#define MODE_JULIA       0
#define MODE_MANDELBROT  1

struct PlanePoint
{
  double _x;
  double _y;
};

struct ScreenPosition
{
  uint16_t _w;
  uint16_t _h;
};

struct ViewPosition
{
  double _x;
  double _y;
};

class View2D
{
private:
  double       _PixelPlaneSize;
  uint16_t     _Width;
  uint16_t     _Height;
  PlanePoint   _Center;
  ViewPosition _ViewPosMin;
  ViewPosition _ViewPosMax;
  uint16_t     _NbIterations;


public:
  View2D(uint16_t iWidth, uint16_t iHeight, double iPlaneSizePixel, const PlanePoint& iCenter)
  {
    _Width = iWidth; _Height = iHeight; _PixelPlaneSize = iPlaneSizePixel; _Center = iCenter;
    _ViewPosMin = ViewPosition{ .0, .0 }; _ViewPosMax = ViewPosition{ 1., 1. };
  }

  uint16_t   Width() const { return _Width; }
  uint16_t   Height() const { return _Height; }
  double     PlaneSizePixel() const { return _PixelPlaneSize; }
  PlanePoint Center() const { return _Center; }
  uint16_t   GetNbIterations() const { return _NbIterations; }

  void translate(int16_t x, int16_t y) { _Center._x -= x * _PixelPlaneSize; _Center._y += y * _PixelPlaneSize; }
  void scale(double ratio) { _PixelPlaneSize *= ratio; }

  void setWidth(uint16_t w) { _Width = w; }
  void setHeight(uint16_t h) { _Height = h; }
  void SetNbIterations(uint16_t nbIter) { _NbIterations = nbIter; }

  void getPlanePoint(const ScreenPosition& iScreenPosition, PlanePoint& oPlanePoint)
  {
    oPlanePoint._x = _Center._x + ((double)iScreenPosition._w - .5 * (double)_Width) * _PixelPlaneSize;
    oPlanePoint._y = _Center._y + (.5 * (double)_Height - (double)iScreenPosition._h) * _PixelPlaneSize;
  }

  void getScreenPosition(PlanePoint& iPlanePoint, ScreenPosition& oScreenPosition)
  {
    oScreenPosition._w = (uint16_t)(.5 * (double)_Width + (iPlanePoint._x - _Center._x) / _PixelPlaneSize);
    oScreenPosition._h = (uint16_t)(.5 * (double)_Height - (iPlanePoint._y - _Center._y) / _PixelPlaneSize);
  }

  void geViewPosition(const ScreenPosition& iScreenPosition, ViewPosition& oViewPosition)
  {
    oViewPosition._x = _ViewPosMin._x + _ViewPosMax._x * ((double)iScreenPosition._w / (double)_Width);
    oViewPosition._y = _ViewPosMin._y + _ViewPosMax._y * (1. - (double)iScreenPosition._h / (double)_Height);
  }

  void geViewPosition(const PlanePoint& iPlanePoint, ViewPosition& oViewPosition)
  {
    oViewPosition._x = _ViewPosMin._x + _ViewPosMax._x * (.5 + (iPlanePoint._x - _Center._x) / _PixelPlaneSize / (double)_Width);
    oViewPosition._y = _ViewPosMin._y + _ViewPosMax._y * (1. - (.5 - (iPlanePoint._y - _Center._y) / _PixelPlaneSize / (double)_Height));
  }
};

class Viewer
{
public:
	Complex mJuliaParam;
	uchar4 *mJuliaGLSetData;
	GLuint mJuliaSetPixelBuffer;
	GLuint mJuliaSetTexture;
	struct cudaGraphicsResource *mJCudaJuliaSetResource; // handles OpenGL-CUDA exchange
	uchar4 *mJuliaCudaSetData;

  ScreenPosition _Pointer;
  View2D         _JuliaView;
  View2D         _MandelbrotView;

  std::map<std::string, bool> _Options;
float _ElapsedTime;
	double mZoomPixelXYWidth;
	double mZoomWindowCenterX, mZoomWindowCenterY;
	
	unsigned int  mMode, mZoom;
	std::set<int> keys;
  timespec tempo1{ 0, 0 };

public:
  Viewer() :mJuliaParam(0, 0),
    _Pointer{0,0},
    _JuliaView(800, 600, 4.0 / 800, PlanePoint{ .0,.0 }),
    _MandelbrotView(1600, 1200, .50 / 800, PlanePoint{ -1.,.0 }) 
  {
    _Options = { { "textInfo", true} };

    mJuliaGLSetData = mJuliaCudaSetData = NULL; mJCudaJuliaSetResource = NULL;
    mJuliaSetPixelBuffer = mJuliaSetTexture = 0;
	  mMode = MODE_MANDELBROT; mZoom = NB_ZOOM_STEPS;
	}

  bool getOption(const std::string& iOptionName)
  {
    return _Options[iOptionName];
  }

  void toggleOption(const std::string& iOptionName)
  {
    _Options[iOptionName] = !_Options[iOptionName];
  }

  View2D& getActiveView()
  {
    if (mMode == MODE_JULIA)
      return _JuliaView;
    return _MandelbrotView;
  }

  const ScreenPosition& getPointerPosition() const { return _Pointer; }

  void getScreenDimensions(uint16_t& oWidth, uint16_t& oHeight)
  {
    oWidth  = getActiveView().Width();
    oHeight = getActiveView().Height();
  }
  
	void setWidth(int w) { _JuliaView.setWidth(w); _MandelbrotView.setWidth(w); }
	void setHeight(int h) { _JuliaView.setHeight(h); _MandelbrotView.setHeight(h); }

 
	void setPointerPosition(uint16_t w, uint16_t h)
	{
    _Pointer = { w,h };
	}
};

Viewer *theViewer = NULL;

bool leftClicked = false;
bool middleClicked = false;
bool rightClicked = false;

void initOpenGLBuffers()
{
	// delete old buffers
	if (theViewer->mJuliaGLSetData)
	{
		free(theViewer->mJuliaGLSetData);
		theViewer->mJuliaGLSetData = 0;
	}

	if (theViewer->mJuliaSetTexture)
	{
		glDeleteTextures(1, &theViewer->mJuliaSetTexture);
		theViewer->mJuliaSetTexture = 0;
	}

	if (theViewer->mJuliaSetPixelBuffer)
	{
		//DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(theViewer->mJuliaSetPixelBuffer));
		cudaGraphicsUnregisterResource(theViewer->mJCudaJuliaSetResource);
		glDeleteBuffers(1, &theViewer->mJuliaSetPixelBuffer);
		theViewer->mJuliaSetPixelBuffer = 0;
	}

// allocate new buffers
uint16_t width, height;
theViewer->getScreenDimensions(width, height);

theViewer->mJuliaGLSetData = (uchar4 *)malloc(4 * width * height);

printf("Creating GL texture...\n");
glEnable(GL_TEXTURE_2D);
glGenTextures(1, &theViewer->mJuliaSetTexture);
glBindTexture(GL_TEXTURE_2D, theViewer->mJuliaSetTexture);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, theViewer->mJuliaGLSetData);
glDisable(GL_TEXTURE_2D);
printf("Texture created.\n");

printf("Creating PBO...\n");
glGenBuffers(1, &theViewer->mJuliaSetPixelBuffer);
glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, theViewer->mJuliaSetPixelBuffer);
glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, 4 * width * height, theViewer->mJuliaGLSetData, GL_STREAM_COPY);
//While a PBO is registered to CUDA, it can't be used
//as the destination for OpenGL drawing calls.
//But in our particular case OpenGL is only used
//to display the content of the PBO, specified by CUDA kernels,
//so we need to register/unregister it only once.

// DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(theViewer->mJuliaSetPixelBuffer) );
checkCudaErrors(cudaGraphicsGLRegisterBuffer(&theViewer->mJCudaJuliaSetResource, theViewer->mJuliaSetPixelBuffer,
  cudaGraphicsMapFlagsWriteDiscard));
printf("PBO created.\n");
}

void convergency(uint32_t* image, uint16_t iImageWidth, uint16_t iImageHeight, uint16_t iWPixel, uint16_t iHPixel, double iXCenter, double iYCenter, double iPixelWidth, uint16_t nbIterations, double* dd = nullptr, uint16_t* iter = nullptr)
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
      //gNbDiv++;
      double dist = 1.4426954 * log(log(radius2) / 1.38629436111);
      if (dd)
        *dd = dist;
      if (image)
      {
        double f = ((n + 1) % 12 - dist) / 12.0;
        if (f < 0)
          f = .0;
        uint32_t color = (unsigned char)(10.0 * f);
        color |= ((unsigned char)(20.0 * f)) << 8;
        color |= ((unsigned char)(255.0 * f)) << 16;
        color |= 255 << 24;
        image[iHPixel*iImageWidth + iWPixel] = color;
      }
      if (iter)
        *iter = n;
      return;
    }
  }

  //gNbConv++;
  if (image)
  {
    uint32_t color = 100;
    color |= 50 << 8;
    color |= 255 << 16;
    color |= 255 << 24;
    image[iHPixel*iImageWidth + iWPixel] = color;
  }

  if (iter)
    *iter = 0;
}

void ComputeThread(uint16_t affinity, uint32_t* image, uint16_t width, uint16_t height, uint32_t start, uint32_t end, double pixelSize, double cx, double cy, uint16_t nbIterations)
{
  using namespace std::chrono_literals;

  PROCESSOR_NUMBER pn;
  GetCurrentProcessorNumberEx(&pn);
  while (pn.Number != affinity)
  {
    std::this_thread::sleep_for(10us);
    GetCurrentProcessorNumberEx(&pn);
  }
   
  /*std::stringstream msg;
  msg << "Execute thread on " << (int)pn.Number << " :" << start << " - " << end << std::endl;
  std::cout << msg.str();*/
  
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
}

void ComputeOnCpu(uint32_t* image, uint16_t width, uint16_t height, double pixelSize, double cx, double cy, uint16_t nbIterations, float& elpasedTime)
{
  std::string mode = "multiFix"; // single multiFix

  if (mode == "single")
  {
    for (uint_fast16_t i = 0; i < width; i++)
      for (uint_fast16_t j = 0; j < height; j++)
        convergency(image, width, height, i, j, cx, cy, pixelSize, nbIterations);
  }
  else if (mode == "multiFix")
  {
    uint32_t totalSize = width*height;
    uint32_t packetSize = (uint32_t)(totalSize / 8.);
    for (uint16_t t = 0; t < std::thread::hardware_concurrency(); t++)
    {
      std::thread* p = nullptr;
      if (t != std::thread::hardware_concurrency() - 1)
        p = new std::thread(ComputeThread, t, image, width, height, t*packetSize, (t + 1)*packetSize, pixelSize, cx, cy, nbIterations);
      else
        p = new std::thread(ComputeThread, t, image, width, height, t*packetSize, totalSize, pixelSize, cx, cy, nbIterations);
      auto handle = p->native_handle();
      SetThreadAffinityMask(handle, (DWORD_PTR)1 << t);
      p->join();
    }
  }

  elpasedTime = 0;

  //std::cout << "div=" << gNbDiv << "   conv=" << gNbConv << std::endl;
  //memset(image, 0xff, w*h * 4);
}

// render image using CUDA or CPU
void renderImage(bool cpu)
{
	if (theViewer->mZoom != NB_ZOOM_STEPS)
	{
		theViewer->mZoom++;
		double ratio;
		ratio = (double)theViewer->mZoom * (double)theViewer->mZoom / (NB_ZOOM_STEPS*NB_ZOOM_STEPS);
		/*
		if (theViewer->mMode == 0)
		{
			theViewer->mPixelXYWidth[0]  = ratio * 4.0 / width + (1 - ratio)*theViewer->mZoomPixelXYWidth;
			theViewer->mWindowCenterX[0] = (1 - ratio)*theViewer->mZoomWindowCenterX;
			theViewer->mWindowCenterY[0] = (1 - ratio)*theViewer->mZoomWindowCenterY;
		}
		else
		{
			theViewer->mPixelXYWidth[1]  = ratio * 3.0 / width + (1 - ratio)*theViewer->mZoomPixelXYWidth;
			theViewer->mWindowCenterX[1] = -2. * ratio + (1 - ratio)*theViewer->mZoomWindowCenterX;
			theViewer->mWindowCenterY[1] = (1 - ratio)*theViewer->mZoomWindowCenterY;
		}*/
	}
	
  theViewer->getActiveView().SetNbIterations((uint16_t)(70. + 50.*log(.1 / theViewer->getActiveView().PlaneSizePixel())));
	// GPU 
	if (1)
  {
    checkCudaErrors(cudaGraphicsMapResources(1, &theViewer->mJCudaJuliaSetResource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&theViewer->mJuliaCudaSetData, &num_bytes, theViewer->mJCudaJuliaSetResource));

    //if (theViewer->mMode == 0)
      //JuliaCuda(theViewer->mJuliaCudaSetData, theViewer->_JuliaView.Width(), theViewer->_JuliaView.Height(),
      //          theViewer->_JuliaView.PlaneSizePixel(), theViewer->_JuliaView.Center()._x, theViewer->_JuliaView.Center()._y, theViewer->mJuliaParam, ((int)(step / 100) % 2)*(100 - step % 100) + ((int)(step / 100 + 1) % 2)*(step % 100));
		//else
    	cuComputeAndFillImageForGrid(theViewer->mJuliaCudaSetData, theViewer->_MandelbrotView.Width(), theViewer->_MandelbrotView.Height(),
                     theViewer->_MandelbrotView.PlaneSizePixel(),
        theViewer->_MandelbrotView.Center()._x, theViewer->_MandelbrotView.Center()._y, theViewer->getActiveView().GetNbIterations(), theViewer->_ElapsedTime);
    //, ((int)(step / 100) % 2)*(100 - step % 100) + ((int)(step / 100 + 1) % 2)*(step % 100));
    
		checkCudaErrors(cudaGraphicsUnmapResources(1, &theViewer->mJCudaJuliaSetResource, 0));
	}
  else
  {
    ComputeOnCpu((uint32_t*)theViewer->mJuliaGLSetData, theViewer->_MandelbrotView.Width(), theViewer->_MandelbrotView.Height(),
      theViewer->_MandelbrotView.PlaneSizePixel(),
      theViewer->_MandelbrotView.Center()._x, theViewer->_MandelbrotView.Center()._y, theViewer->getActiveView().GetNbIterations(), theViewer->_ElapsedTime);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, 4 * theViewer->_MandelbrotView.Width()*theViewer->_MandelbrotView.Height(), theViewer->mJuliaGLSetData, GL_STREAM_COPY);
  }
}


void displayFunc()
{
  static int frame  = 0;
  if (frame > 0)
    exit(1);
  frame++;

	glClear(GL_COLOR_BUFFER_BIT);
	
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, theViewer->mJuliaSetPixelBuffer);

	renderImage(true);

  uint16_t width, height;
  theViewer->getScreenDimensions(width, height);
  
	// load texture from PBO
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, theViewer->mJuliaSetTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, (char*)0);
	
	// fragment program is required to display floating point texture
	//glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
	//glEnable(GL_FRAGMENT_PROGRAM_ARB);

  glColor3f(1., 1., 1.);
	glBegin(GL_QUADS);
		glTexCoord2f(0.f, 0.f);
	glVertex2f(.0f, 1.f);
		glTexCoord2f(1.f, .0f);
	glVertex2f(1.f, 1.f);
		glTexCoord2f(1.f, 1.f);
	glVertex2f(1.0f, 0.0f);
		glTexCoord2f(0.f, 1.0f);
	glVertex2f(0.f, 0.f);
	glEnd();
	
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);
	
	/* red frame
	glLineWidth(3.0);
	glColor3f(1.0, .0, .0);
	glBegin(GL_LINE_LOOP);
	glVertex2f(.0001f, .0001f);
	glVertex2f(.9999f, .0001f);
	glVertex2f(.9999f, .9999f);

	glVertex2f(.0001f, .9999f);
	glEnd();*/
  static int step = 0;
  static int end  = 0;
  static std::vector<PlanePoint> points;
  static PlanePoint tempo1PtInit;
  static PlanePoint tempo1Pt;
 // if (theViewer->mMode == MODE_MANDELBROT)
  {
    if (leftClicked) // && theViewer->tempo1.tv_sec == 0)
    {
      //timespec_get(&theViewer->tempo1, TIME_UTC);
      theViewer->getActiveView().getPlanePoint(theViewer->getPointerPosition(), tempo1PtInit);
      tempo1Pt = { .0,.0 };
      points.clear();
      points.push_back(tempo1PtInit);
      step = 1;
      end = 0;
    }

    if (step) //theViewer->tempo1.tv_sec > 0)
    {
      //timespec ts;
      //timespec_get(&ts, TIME_UTC);
      //double delta1 = (ts.tv_sec-theViewer->tempo1.tv_sec) + (ts.tv_nsec - theViewer->tempo1.tv_nsec) / 1000000000;

      if (!end) //delta1 > step + 0.01)
      {
        double u = tempo1Pt._x*tempo1Pt._x - tempo1Pt._y*tempo1Pt._y;
        double v = 2 * tempo1Pt._x*tempo1Pt._y;
        tempo1Pt._x = u + tempo1PtInit._x;
        tempo1Pt._y = v + tempo1PtInit._y;
        points.push_back(tempo1Pt);
        step++; // = delta1;
      }

      glColor3f(1.0, .0, .0);
      glBegin(GL_LINE_STRIP);
      for (auto pt : points)
      {
        ViewPosition viewPt;
        theViewer->getActiveView().geViewPosition(pt, viewPt);
        glVertex2f((GLfloat)viewPt._x, (GLfloat)viewPt._y);
      }
      glEnd();

      glPointSize(10.0);
      glBegin(GL_POINTS);
      auto pt = points[0];
      ViewPosition viewPt;
      theViewer->getActiveView().geViewPosition(pt, viewPt);
      glColor3f(.0, 1., .0);
      glVertex2f((GLfloat)viewPt._x, (GLfloat)viewPt._y);
      pt = points[points.size() - 1];
      theViewer->getActiveView().geViewPosition(pt, viewPt);
      if (!end)
        glColor3f(.0, .0, 1.);
      glVertex2f((GLfloat)viewPt._x, (GLfloat)viewPt._y);
      glEnd();

      if (!end)
      {
        if (tempo1Pt._x*tempo1Pt._x + tempo1Pt._y*tempo1Pt._y > 20 || step > 100)
          end = 1;
      }
    }
  }
  
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  if (theViewer->getOption("textInfo"))
  {
    static int prTime = 0;
    int time = glutGet(GLUT_ELAPSED_TIME);
    int dTime= time - prTime;
    prTime = time;

    char str[150];
    PlanePoint planePos;
    theViewer->getActiveView().getPlanePoint(theViewer->getPointerPosition(), planePos);
    double dist=.0, ax=.0, ay=.0, convergence=.0;
    uint16_t iter = 0;
    //convergency(nullptr, theViewer->_MandelbrotView.Width(), theViewer->_MandelbrotView.Height(),
     //theViewer->_Pointer._w, theViewer->_Pointer._h,
      //theViewer->_MandelbrotView.Center()._x, theViewer->_MandelbrotView.Center()._y, theViewer->_MandelbrotView.PlaneSizePixel(), theViewer->getActiveView().GetNbIterations(), &dist, &iter);
   
    /*if (theViewer->mMode == 0)
      iter = JuliaCompute(planePos._x, planePos._y, theViewer->mJuliaParam.real(), theViewer->mJuliaParam.imaginary());
    else*/
    iter = MandelbrotGPU(theViewer->_Pointer._w, theViewer->_Pointer._h, theViewer->_MandelbrotView.Width(), theViewer->_MandelbrotView.Height(),
      theViewer->_MandelbrotView.PlaneSizePixel(), theViewer->_MandelbrotView.Center()._x, theViewer->_MandelbrotView.Center()._y, 
      theViewer->getActiveView().GetNbIterations(), dist, ax, ay);

    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos2f(.05f, .12f);
    sprintf_s(str, "T=%4dms CudaT=%5.1lf Iter=%4d P=[%4d,%4d] W=[%4d,%4d] C=[%5lf,%5lf] Z=[%5lf,%5lf]", dTime, theViewer->_ElapsedTime, theViewer->getActiveView().GetNbIterations(), theViewer->_Pointer._w, theViewer->_Pointer._h, width, height, theViewer->mJuliaParam.real(),
      theViewer->mJuliaParam.imaginary(), planePos._x, planePos._y);
    int len = (int)strlen(str);
    for (int i = 0; i < len; i++) {
      glutBitmapCharacter(GLUT_BITMAP_8_BY_13, str[i]);
    }
    glRasterPos2f(.05f, .06f);
    sprintf_s(str, "s=%4d dist=%5lf cv=%5lf asymp=[%5lf,%5lf]", iter, dist, convergence, ax, ay);
    len = (int)strlen(str);
    for (int i = 0; i < len; i++) {
      glutBitmapCharacter(GLUT_BITMAP_8_BY_13, str[i]);
    }
  }

	glutSwapBuffers();
}

void ctrlKeyDown(int key, int x, int y)
{
	theViewer->keys.insert(key);
}

void ctrlKeyUp(int key, int x, int y)
{
	theViewer->keys.erase(key);
}

void asciiKeyDown(unsigned char key, int x, int y)
{
	theViewer->keys.insert((int)key);

  switch (key)
  {
  case 'm':
    ++theViewer->mMode %= 2;
    break;

  case 't':
    theViewer->toggleOption(std::string("textInfo"));
    break;

  case 'z':
    theViewer->mZoom = 0;
    if (theViewer->mMode == 0)
    {
      /*theViewer->mZoomPixelXYWidth  = theViewer->mPixelXYWidth[0];
      theViewer->mZoomWindowCenterX = theViewer->mWindowCenterX[0];
      theViewer->mZoomWindowCenterY = theViewer->mWindowCenterY[0];*/
    }
    break;
  }
		/*theViewer->mPixelXYWidth[1] = 3.0 / width;
    theViewer->mWindowCenterX[1] = -2.0;
    theViewer->mWindowCenterY[1] = .0;*/
}

void asciiKeyUp(unsigned char key, int x, int y)
{
	theViewer->keys.erase((int)key);
}

void clickFunc(int button, int state, int x, int y)
{
	if (button == 0)
	{
		leftClicked = !leftClicked;
	}

	if (button == 1)
	{
		middleClicked = !middleClicked;
	}

	if (button == 2)
	{
		rightClicked = !rightClicked;
	}

	int modifiers = glutGetModifiers();

	if (leftClicked && (modifiers & GLUT_ACTIVE_SHIFT))
	{
		leftClicked = 0;
		middleClicked = 1;
	}

	if (state == GLUT_UP)
	{
		leftClicked = 0;
		middleClicked = 0;
	}
}

void motionFunc(int w, int h)
{
  ScreenPosition pointer = theViewer->getPointerPosition();

  if (glutGetModifiers() & GLUT_ACTIVE_SHIFT)
  {
    // Scale
    double ratio = pow(2,((uint16_t)h - pointer._h) / 200.0);
    theViewer->getActiveView().scale(ratio);
  }
  else if (glutGetModifiers() & GLUT_ACTIVE_CTRL)
	{
    // Translate
    theViewer->getActiveView().translate((int16_t)w - pointer._w, (int16_t)h - pointer._h);
	}
	theViewer->setPointerPosition((uint16_t)w, (uint16_t)h);

	if (glutGetModifiers() & GLUT_ACTIVE_ALT)
	{
		PlanePoint pt;
    theViewer->getActiveView().getPlanePoint(pointer, pt);
		theViewer->mJuliaParam = Complex(pt._x, pt._y);
	}
}

void reshapeFunc(int w, int h)
{
  uint16_t width, height;
  theViewer->getScreenDimensions(width, height);

	double dw = (double)width / w;
	double dh = (double)height / h;
	/*if (dw > dh)
		theViewer->mPixelXYWidth[0] *= dw;
	else
		theViewer->mPixelXYWidth[0] *= dh;*/
	theViewer->setWidth(w);
	theViewer->setHeight(h);

	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glOrtho(0.0, 1.0, 0.0, 1.0, .0, 1.0);

	initOpenGLBuffers();
}


void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(FRAME_PERIOD, timerEvent, 0);
	}
}

void cleanup()
{
	if (theViewer->mJuliaGLSetData)
	{
		free(theViewer->mJuliaGLSetData);
		theViewer->mJuliaGLSetData = 0;
	}

	checkCudaErrors(cudaGraphicsUnregisterResource(theViewer->mJCudaJuliaSetResource));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glDeleteBuffers(1, &theViewer->mJuliaSetPixelBuffer);
	glDeleteTextures(1, &theViewer->mJuliaSetTexture);
	//glDeleteProgramsARB(1, &gl_Shader);
}


int main(int argc, char **argv)
{
	theViewer = new Viewer();

  uint16_t width, height;
  theViewer->getScreenDimensions(width, height);

	int dev = findCudaGLDevice(argc, (const char **)argv);

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	printf("deviceProp.multiProcessorCount=%d\n", deviceProp.multiProcessorCount);

	if (!checkCudaCapabilities(1, 1))
		exit(-1);

	// Initialize OpenGL context first before the CUDA context is created. This is needed
	// to achieve optimal performance with OpenGL/CUDA interop.
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 50);
	glutCreateWindow(argv[0]);
	
	glutDisplayFunc(displayFunc);
	glutKeyboardFunc(asciiKeyDown);
	glutKeyboardUpFunc(asciiKeyUp);
	glutSpecialFunc(ctrlKeyDown);
	glutSpecialUpFunc(ctrlKeyUp);
	glutMouseFunc(clickFunc);
	glutPassiveMotionFunc(motionFunc);
	glutReshapeFunc(reshapeFunc);
	glutTimerFunc(FRAME_PERIOD, timerEvent, 0); 

	if (!isGLVersionSupported(1, 5) || !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
		exit(-2);

	glClearColor((GLclampf)0.2, (GLclampf)0., (GLclampf)0., (GLclampf)1.0);

	initOpenGLBuffers();

	glutCloseFunc(cleanup);

	glutMainLoop();

	delete theViewer;
}