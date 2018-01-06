#pragma once

#include <cstdint>

#include "CudaIntegration.h"

class Complex
{
private:
	double _Real;
	double _Imaginary;

public:
  __CudaCallable__ Complex() { _Real = _Imaginary = .0; }
  __CudaCallable__ Complex(const Complex& copy) { _Real = copy._Real; _Imaginary = copy._Imaginary; }
  __CudaCallable__ Complex(const double& r, const double& i) { _Real = r; _Imaginary = i; }

  __CudaCallable__ const double& real() const { return _Real; }
  __CudaCallable__ const double& imaginary() const { return _Imaginary; }
	
  __CudaCallable__ double modulus() const { return _Real*_Real+_Imaginary*_Imaginary; }

  __CudaCallable__ double distanceTo(const Complex& to) const { double dx = _Real-to._Real; double dy = _Imaginary - to._Imaginary; return sqrt(dx*dx + dy*dy); }
  __CudaCallable__ Complex operator + (Complex operand) const { return Complex(_Real + operand._Real, _Imaginary + operand._Imaginary); }
  __CudaCallable__ Complex operator * (Complex operand) const { return Complex(_Real * operand._Real - _Imaginary * operand._Imaginary, _Real * operand._Imaginary + operand._Real * _Imaginary); }
  __CudaCallable__ Complex operator * (double ratio) const { return Complex(ratio*_Real, ratio*_Imaginary); }
};
