#if defined(__CUDACC__)
#define __CudaCallable__ __device__ __host__ inline
#else
#define __CudaCallable__
#endif
