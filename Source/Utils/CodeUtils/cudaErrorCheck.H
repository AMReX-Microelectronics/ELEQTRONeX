////////////////////////////////////////////////////////////////////////////////
// CUDA error checking
////////////////////////////////////////////////////////////////////////////////
#ifdef AMREX_USE_CUDA
#define checkCudaErrors(val) \
    __checkCudaErrors__((val), #val, __FILE__, __LINE__)

template <typename T>
inline void __checkCudaErrors__(T code, const char *func, const char *file,
                                int line)
{
    if (code)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
                (unsigned int)code, func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#else
// No-op for non-CUDA builds
#define checkCudaErrors(val) do { (void)(val); } while(0)
#endif
