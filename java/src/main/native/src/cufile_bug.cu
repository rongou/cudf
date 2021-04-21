// build:
// nvcc -lcuda -lcufile cufile_bug.cu
//
// run:
// ./a.out

#include <cassert>

#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>

#include <sys/stat.h>

#define DEVICE 0
#define SIZE 4097
#define OUT "/tmp/cufile"

int main() {
  assert(cudaSetDevice(DEVICE) == cudaSuccess);
  assert(cudaFree(0) == cudaSuccess);

  void *buffer;
  assert(cudaMalloc(&buffer, SIZE) == cudaSuccess);
  assert(cudaMemset(buffer, 42, SIZE) == cudaSuccess);

  assert(cuFileDriverOpen().err == CU_FILE_SUCCESS);

  auto const file_descriptor = open(OUT, O_CREAT | O_WRONLY | O_DIRECT, S_IRUSR | S_IWUSR);
  assert(file_descriptor >= 0);
  CUfileDescr_t cufile_descriptor{CU_FILE_HANDLE_TYPE_OPAQUE_FD, file_descriptor};
  CUfileHandle_t cufile_handle{};
  assert(cuFileHandleRegister(&cufile_handle, &cufile_descriptor).err == CU_FILE_SUCCESS);

  CUcontext ctx0;
  assert(cuCtxGetCurrent(&ctx0) == CUDA_SUCCESS);

  assert(cuFileWrite(cufile_handle, buffer, SIZE, 0, 0) == SIZE);

  CUcontext ctx1;
  assert(cuCtxGetCurrent(&ctx1) == CUDA_SUCCESS);
  assert(ctx0 == ctx1);

  assert(cudaFree(buffer) == cudaSuccess);
  cuFileHandleDeregister(cufile_handle);
  assert(close(file_descriptor) == 0);

  assert(cuFileDriverClose().err == CU_FILE_SUCCESS);

  return 0;
}
