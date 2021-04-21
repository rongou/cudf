// build:
// nvcc -lcufile cufile_bug.cu
//
// run:
// ./a.out

#include <cassert>
#include <string>

#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>

#include <sys/stat.h>

int main() {
  auto const size = 1 << 30;

  assert(cudaSetDevice(1) == cudaSuccess);
  assert(cudaFree(0) == cudaSuccess);

  void *buffers[2];
  for (int i = 0; i < 2; i++) {
    assert(cudaMalloc(&buffers[i], size) == cudaSuccess);
    assert(cudaMemset(buffers[i], i, size) == cudaSuccess);
  }

  assert(cuFileDriverOpen().err == CU_FILE_SUCCESS);

  for (int i = 0; i < 2; i++) {
    assert(cudaSetDevice(1) == cudaSuccess);
    auto const file_descriptor = open(("/data/rou/tmp/cufile" + std::to_string(i)).c_str(),
                                      O_CREAT | O_WRONLY | O_DIRECT, S_IRUSR | S_IWUSR);
    assert(file_descriptor >= 0);
    CUfileDescr_t cufile_descriptor{CU_FILE_HANDLE_TYPE_OPAQUE_FD, file_descriptor};
    CUfileHandle_t cufile_handle{};
    assert(cuFileHandleRegister(&cufile_handle, &cufile_descriptor).err == CU_FILE_SUCCESS);

    assert(cuFileWrite(cufile_handle, buffers[i], size, 0, 0) == size);

    assert(cudaFree(buffers[i]) == cudaSuccess);
    cuFileHandleDeregister(cufile_handle);
    assert(close(file_descriptor) == 0);
  }

  assert(cuFileDriverClose().err == CU_FILE_SUCCESS);

  return 0;
}
