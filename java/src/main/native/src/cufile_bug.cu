// build:
// nvcc --default-stream per-thread -lcufile cufile_bug.cu
//
// run:
// ./a.out

#include <cassert>
#include <string>
#include <thread>

#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>

#include <sys/stat.h>

#define DEVICE 0
#define SIZE 1 << 30
#define BASE "/tmp/cufile"

void write_buffers(void **buffers) {
  for (int i = 0; i < 2; i++) {
    assert(cudaSetDevice(DEVICE) == cudaSuccess);
    auto const file_descriptor = open((BASE + std::to_string(i)).c_str(),
                                      O_CREAT | O_WRONLY | O_DIRECT, S_IRUSR | S_IWUSR);
    assert(file_descriptor >= 0);
    CUfileDescr_t cufile_descriptor{CU_FILE_HANDLE_TYPE_OPAQUE_FD, file_descriptor};
    CUfileHandle_t cufile_handle{};
    assert(cuFileHandleRegister(&cufile_handle, &cufile_descriptor).err == CU_FILE_SUCCESS);

    assert(cuFileWrite(cufile_handle, buffers[i], SIZE, 0, 0) == SIZE);

    assert(cudaFree(buffers[i]) == cudaSuccess);
    cuFileHandleDeregister(cufile_handle);
    assert(close(file_descriptor) == 0);
  }
}

int main() {
  assert(cudaSetDevice(DEVICE) == cudaSuccess);
  assert(cudaFree(0) == cudaSuccess);

  void *buffers[2];
  for (int i = 0; i < 2; i++) {
    assert(cudaMalloc(&buffers[i], SIZE) == cudaSuccess);
    assert(cudaMemset(buffers[i], i, SIZE) == cudaSuccess);
  }

  assert(cuFileDriverOpen().err == CU_FILE_SUCCESS);
  std::thread t{write_buffers, buffers};
  t.join();
  assert(cuFileDriverClose().err == CU_FILE_SUCCESS);

  return 0;
}
