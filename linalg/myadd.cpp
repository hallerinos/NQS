#include <torch/extension.h>
#include <ATen/ATen.h>

// // A simple C++ function using ATen
// void mymm(torch::Tensor x, torch::Tensor y, int n_iter) {
//   // x and y are torch::Tensor, but compatible with at:: functions
//   // Perform the operation using ATen functions

//   for (int i = 0; i < n_iter; i++) {
//     // std::cout << i << "\n";
//     at::linalg_matmul(x, y);
//   }
// }

// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}

// Binding code (usually in a separate file or block)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mymm", &mymm, "Performs multiplication: x @ y");
}
