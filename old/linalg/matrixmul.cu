#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "kernel.cu"
#include "dev_array.h"
#include <math.h>

using namespace std;

int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N = 2048;
    int SIZE = N*N;

    // Allocate memory on the host
    vector<double> h_A(SIZE);
    vector<double> h_B(SIZE);
    vector<double> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = sin(i);
            h_B[i*N+j] = cos(j);
        }
    }

    // Allocate memory on the device
    dev_array<double> d_A(SIZE);
    dev_array<double> d_B(SIZE);
    dev_array<double> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    for (int i = 0; i < 1000000; i++){
        matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    };
    cudaDeviceSynchronize();

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

    // double *cpu_C;
    // cpu_C=new double[SIZE];

    // // Now do the matrix multiplication on the CPU
    // double sum;
    // for (int row=0; row<N; row++){
    //     for (int col=0; col<N; col++){
    //         sum = 0.f;
    //         for (int n=0; n<N; n++){
    //             sum += h_A[row*N+n]*h_B[n*N+col];
    //         }
    //         cpu_C[row*N+col] = sum;
    //     }
    // }

    // double err = 0;
    // // Check the result and make sure it is correct
    // for (int ROW=0; ROW < N; ROW++){
    //     for (int COL=0; COL < N; COL++){
    //         err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
    //     }
    // }

    // cout << "Error: " << err << endl;

    return 0;
}