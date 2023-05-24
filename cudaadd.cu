#include <iostream>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;

__global__ void vectorAdd(const float* a, const float* b, float* c, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    int size = 1000000; 
    size_t bytes = size * sizeof(float);

    float* h_a = new float[size];
    float* h_b = new float[size];
    float* h_c = new float[size];

    //Generate numbers
    for (int i = 0; i < size; ++i) {
        h_a[i] = rand()%1000;
        h_b[i] = rand()%1000;
    }

    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;


    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("Added Number: ");
    for (int i = 0; i < 10; ++i) {
        cout<<h_c[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
