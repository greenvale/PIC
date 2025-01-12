#include <iostream>
#include <stdio.h>
#include <memory>
#include <fstream>
#include <chrono>
#include <random>
#include <utility>
#include <string>
#include <cstdlib>
#include <algorithm>

__global__ void push(double *x0, double *x1, double *v0, double *v1, int N, double dt) {
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    for ( ; i < N; i += blockDim.x * gridDim.x) {
        x1[i] = x0[i] + v0[i] * dt;
        x1[i] = x1[i] - __double2int_rd(x1[i]);
        v1[i] = v0[i] * 1.01;
    }
}

__global__ void histogram(double *x, int *bins, int N, int Nbins)
{
    extern __shared__ int local_bins[];

    if (threadIdx.x < Nbins) {
        local_bins[threadIdx.x] = 0;
    }

    __syncthreads();

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    for ( ; i < N; i += blockDim.x * gridDim.x) {
        int bin_idx = __double2int_rd(x[i] * Nbins);
        atomicAdd(&local_bins[bin_idx], 1);
    }

    __syncthreads();

    if (threadIdx.x < Nbins) {
        atomicAdd(&bins[threadIdx.x], local_bins[threadIdx.x]);
    }

}


int main(int argc, char *argv[])
{
    int blockSize = 256;
    int numBlocks = 32;

    int Np = 2 << 16;
    int Nbins = 32;
    int sharedMemorySize = Nbins * sizeof(int);

    double dt = 0.001;
    int Nt = 2 << 8;

    // particle positions
    double *x0 = new double[Np];
    double *x1 = new double[Np];
    double *x0_h = new double[Np];
    double *x1_h = new double[Np];
    double *x0_d;
    double *x1_d;
    cudaMalloc((void**)&x0_d, Np * sizeof(double));
    cudaMalloc((void**)&x1_d, Np * sizeof(double));
    
    // particle velocities
    double *v0 = new double[Np];
    double *v1 = new double[Np];
    double *v0_h = new double[Np];
    double *v1_h = new double[Np];
    double *v0_d;
    double *v1_d;
    cudaMalloc((void**)&v0_d, Np * sizeof(double));
    cudaMalloc((void**)&v1_d, Np * sizeof(double));

    // density
    int *bins0 = new int[Nbins];
    int *bins1 = new int[Nbins];
    int *bins0_h = new int[Nbins];
    int *bins1_h = new int[Nbins];
    int *bins0_d;
    int *bins1_d;
    cudaMalloc((void**)&bins0_d, Nbins * sizeof(int));
    cudaMalloc((void**)&bins1_d, Nbins * sizeof(int));

    // initialise position and velocity of particles
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist_x(0.5, 0.1);
    std::normal_distribution<> dist_v(1.0, 0.01);
    for (int i = 0; i < Np; ++i) {
        x0[i] = dist_x(gen);
        x0[i] = x0[i] - std::floor(x0[i]);
        v0[i] = dist_v(gen);
        x0_h[i] = x0[i];
        v0_h[i] = v0[i];
    }

    cudaMemcpy(x0_d, x0_h, Np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v0_d, v0_h, Np * sizeof(double), cudaMemcpyHostToDevice);

    std::ofstream outfile("dens.csv");


    if (false) {
        for (int k = 0; k < Nt; ++k) {

            for (int i = 0; i < Np; ++i) {
                x1[i] = x0[i] + v0[i] * dt;
                x1[i] = x1[i] - std::floor(x1[i]);
                v1[i] = v0[i] * 1.01;
            }

            for (int i = 0; i < Nbins; ++i) {
                bins1[i] = 0;
            }
            for (int i = 0; i < Np; ++i) {
                int idx = static_cast<int>(std::floor(x1[i] * Nbins));
                bins1[idx] += 1;
            }

            std::copy(x1, x1 + Np, x0);
            std::copy(v1, v1 + Np, v0);
            std::copy(bins1, bins1 + Nbins, bins0);
        }
    }


    for (int k = 0; k < Nt; ++k) {
        // push particles
        push<<<numBlocks, blockSize>>>(x0_d, x1_d, v0_d, v1_d, Np, dt);

        // intialise bins at t=k+1 to be empty
        for (int i = 0; i < Nbins; ++i) {
            bins1_h[i] = 0;
        }
        cudaMemcpy(bins1_d, bins1_h, Nbins * sizeof(int), cudaMemcpyHostToDevice); 

        // get density
        histogram<<<numBlocks, blockSize, sharedMemorySize>>>(x1_d, bins1_d, Np, Nbins);
        
        // record density
        cudaMemcpy(bins1_h, bins1_d, Nbins * sizeof(int), cudaMemcpyDeviceToHost);
        if (outfile.is_open()) {
            for (int i = 0; i < Nbins; ++i) {
                outfile << bins1_h[i];
                if (i < Nbins - 1) {
                    outfile << ",";
                }
            }            
            if (k < Nt - 1) {
                outfile << "\n";
            }
        }

        // transfer data at k+1 to k
        // previously copied memory on the device from k+1 -> k but instead choosing to swap pointers
        //cudaMemcpy(x0_d, x1_d, Np * sizeof(double), cudaMemcpyDeviceToDevice);
        //cudaMemcpy(v0_d, v1_d, Np * sizeof(double), cudaMemcpyDeviceToDevice);
        //cudaMemcpy(bins0_d, bins1_d, Nbins * sizeof(int), cudaMemcpyDeviceToDevice);

        double *tmp_ptr;

        tmp_ptr = x0_d;
        x0_d = x1_d;
        x1_d = tmp_ptr;

        tmp_ptr = v0_d;
        v0_d = v1_d;
        v1_d = tmp_ptr;

        int *tmp_ptr2;
        tmp_ptr2 = bins0_d;
        bins0_d = bins1_d;
        bins1_d = tmp_ptr2;

    }
}