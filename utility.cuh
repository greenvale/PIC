#pragma once

#include <iostream>
#include <iomanip>

template <typename TError>
void checkCudaError(TError result)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorName(result) << "\n";
    }
}

template <typename TData>
void print_row(std::ofstream &os, const size_t &N, const TData *arr)
{
    for (size_t idx = 0; idx < N; ++idx)
    {
        if (idx > 0)
            os << ", ";
        os << arr[idx];
    }
    os << "\n";
}

std::string ptr2str(double *ptr)
{
    std::stringstream ss;
    if (ptr == nullptr)
        ss << "nullptr";
    else
        ss << ptr;
    return ss.str();
}