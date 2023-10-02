#include <iostream>
#include <stdio.h>
#include <memory>
#include <fstream>

#include "fluid.hpp"
#include "utility.cuh"
#include "fluid.cuh"

#define CUDA 1

int main(int argc, char *argv[])
{
    std::ofstream pos_os;
    std::ofstream dns_os;
    pos_os.open("pos.csv", std::ios::trunc);
    dns_os.open("dns.csv", std::ios::trunc);

    auto config = std::make_shared<Config>();
    config->Lx = 1.0;
    config->ncells = 1<<5;
    config->npts = 1<<20;
    config->dt = 0.001;
    config->mass = 1.;

#if (CUDA==1)
    
    Fluid fluidCPU(config);
    cudaFluid fluidGPU(config);

    fluidCPU.update_mmts();
    fluidGPU.update_mmts();
    fluidGPU.device2host();

    print_row(dns_os, config->nfaces, fluidCPU.m_dens_h);
    print_row(dns_os, config->nfaces, fluidGPU.m_dens_h);
    
#else
    std::cout << "Running on CPU\n";
    Fluid fluid(config);

    int print_freq = 100;

    for (int k = 0; k < 10000; ++k)
    {
        fluid.update_mmts();
        fluid.update_forces();
        fluid.push_pts();

        if (k % print_freq == 0)
        {
            std::cout << "Printed @ k =" << k << "\n";
            //print_row(pos_os, config->npts, fluid.m_ppos);
            print_row(dns_os, config->nfaces, fluid.m_dens_h);
        }
    }
    
    pos_os.close();
    dns_os.close();
#endif

}