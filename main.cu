#include <iostream>
#include <stdio.h>
#include <memory>
#include <fstream>

#include "utility.cuh"
#include "fluid.cuh"

int main(int argc, char *argv[])
{
    std::ofstream pos_os;
    std::ofstream dns_os;
    pos_os.open("pos.csv", std::ios::trunc);
    dns_os.open("dns.csv", std::ios::trunc);

    double vmin, vmax, xdelta, vdelta;
    size_t nv;

    auto config = std::make_shared<FluidConfig>();
    config->Lx = 1.0;
    config->ncells = 1<<5;
    config->np = 1<<10;
    config->dt = 0.001;
    config->mass = 1.;
    
    vmin = -0.5;
    vmax = 0.5;
    xdelta = 0.2;
    vdelta = 0.1;
    nv = config->ncells;
    auto pmf = perturbed_pmf(config->ncells, nv, 0., config->Lx, vmin, vmax, xdelta, vdelta);

    Fluid fluid;
    fluid.init_config(config);

    fluid.allocate("h", "ppos");
    fluid.allocate("h", "pvel");
    
    fluid.init_particles_cpu(pmf, nv, vmin, vmax);

    fluid.allocate("d", "ppos");
    fluid.allocate("d", "pvel");

    fluid.copy("h2d", "ppos");
    fluid.copy("h2d", "pvel");

    //fluid.memory_dump("d", "ppos");
    
    fluid.memory_summary();

    pos_os.close();
    dns_os.close();

}