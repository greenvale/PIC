#include <iostream>
#include <stdio.h>
#include <memory>
#include <fstream>
#include <chrono>

#include "../utils/utility.cuh"
#include "fluid.cuh"

void test_cpu(const FluidConfig &config, const Distribution &dist, int nt)
{
    std::cout << "Starting CPU test\n";
    std::cout << "np=" << config.np << "; nx=" << config.ncells << "; nt=" << nt << "\n";

    Fluid fluid(config);

    fluid.m_ppos.allocate_h();
    fluid.m_pvel.allocate_h();
    fluid.m_dens.allocate_h();
    fluid.m_pres.allocate_h();

    fluid.init_particles_h(dist);

    auto start = std::chrono::high_resolution_clock::now();

    int k_steps = std::max<int>(1, (int)std::floor(nt / 10.));
    for (int k = 0; k < nt; ++k)
    {
        fluid.update_momts_h();
        fluid.update_fields_h();
        fluid.push_particles_h();
        
        // print progress
        if (k%k_steps == 0)
        {
            std::cout << (int)std::floor(double(k) * 100. / nt) << "% done\n";
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration_c = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double duration = double(duration_c.count()) / 1000000.;
    std::cout << "Finished CPU test\n";
    std::cout << "Duration = " << duration << "s\n";
}

void test_gpu(const FluidConfig &config, const Distribution &dist, int nt)
{
    std::cout << "Starting GPU test\n";
    std::cout << "np=" << config.np << "; nx=" << config.ncells << "; nt=" << nt << "\n";

    Fluid fluid(config);

    // allocate memory for cpu and gpu
    fluid.m_ppos.allocate_h();
    fluid.m_pvel.allocate_h();
    fluid.m_dens.allocate_h();
    fluid.m_pres.allocate_h();
    fluid.m_ppos.allocate_d();
    fluid.m_pvel.allocate_d();
    fluid.m_dens.allocate_d();
    fluid.m_pres.allocate_d();

    // initialise the particle data on the gpu
    fluid.init_particles_h(dist);

    // move particle data to gpu and delete from cpu
    fluid.m_ppos.cpy_h2d();
    fluid.m_pvel.cpy_h2d();
    fluid.m_ppos.free_h();
    fluid.m_pvel.free_h();

    auto start = std::chrono::high_resolution_clock::now();

    int k_steps = std::max<int>(1, (int)std::floor(nt / 10.));
    for (int k = 0; k < nt; ++k)
    {
        fluid.update_momts_d();

        // send updated moments on gpu to cpu
        // update fields on cpu then send to gpu for particle push
        fluid.m_dens.cpy_d2h();
        fluid.update_fields_h();
        fluid.m_pres.cpy_h2d();
        
        fluid.push_particles_d();
        
        // print progress
        if (k%k_steps == 0)
        {
            std::cout << (int)std::floor(double(k) * 100. / nt) << "% done\n";
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration_c = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double duration = double(duration_c.count()) / 1000000.;
    std::cout << "Finished GPU test\n";
    std::cout << "Duration = " << duration << "s\n";
}

int main(int argc, char *argv[])
{
/*
    std::ofstream pos_os;
    std::ofstream dns_os;
    pos_os.open("pos.csv", std::ios::trunc);
    dns_os.open("dns.csv", std::ios::trunc);

    pos_os.close();
    dns_os.close();
*/
    int nt = 1<<10;

    FluidConfig config;
    config.Lx = 1.0;
    config.ncells = 1<<5;
    config.np = 1<<20;
    config.dt = 0.001;
    config.mass = 1.;
    config.complete();

    Distribution dist;
    dist.vmin = -0.5;
    dist.vmax = 0.5;
    dist.nv = config.ncells;
    int xdelta = 0.2;
    int vdelta = 0.1;
    dist.pmf = perturbed_pmf(config.ncells, dist.nv, 0., config.Lx, dist.vmin, dist.vmax, xdelta, vdelta);

    test_cpu(config, dist, nt);

}