#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>

#include "utility.cuh"

struct Config
{
    bool cuda_enabled = false;
    int ncells;
    int nfaces;
    int np;
    int nt;
    double Lx;
    double dt;
    double dx;
    double weight;
    double mass;
};

class Fluid
{
public:
    Fluid();
    Fluid(const std::shared_ptr<Config> &config, int nv, double vmin, double vmax, double xdelta, double vdelta);
    ~Fluid();

    Fluid(Fluid &rhs) = delete;
    void operator=(const Fluid &rhs) = delete;

    void init_config(const std::shared_ptr<Config> &config);
    void init_particles_cpu(const std::vector<double> &pmf, int nv, double vmin, double vmax);

    // memory utility functions
    void cpy_momts_h2d();
    void cpy_momts_d2h();
    void cpy_fields_h2d();
    void cpy_fields_d2h();
    
    void accm_momts_cpu();

    double shape1(const double &x1, const double &x2);

    std::shared_ptr<Config> m_config;

    // cpu memory ptrs
    double *m_dens_h;
    double *m_ppos_h;
    double *m_pvel_h;
    double *m_frce_h;

    // gpu memory ptrs
    double *m_ppos_d;
    double *m_pvel_d;
    double *m_dens_d;
    double *m_frce_d;
};

Fluid::Fluid(const std::shared_ptr<Config> &config, int nv, double vmin, double vmax, double xdelta, double vdelta)
{
    // initialise config -- fill in any missing details
    init_config(config);

    // calculate pmf (unnormalised)
    std::vector<double> pmf(m_config->ncells * nv);
    for (int i = 0; i < m_config->ncells; ++i)
    {
        for (int j = 0; j < nv; ++j)
        {
            double xcentre = (i + 0.5) * m_config->dx;
            double vcentre = (j + 0.5) * (vmax-vmin)/nv;
            pmf[i*nv + j] = 1.0 + xdelta*std::sin(2*M_PI*(xcentre/m_config->Lx)) + vdelta*std::sin(2*M_PI*(vcentre/(vmax-vmin)));
        }
    }

    // allocate CPU memory
    m_ppos_h = (double*)malloc(m_config->np * sizeof(double));
    m_pvel_h = (double*)malloc(m_config->np * sizeof(double));
    m_dens_h = (double*)malloc(m_config->nfaces * sizeof(double));
    m_frce_h = (double*)malloc(m_config->nfaces * sizeof(double));

    if (m_config->cuda_enabled == true)
    {        
        // allocate GPU memory
        checkCudaError(cudaMalloc((void**)&m_ppos_d, m_config->np * sizeof(double)));
        checkCudaError(cudaMalloc((void**)&m_pvel_d, m_config->np * sizeof(double)));
        checkCudaError(cudaMalloc((void**)&m_dens_d, m_config->nfaces * sizeof(double)));
        checkCudaError(cudaMalloc((void**)&m_frce_d, m_config->nfaces * sizeof(double)));
    }

    // initialise particles into CPU memory
    init_particles_cpu(pmf, nv, vmin, vmax);

    if (m_config->cuda_enabled == true)
    {
        // copy over initial particle data to GPU from CPU
        // then delete CPU memory as it's no longer required
        checkCudaError(cudaMemcpy(m_ppos_d, m_ppos_h, m_config->np*sizeof(double), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(m_ppos_d, m_ppos_h, m_config->np*sizeof(double), cudaMemcpyHostToDevice));
        free(m_ppos_h);
        free(m_pvel_h);
        m_ppos_h = nullptr;
        m_pvel_h = nullptr;
    }

    std::cout << "Initialisation complete\n";
}

Fluid::~Fluid()
{
    free(m_dens_h);
    free(m_frce_h);
    if (m_config->cuda_enabled == false)
    {
        free(m_ppos_h);
        free(m_pvel_h);
    }

    if (m_config->cuda_enabled == true)
    {
        checkCudaError(cudaFree(m_ppos_d));
        checkCudaError(cudaFree(m_pvel_d));
        checkCudaError(cudaFree(m_dens_d));
        checkCudaError(cudaFree(m_frce_d));
    }

    std::cout << "Destruction complete\n";
}

void Fluid::init_particles_cpu(const std::vector<double> &pmf, int nv, double vmin, double vmax)
{
    double dv = (vmax - vmin)/nv;
    double sum = std::accumulate(pmf.cbegin(), pmf.cend(), 0.);
    int pidx = 0;
    for (int i = 0; i < m_config->ncells; ++i)
    {
        for (int j = 0; j < nv; ++j)
        {
            int target = std::floor(m_config->np * pmf[i*nv + j] / sum);
            for (int k = 0; k < target; ++k)
            {
                double uniform_rand1 = std::rand() / double(RAND_MAX);
                double uniform_rand2 = std::rand() / double(RAND_MAX);
                m_ppos_h[pidx] = (i + uniform_rand1) * m_config->dx;
                m_pvel_h[pidx] = vmin + (j + uniform_rand2) * dv;
                ++pidx;
            }
        }
    }
    // allocate surplus particles randomly within the domain
    for ( ; pidx < m_config->np; ++pidx)
    {
        int i = std::rand() % m_config->ncells;
        int j = std::rand() % nv;
        double uniform_rand1 = std::rand() / double(RAND_MAX);
        double uniform_rand2 = std::rand() / double(RAND_MAX);
        m_ppos_h[pidx] = (i + uniform_rand1) * m_config->dx;
        m_pvel_h[pidx] = vmin + (j + uniform_rand2) * dv;
    }
}

void Fluid::cpy_momts_h2d()
{
    if (m_config->cuda_enabled == true)
    {
        checkCudaError(cudaMemcpy(m_dens_d, m_dens_h, m_config->nfaces*sizeof(double), cudaMemcpyHostToDevice));
    }
}

void Fluid::cpy_momts_d2h()
{
    if (m_config->cuda_enabled == true)
    {
        checkCudaError(cudaMemcpy(m_dens_h, m_dens_d, m_config->nfaces*sizeof(double), cudaMemcpyDeviceToHost));
    }
}

inline double Fluid::shape1(const double &x1, const double &x2)
{
    double result = 1.0 - std::abs(x1 - x2)/m_config->dx;
    result = std::max(0., result);
    return result;
}

void Fluid::accm_momts_cpu()
{
    // set density to be zero
    std::fill(m_dens_h, m_dens_h + m_config->nfaces, 0.);

    for (int idx = 0; idx < m_config->np; ++idx)
    {
        int cidx = std::floor(m_ppos_h[idx] / m_config->dx);
        
        m_dens_h[cidx]   += m_config->weight * shape1(m_ppos_h[idx], m_config->dx*cidx) / m_config->dx;
        m_dens_h[cidx+1] += m_config->weight * shape1(m_ppos_h[idx], m_config->dx*(cidx+1)) / m_config->dx;
    }
}

__device__ double shape1_d(const double &dx, const double &x1, const double &x2)
{
    double result = 1.0 - abs(x1 - x2)/dx;
    result = max(0., result);
    return result;
}

__global__ void kern_accm_momts(int np, double dx, double weight, double *dens, double *ppos)
{
    int pidx = threadIdx.x + blockIdx.x * blockDim.x;
    for ( ; pidx < np; pidx += blockDim.x * gridDim.x)
    {
        
    }
}
