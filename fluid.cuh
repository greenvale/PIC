#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <set>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "math.h"
#include "../utils/utility.cuh"
#include "../utils/storage.cuh"

std::vector<double> perturbed_pmf(int nx, int nv, double xmin, double xmax, double vmin, double vmax, double xdelta, double vdelta)
{
    double dx = (xmax - xmin) / nx;
    double dv = (vmax - vmin) / nv;
    std::vector<double> pmf(nx * nv);
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < nv; ++j)
        {
            double xcentre = (i + 0.5) * dx;
            double vcentre = (j + 0.5) * dv;
            pmf[i*nv + j] = 1.0 + xdelta*std::sin(2*M_PI*(xcentre/(xmax-xmin))) + vdelta*std::sin(2*M_PI*(vcentre/(vmax-vmin)));
        }
    }
    double sum = std::accumulate(pmf.begin(), pmf.end(), 0.);
    for (auto &x : pmf)
    {
        x /= sum;
    }
    return pmf;
}

// distribution data for fluid (taking position and velocity)
struct Distribution
{
    double xmin, xmax, vmin, vmax;
    size_t nx, nv;
    std::vector<double> pmf;
};

// declare fluid class to make friend of fluid config
class Fluid;

class FluidConfig
{
    friend Fluid;
public:
    size_t ncells;
    size_t np;
    double Lx;
    double dt;
    double mass;
    double charge;
    void complete()
    {
        if (completed == true)
        {
            throw std::runtime_error("config has already been completed and configured");
        }
        assert(completed == false);
        dx = Lx / ncells;
        nfaces = ncells + 1;
        weight = 1.0 / np;
        completed = true;
    }
private:
    bool completed = false;
    double dx;
    double weight;
    size_t nfaces;
};

class Fluid
{
public:
    Fluid() = delete;
    Fluid(const FluidConfig &config);
    ~Fluid();

    Fluid(Fluid &rhs) = delete;
    void operator=(const Fluid &rhs) = delete;

    void init_particles_h(const Distribution &dist);
    
    double shape1(const double &x1, const double &x2);
    double LHSface(const int &idx);
    double RHSface(const int &idx);
    double centre(const int &idx);
    int pos2idx(const double &pos);

    void update_momts_h();
    void update_fields_h();
    void push_particles_h();

    void update_momts_d();
    void update_fields_d();
    void push_particles_d();

public:
    FluidConfig m_config;

    Storage<double> m_dens;
    Storage<double> m_pres;
    Storage<double> m_ppos;
    Storage<double> m_pvel;
};

Fluid::Fluid(const FluidConfig &config) :
    m_config(config),
    m_dens(Storage<double>(config.nfaces)),
    m_pres(Storage<double>(config.nfaces)),
    m_ppos(Storage<double>(config.np)),
    m_pvel(Storage<double>(config.np))
{
    if (config.completed == false)
    {
        throw std::runtime_error("fluid config must be completed before initialisation");
    }
}

Fluid::~Fluid()
{
    
}

void Fluid::init_particles_h(const Distribution &dist)
{
    double *ppos_ptr = m_ppos.hptr();
    double *pvel_ptr = m_pvel.hptr();

    double dv = (dist.vmax - dist.vmin) / double(dist.nv);
    double sum = std::accumulate(dist.pmf.cbegin(), dist.pmf.cend(), 0.);
    int pidx = 0;
    for (int i = 0; i < m_config.ncells; ++i)
    {
        for (int j = 0; j < dist.nv; ++j)
        {
            int target = std::floor(m_config.np * dist.pmf[i*dist.nv + j] / sum);
            for (int k = 0; k < target; ++k)
            {
                double uniform_rand1 = double(std::rand()) / double(RAND_MAX);
                double uniform_rand2 = double(std::rand()) / double(RAND_MAX);
                *(ppos_ptr) = (i + uniform_rand1) * m_config.dx;
                *(pvel_ptr) = dist.vmin + (j + uniform_rand2) * dv;
                ppos_ptr++;
                pvel_ptr++;
                ++pidx;
            }
        }
    }
    // allocate surplus particles randomly within the domain
    for ( ; pidx < m_config.np; ++pidx)
    {
        int i = std::rand() % m_config.ncells;
        int j = std::rand() % dist.nv;
        double uniform_rand1 = double(std::rand()) / double(RAND_MAX);
        double uniform_rand2 = double(std::rand()) / double(RAND_MAX);
        *(ppos_ptr) = (i + uniform_rand1) * m_config.dx;
        *(pvel_ptr) = dist.vmin + (j + uniform_rand2) * dv;
        ppos_ptr++;
        pvel_ptr++;
    }
}

inline double Fluid::shape1(const double &x1, const double &x2)
{
    double result = 1.0 - std::abs(x1 - x2)/m_config.dx;
    result = std::max(0., result);
    return result;
}

inline double Fluid::LHSface(const int &idx)
{
    return m_config.dx * idx;
}

inline double Fluid::RHSface(const int &idx)
{
    return m_config.dx * (idx + 1);
}

inline double Fluid::centre(const int &idx)
{
    return m_config.dx * (idx + 0.5);
}

inline int Fluid::pos2idx(const double &pos)
{
    return std::floor(pos / m_config.dx);
}

// accumulates movments
void Fluid::update_momts_h()
{
    double *dens_ptr = m_dens.hptr();
    double *ppos_ptr = m_ppos.hptr();

    // set density to be zero
    std::fill(dens_ptr, dens_ptr + m_config.nfaces, 0.);

    for (int idx = 0; idx < m_config.np; ++idx)
    {
        int cidx = std::floor(ppos_ptr[idx] / m_config.dx);
        
        dens_ptr[cidx]   += m_config.weight * shape1(ppos_ptr[idx], m_config.dx*cidx) / m_config.dx;
        dens_ptr[cidx+1] += m_config.weight * shape1(ppos_ptr[idx], m_config.dx*(cidx+1)) / m_config.dx;
    }
}

void Fluid::update_fields_h()
{
    double *dens_ptr = m_dens.hptr();
    double *pres_ptr = m_pres.hptr();

    double alpha = 0.1;

    // calculate force on boundaries first
    pres_ptr[0] = -alpha * (dens_ptr[1] - dens_ptr[m_config.ncells - 1]) / (2*m_config.dx); 
    pres_ptr[m_config.ncells] = pres_ptr[0];

    // calculate force on inner faces
    for (int idx = 1; idx < m_config.ncells - 1; ++idx)
    {
        pres_ptr[idx] = -alpha * (dens_ptr[idx+1] - dens_ptr[idx-1]) / (2*m_config.dx);
    }
}

void Fluid::push_particles_h()
{
    double *ppos_ptr = m_ppos.hptr();
    double *pvel_ptr = m_pvel.hptr();
    double *dens_ptr = m_dens.hptr();
    double *pres_ptr = m_pres.hptr();

    double dpos, dvel, accel, shapeLHS, shapeRHS;
    int cidx;

    for (int pidx = 0; pidx < m_config.np; ++pidx)
    {
        // get cell index given particle position
        cidx = pos2idx(ppos_ptr[pidx]);

        // calculate shapes for lhs/rhs cell faces
        shapeLHS = shape1(LHSface(cidx), ppos_ptr[pidx]);
        shapeRHS = shape1(RHSface(cidx), ppos_ptr[pidx]);

        // calculate acceleration and change in vel and pos
        accel = (shapeLHS*pres_ptr[cidx] + shapeRHS*pres_ptr[cidx+1]) / m_config.mass;
        dvel = accel * m_config.dt;
        dpos = pvel_ptr[pidx]*m_config.dt + 0.5*accel*m_config.dt*m_config.dt;

        ppos_ptr[pidx] += dpos;
        pvel_ptr[pidx] += dvel;

        // enforce periodic boundary conditions
        while (ppos_ptr[pidx] < 0.)
        {
            ppos_ptr[pidx] += m_config.Lx;
        }
        while (ppos_ptr[pidx] >= m_config.Lx)
        {
            ppos_ptr[pidx] -= m_config.Lx;
        }
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

void Fluid::update_momts_d()
{

}

void Fluid::update_fields_d()
{

}

void Fluid::push_particles_d()
{
    
}