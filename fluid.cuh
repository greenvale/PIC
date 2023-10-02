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
#include "math.h"
#include "utility.cuh"

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
private:
    double dx;
    double weight;
    size_t nfaces;
    void complete()
    {
        dx = Lx / ncells;
        nfaces = ncells + 1;
        weight = 1.0 / np;
    }
};

class Fluid
{
public:
    Fluid();
    ~Fluid();

    Fluid(Fluid &rhs) = delete;
    void operator=(const Fluid &rhs) = delete;

    void init_config(const std::shared_ptr<FluidConfig> &config);
    void init_particles_cpu(const std::vector<double> &pmf, int nv, double vmin, double vmax);
    
    double shape1(const double &x1, const double &x2);

    void update_momts_cpu();
    void update_fields_cpu();
    void push_particles_cpu();

    void update_momts_gpu();
    void update_fields_gpu();
    void update_particles_gpu();

    // memory management functions
    void allocate(std::string owner, std::string var);
    void destroy(std::string owner, std::string var, bool throw_err);
    void copy(std::string path, std::string var);
    void memory_dump(std::string owner, std::string var);
    void memory_summary();

public:
    std::shared_ptr<FluidConfig> m_config;
    static const std::vector<std::string> m_moment_names;
    static const std::vector<std::string> m_field_names;
    static const std::vector<std::string> m_storage_type_names;

    std::unordered_map<std::string, std::unordered_map<std::string, double*>> m_storage;
    std::unordered_map<std::string, std::string> m_storage_types;
    std::unordered_map<std::string, size_t> m_storage_sizes;
    std::set<std::string> m_vars;
};

const std::vector<std::string> Fluid::m_moment_names =  {"dens"};
const std::vector<std::string> Fluid::m_field_names = {"presr"};
const std::vector<std::string> Fluid::m_storage_type_names = {"ptcl", "cell"};

Fluid::Fluid()
{
    m_storage["h"]["ppos"] = nullptr;
    m_storage["h"]["pvel"] = nullptr;
    m_storage["d"]["ppos"] = nullptr;
    m_storage["d"]["pvel"] = nullptr;
    m_storage_types["ppos"] = "ptcl";
    m_storage_types["pvel"] = "ptcl";
    
    for (auto &n : m_moment_names)
    {
        m_storage["h"][n] = nullptr;
        m_storage["d"][n] = nullptr;
        m_storage_types[n] = "cell";
    }
    for (auto &n : m_field_names)
    {
        m_storage["h"][n] = nullptr;
        m_storage["d"][n] = nullptr;
        m_storage_types[n] = "cell";
    }
    
    // store all variables in a set
    for (auto &pair : m_storage["h"])
    {
        m_vars.insert(pair.first);
    }
}

Fluid::~Fluid()
{
    for (auto &pair : m_storage["h"])
    {
        destroy("h", pair.first, false);
    }
    for (auto &pair : m_storage["d"])
    {
        destroy("d", pair.first, false);
    }
    std::cout << "Destruction complete\n";
}

void Fluid::init_config(const std::shared_ptr<FluidConfig> &config)
{
    m_config = config;
    m_config->complete();
    m_storage_sizes["ptcl"] = m_config->np;
    m_storage_sizes["cell"] = m_config->nfaces;
}

void Fluid::init_particles_cpu(const std::vector<double> &pmf, int nv, double vmin, double vmax)
{
    double *ppos_ptr = m_storage["h"]["ppos"];
    double *pvel_ptr = m_storage["h"]["pvel"];
    if (ppos_ptr == nullptr || pvel_ptr == nullptr)
    {
        std::cerr << "ppos and/or ppvel not initialised on h\n";
        exit(1);
    }

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
                ppos_ptr[pidx] = (i + uniform_rand1) * m_config->dx;
                pvel_ptr[pidx] = vmin + (j + uniform_rand2) * dv;
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
        ppos_ptr[pidx] = (i + uniform_rand1) * m_config->dx;
        ppos_ptr[pidx] = vmin + (j + uniform_rand2) * dv;
    }
}

inline double Fluid::shape1(const double &x1, const double &x2)
{
    double result = 1.0 - std::abs(x1 - x2)/m_config->dx;
    result = std::max(0., result);
    return result;
}

// accumulates movments
void Fluid::update_momts_cpu()
{
    double *dens_ptr = m_storage["h"]["dens"];
    double *ppos_ptr = m_storage["h"]["ppos"];

    // set density to be zero
    std::fill(dens_ptr, dens_ptr + m_config->nfaces, 0.);

    for (int idx = 0; idx < m_config->np; ++idx)
    {
        int cidx = std::floor(ppos_ptr[idx] / m_config->dx);
        
        dens_ptr[cidx]   += m_config->weight * shape1(ppos_ptr[idx], m_config->dx*cidx) / m_config->dx;
        dens_ptr[cidx+1] += m_config->weight * shape1(ppos_ptr[idx], m_config->dx*(cidx+1)) / m_config->dx;
    }
}

void Fluid::update_fields_cpu()
{

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

/* Memory management functions */

void Fluid::allocate(std::string owner, std::string var)
{
    if (m_storage[owner][var] == nullptr)
    {
        size_t size = m_storage_sizes[m_storage_types[var]];
        if (owner == "h")
        {
            m_storage[owner][var] = (double*)malloc(size * sizeof(double));
        }
        else if (owner == "d")
        {
            double *ptr;
            checkCudaError(cudaMalloc((void**) &ptr, size));
            m_storage[owner][var] = ptr; 
        }
    }
    else
    {
        std::cerr << "Failed to allocate " << var << " to " << owner << "\n";
        exit(1);
    }
}

void Fluid::destroy(std::string owner, std::string var, bool throw_err = true)
{
    if (m_storage[owner][var] != nullptr)
    {
        if (owner == "h")
        {
            free(m_storage[owner][var]);
        }
        else
        {
            checkCudaError(cudaFree(m_storage[owner][var]));
        }
        m_storage[owner][var] = nullptr;
    }
    else if (throw_err)
    {
        std::cerr << "Failed to free " << var << " on " << owner << "\n";
        exit(1);
    }
}

void Fluid::copy(std::string path, std::string var)
{
    if (m_storage["h"][var] != nullptr && m_storage["d"][var] != nullptr)
    {
        size_t size = m_storage_sizes[m_storage_types[var]];
        cudaMemcpyKind memcpyKind;
        if (path == "h2d")
        {
            memcpyKind = cudaMemcpyHostToDevice;
        }
        else if (path == "d2h")
        {
            memcpyKind = cudaMemcpyDeviceToHost;
        }
        else
        {
            std::cerr << "Invalid copy path provided\n";
            exit(1);
        }

        double *dest = m_storage[path.substr(2,1)][var];
        double *src  = m_storage[path.substr(0,1)][var];

        checkCudaError(cudaMemcpy((void*)dest, (void*)src, size*sizeof(double), memcpyKind));
    }
    else
    {
        std::cerr << "Failed to copy " << var << " from " << path[0] << " to " << path[2] << "\n";
        exit(1);
    }
}

__global__ void kern_memory_dump(size_t size, double *ptr)
{
    for (size_t i = 0; i < size; ++i)
    {
        ptr[i] = 1.0 * i;
        printf("%lu, %f\n", i, ptr[i]);
    }
}

void Fluid::memory_dump(std::string owner, std::string var)
{
    double *ptr = m_storage[owner][var];
    if (ptr != nullptr)
    {    
        std::cout << "Memory dump BEGIN: " << var << " on " << owner << "\n";
        if (owner == "h")
        {
            for (size_t i = 0; i < m_storage_sizes[m_storage_types[var]]; ++i)
            {
                std::cout << i << ", " << ptr[i] << "\n";
            }
        }
        else if (owner == "d")
        {
            kern_memory_dump<<<1, 1>>>(m_storage_sizes[m_storage_types[var]], m_storage[owner][var]);
            checkCudaError(cudaDeviceSynchronize());
        }
        std::cout << "Memory dump END\n";
    }
    else
    {
        std::cerr << "Memory dump failed : " << var << " is not allocated on " << owner << "\n";
    }
}

void Fluid::memory_summary()
{
    size_t col_width = 20;

    std::cout << std::setw(col_width) << "Var";
    std::cout << std::setw(col_width) << "Size";
    std::cout << std::setw(col_width) << "Host"; 
    std::cout << std::setw(col_width) << "Device\n";
    for (int i = 0; i < 40; ++i)
        std::cout << "--";
    std::cout << "\n";

    for (auto &v : m_vars)
    {   
        std::cout << std::setw(col_width) << v;
        std::cout << std::setw(col_width) << m_storage_sizes[m_storage_types[v]];
        std::cout << std::setw(col_width) << ptr2str(m_storage["h"][v]);
        std::cout << std::setw(col_width) << ptr2str(m_storage["d"][v]);
        std::cout << "\n\n";
    }
}