#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        hipError_t err__ = (call);                                                      \
        if (err__ != hipSuccess) {                                                      \
            throw std::runtime_error(std::string("CUDA error: ") + hipGetErrorString(err__)); \
        }                                                                                \
    } while (0)

namespace param {
constexpr int n_steps = 200000;
constexpr double dt = 60.0;
constexpr double eps = 1e-3;
constexpr double G = 6.674e-11;
constexpr double planet_radius = 1e7;
constexpr double missile_speed = 1e6;
inline double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<int>& is_device) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    is_device.resize(n);
    std::string type;
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type;
        is_device[i] = (type == "device") ? 1 : 0;
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

struct PlanetAsteroidStats {
    int planet_idx;
    int asteroid_idx;
    int track_min;
    int track_collision;
    double planet_radius_sq;
    double min_dist_sq;
    int hit_step;
};

struct DeviceScenarioStats {
    int planet_idx;
    int device_idx;
    int mass_zeroed;
    int destroyed_step;
    double missile_speed;
    double dt;
};

struct Problem3Result {
    int best_device;
    int best_hit_step;
    double best_cost;
};

template <int ILP>
__global__ void integrate_kernel(int n, double* qx, double* qy, double* qz, double* vx,
    double* vy, double* vz, double* m, const int* is_device, double current_time,
    double eps, double G, double dt, double device_mass_scale,
    PlanetAsteroidStats* distance_stats, DeviceScenarioStats* device_stats, int step) {
    // ... (Keep existing ILP kernel code) ...
    // Note: Since I cannot use "Keep existing code" in replacement, I must paste the full ILP kernel here again or use a separate tool call.
    // To be safe and correct, I will paste the full ILP kernel code I wrote in the previous step, 
    // AND add the new kernel below it.
    
    extern __shared__ unsigned char shared_mem[];
    double* s_qx = reinterpret_cast<double*>(shared_mem);
    double* s_qy = s_qx + blockDim.x;
    double* s_qz = s_qy + blockDim.x;
    double* s_m = s_qz + blockDim.x;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i0 = (bid * ILP) * blockDim.x + tid;

    double r_qx[ILP], r_qy[ILP], r_qz[ILP];
    double r_vx[ILP], r_vy[ILP], r_vz[ILP];
    double r_ax[ILP] = {0}, r_ay[ILP] = {0}, r_az[ILP] = {0};
    bool active[ILP];

    for (int k = 0; k < ILP; ++k) {
        int idx = i0 + k * blockDim.x;
        active[k] = (idx < n);
        if (active[k]) {
            r_qx[k] = qx[idx];
            r_qy[k] = qy[idx];
            r_qz[k] = qz[idx];
            r_vx[k] = vx[idx];
            r_vy[k] = vy[idx];
            r_vz[k] = vz[idx];
        }
    }
    
    int stats_step = step - 1;
    if (stats_step >= 0) {
        for (int k = 0; k < ILP; ++k) {
            int idx = i0 + k * blockDim.x;
            if (!active[k]) continue;

            if (idx == 0 && distance_stats) {
                int p_idx = distance_stats->planet_idx;
                int a_idx = distance_stats->asteroid_idx;
                double dx = qx[p_idx] - qx[a_idx];
                double dy = qy[p_idx] - qy[a_idx];
                double dz = qz[p_idx] - qz[a_idx];
                double dist_sq = dx * dx + dy * dy + dz * dz;
                if (distance_stats->track_min && dist_sq < distance_stats->min_dist_sq) {
                    distance_stats->min_dist_sq = dist_sq;
                }
                if (distance_stats->track_collision && distance_stats->hit_step == -1 && dist_sq < distance_stats->planet_radius_sq) {
                    distance_stats->hit_step = stats_step;
                }
            }

            if (device_stats && idx == device_stats->device_idx) {
                 int p_idx = device_stats->planet_idx;
                 double dx = qx[p_idx] - r_qx[k];
                 double dy = qy[p_idx] - r_qy[k];
                 double dz = qz[p_idx] - r_qz[k];
                 double dist_sq = dx * dx + dy * dy + dz * dz;
                 
                 if (device_stats->destroyed_step == -1) {
                     double missile_dist = stats_step * device_stats->dt * device_stats->missile_speed;
                     if (missile_dist * missile_dist > dist_sq) {
                         device_stats->destroyed_step = stats_step;
                     }
                 }
                 if (device_stats->destroyed_step != -1 && !device_stats->mass_zeroed) {
                     m[idx] = 0.0;
                     device_stats->mass_zeroed = 1;
                 }
            }
        }
    }

    for (int tile = 0; tile < n; tile += blockDim.x) {
        int load_idx = tile + tid;
        if (load_idx < n) {
            s_qx[tid] = qx[load_idx];
            s_qy[tid] = qy[load_idx];
            s_qz[tid] = qz[load_idx];
            double val_m = m[load_idx];
            if (is_device[load_idx]) val_m *= device_mass_scale;
            s_m[tid] = val_m;
        } else {
            s_qx[tid] = 0.0; s_qy[tid] = 0.0; s_qz[tid] = 0.0; s_m[tid] = 0.0;
        }
        __syncthreads();

        int tile_count = (tile + blockDim.x > n) ? (n - tile) : blockDim.x;

        #pragma unroll 16
        for (int j = 0; j < tile_count; ++j) {
            double sj_qx = s_qx[j];
            double sj_qy = s_qy[j];
            double sj_qz = s_qz[j];
            double sj_m  = s_m[j];

            for (int k = 0; k < ILP; ++k) {
                if (!active[k]) continue;
                
                double dx = sj_qx - r_qx[k];
                double dy = sj_qy - r_qy[k];
                double dz = sj_qz - r_qz[k];
                double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
                double inv_dist = rsqrt(dist2);
                double inv_dist3 = inv_dist * inv_dist * inv_dist;
                double scale = G * sj_m * inv_dist3;
                
                r_ax[k] += scale * dx;
                r_ay[k] += scale * dy;
                r_az[k] += scale * dz;
            }
        }
        __syncthreads();
    }

    for (int k = 0; k < ILP; ++k) {
        if (active[k]) {
            int idx = i0 + k * blockDim.x;
            r_vx[k] += r_ax[k] * dt;
            r_vy[k] += r_ay[k] * dt;
            r_vz[k] += r_az[k] * dt;
            r_qx[k] += r_vx[k] * dt;
            r_qy[k] += r_vy[k] * dt;
            r_qz[k] += r_vz[k] * dt;

            vx[idx] = r_vx[k];
            vy[idx] = r_vy[k];
            vz[idx] = r_vz[k];
            qx[idx] = r_qx[k];
            qy[idx] = r_qy[k];
            qz[idx] = r_qz[k];
        }
    }
}

__global__ void integrate_kernel_all_shared(int n, double* qx, double* qy, double* qz, double* vx,
    double* vy, double* vz, double* m, const int* is_device, double current_time,
    double eps, double G, double dt, double device_mass_scale,
    PlanetAsteroidStats* distance_stats, DeviceScenarioStats* device_stats, int step) {
    
    // Max N=1024. Shared memory needed: 1024 * 4 * 8 bytes = 32KB.
    __shared__ double s_qx[1024];
    __shared__ double s_qy[1024];
    __shared__ double s_qz[1024];
    __shared__ double s_m[1024];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Cooperative Load: All threads load the full dataset
    for (int k = tid; k < n; k += blockDim.x) {
        s_qx[k] = qx[k];
        s_qy[k] = qy[k];
        s_qz[k] = qz[k];
        double val_m = m[k];
        if (is_device[k]) val_m *= device_mass_scale;
        s_m[k] = val_m;
    }
    __syncthreads();

    if (i >= n) return;

    double qxi = s_qx[i];
    double qyi = s_qy[i];
    double qzi = s_qz[i];
    double vxi = vx[i];
    double vyi = vy[i];
    double vzi = vz[i];
    double axi = 0.0;
    double ayi = 0.0;
    double azi = 0.0;

    // Stats Logic (Simplified for N=1024 case)
    int stats_step = step - 1;
    if (stats_step >= 0) {
        if (i == 0 && distance_stats) {
            int p_idx = distance_stats->planet_idx;
            int a_idx = distance_stats->asteroid_idx;
            // Use shared memory for consistency within this step
            double dx = s_qx[p_idx] - s_qx[a_idx];
            double dy = s_qy[p_idx] - s_qy[a_idx];
            double dz = s_qz[p_idx] - s_qz[a_idx];
            double dist_sq = dx * dx + dy * dy + dz * dz;
            if (distance_stats->track_min && dist_sq < distance_stats->min_dist_sq) {
                distance_stats->min_dist_sq = dist_sq;
            }
            if (distance_stats->track_collision && distance_stats->hit_step == -1 && dist_sq < distance_stats->planet_radius_sq) {
                distance_stats->hit_step = stats_step;
            }
        }
        if (device_stats && i == device_stats->device_idx) {
             int p_idx = device_stats->planet_idx;
             double dx = s_qx[p_idx] - qxi;
             double dy = s_qy[p_idx] - qyi;
             double dz = s_qz[p_idx] - qzi;
             double dist_sq = dx * dx + dy * dy + dz * dz;
             if (device_stats->destroyed_step == -1) {
                 double missile_dist = stats_step * device_stats->dt * device_stats->missile_speed;
                 if (missile_dist * missile_dist > dist_sq) {
                     device_stats->destroyed_step = stats_step;
                 }
             }
             if (device_stats->destroyed_step != -1 && !device_stats->mass_zeroed) {
                 m[i] = 0.0;
                 device_stats->mass_zeroed = 1;
             }
        }
    }

    #pragma unroll 32
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        double dx = s_qx[j] - qxi;
        double dy = s_qy[j] - qyi;
        double dz = s_qz[j] - qzi;
        double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
        double inv_dist = rsqrt(dist2);
        double inv_dist3 = inv_dist * inv_dist * inv_dist;
        double scale = G * s_m[j] * inv_dist3;
        axi += scale * dx;
        ayi += scale * dy;
        azi += scale * dz;
    }

    vxi += axi * dt;
    vyi += ayi * dt;
    vzi += azi * dt;
    qxi += vxi * dt;
    qyi += vyi * dt;
    qzi += vzi * dt;

    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;
    qx[i] = qxi;
    qy[i] = qyi;
    qz[i] = qzi;
}

class NBodySimulatorGPU {
public:
    NBodySimulatorGPU(int n, const std::vector<int>& is_device) : n_(n) {
        CUDA_CHECK(hipMalloc(&d_qx_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_qy_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_qz_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_vx_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_vy_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_vz_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_m_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_is_device_, n_ * sizeof(int)));
        CUDA_CHECK(hipMemcpy(d_is_device_, is_device.data(), n_ * sizeof(int),
            hipMemcpyHostToDevice));
    }
    // ... (Destructor and other methods same as before) ...
    ~NBodySimulatorGPU() {
        hipFree(d_qx_); hipFree(d_qy_); hipFree(d_qz_);
        hipFree(d_vx_); hipFree(d_vy_); hipFree(d_vz_);
        hipFree(d_m_); hipFree(d_is_device_);
        hipFree(d_qx_backup_); hipFree(d_qy_backup_); hipFree(d_qz_backup_);
        hipFree(d_vx_backup_); hipFree(d_vy_backup_); hipFree(d_vz_backup_);
        hipFree(d_m_backup_);
    }

    void set_state(const std::vector<double>& qx, const std::vector<double>& qy,
        const std::vector<double>& qz, const std::vector<double>& vx,
        const std::vector<double>& vy, const std::vector<double>& vz,
        const std::vector<double>& m) {
        CUDA_CHECK(hipMemcpy(d_qx_, qx.data(), n_ * sizeof(double), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_qy_, qy.data(), n_ * sizeof(double), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_qz_, qz.data(), n_ * sizeof(double), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_vx_, vx.data(), n_ * sizeof(double), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_vy_, vy.data(), n_ * sizeof(double), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_vz_, vz.data(), n_ * sizeof(double), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_m_, m.data(), n_ * sizeof(double), hipMemcpyHostToDevice));
        has_backup_ = false;
    }

    void run_step(int step, PlanetAsteroidStats* distance_stats = nullptr,
        DeviceScenarioStats* device_stats = nullptr, hipStream_t stream = 0) {
        double current_time = step * param::dt;
        double device_mass_scale = 1.0 + 0.5 * fabs(sin(current_time / 6000.0));
        
        if (n_ <= 1024) {
             // All-Shared-Memory Path
             int threads = 64; // Use smaller blocks to get more blocks on GPU
             int blocks = (n_ + threads - 1) / threads;
             // No dynamic shared mem needed, it's static in kernel
             integrate_kernel_all_shared<<<blocks, threads, 0, stream>>>(
                n_, d_qx_, d_qy_, d_qz_, d_vx_, d_vy_, d_vz_, d_m_, d_is_device_,
                current_time, param::eps, param::G, param::dt, device_mass_scale,
                distance_stats, device_stats, step);
             CUDA_CHECK(hipGetLastError());
             
             if (step == param::n_steps) {
                integrate_kernel_all_shared<<<blocks, threads, 0, stream>>>(
                    n_, d_qx_, d_qy_, d_qz_, d_vx_, d_vy_, d_vz_, d_m_, d_is_device_,
                    current_time, param::eps, param::G, 0.0, device_mass_scale,
                    distance_stats, device_stats, step + 1);
                CUDA_CHECK(hipGetLastError());
             }
        } else {
            // ILP Path for larger N
            int threads = 256;
            constexpr int ILP = 4;
            int blocks = (n_ + (threads * ILP) - 1) / (threads * ILP);
            size_t shared_bytes = threads * (4 * sizeof(double));
            
            integrate_kernel<ILP><<<blocks, threads, shared_bytes, stream>>>(
                n_, d_qx_, d_qy_, d_qz_, d_vx_, d_vy_, d_vz_, d_m_, d_is_device_,
                current_time, param::eps, param::G, param::dt, device_mass_scale,
                distance_stats, device_stats, step);
            CUDA_CHECK(hipGetLastError());
            
            if (step == param::n_steps) {
                integrate_kernel<ILP><<<blocks, threads, shared_bytes, stream>>>(
                    n_, d_qx_, d_qy_, d_qz_, d_vx_, d_vy_, d_vz_, d_m_, d_is_device_,
                    current_time, param::eps, param::G, 0.0, device_mass_scale,
                    distance_stats, device_stats, step + 1);
                CUDA_CHECK(hipGetLastError());
            }
        }
    }

    void record_distance_stats(PlanetAsteroidStats* stats, int step, hipStream_t stream = 0) {
        double current_time = step * param::dt;
        double device_mass_scale = 1.0 + 0.5 * fabs(sin(current_time / 6000.0));
        
        if (n_ <= 1024) {
             int threads = 64;
             int blocks = (n_ + threads - 1) / threads;
             integrate_kernel_all_shared<<<blocks, threads, 0, stream>>>(
                n_, d_qx_, d_qy_, d_qz_, d_vx_, d_vy_, d_vz_, d_m_, d_is_device_,
                current_time, param::eps, param::G, 0.0, device_mass_scale,
                stats, nullptr, step + 1);
             CUDA_CHECK(hipGetLastError());
        } else {
            int threads = 256;
            constexpr int ILP = 4;
            int blocks = (n_ + (threads * ILP) - 1) / (threads * ILP);
            size_t shared_bytes = threads * (4 * sizeof(double));
            
            integrate_kernel<ILP><<<blocks, threads, shared_bytes, stream>>>(
                n_, d_qx_, d_qy_, d_qz_, d_vx_, d_vy_, d_vz_, d_m_, d_is_device_,
                current_time, param::eps, param::G, 0.0, device_mass_scale,
                stats, nullptr, step + 1);
            CUDA_CHECK(hipGetLastError());
        }
    }

    void backup_state() {
        allocate_backups();
        copy_state(d_qx_backup_, d_qy_backup_, d_qz_backup_, d_vx_backup_, d_vy_backup_,
            d_vz_backup_, d_m_backup_, hipMemcpyDeviceToDevice);
        has_backup_ = true;
    }

    void restore_from_backup() {
        if (!has_backup_) {
            throw std::runtime_error("Simulation backup requested before initialization");
        }
        copy_state(d_qx_, d_qy_, d_qz_, d_vx_, d_vy_, d_vz_, d_m_, hipMemcpyDeviceToDevice,
            true);
    }

private:
    void allocate_backups() {
        if (d_qx_backup_) {
            return;
        }
        CUDA_CHECK(hipMalloc(&d_qx_backup_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_qy_backup_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_qz_backup_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_vx_backup_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_vy_backup_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_vz_backup_, n_ * sizeof(double)));
        CUDA_CHECK(hipMalloc(&d_m_backup_, n_ * sizeof(double)));
    }

    void copy_state(double* qx_dst, double* qy_dst, double* qz_dst, double* vx_dst,
        double* vy_dst, double* vz_dst, double* m_dst, hipMemcpyKind kind,
        bool from_backup = false) {
        const double* qx_src = from_backup ? d_qx_backup_ : d_qx_;
        const double* qy_src = from_backup ? d_qy_backup_ : d_qy_;
        const double* qz_src = from_backup ? d_qz_backup_ : d_qz_;
        const double* vx_src = from_backup ? d_vx_backup_ : d_vx_;
        const double* vy_src = from_backup ? d_vy_backup_ : d_vy_;
        const double* vz_src = from_backup ? d_vz_backup_ : d_vz_;
        const double* m_src = from_backup ? d_m_backup_ : d_m_;
        CUDA_CHECK(hipMemcpy(qx_dst, qx_src, n_ * sizeof(double), kind));
        CUDA_CHECK(hipMemcpy(qy_dst, qy_src, n_ * sizeof(double), kind));
        CUDA_CHECK(hipMemcpy(qz_dst, qz_src, n_ * sizeof(double), kind));
        CUDA_CHECK(hipMemcpy(vx_dst, vx_src, n_ * sizeof(double), kind));
        CUDA_CHECK(hipMemcpy(vy_dst, vy_src, n_ * sizeof(double), kind));
        CUDA_CHECK(hipMemcpy(vz_dst, vz_src, n_ * sizeof(double), kind));
        CUDA_CHECK(hipMemcpy(m_dst, m_src, n_ * sizeof(double), kind));
    }

    int n_;
    double *d_qx_{nullptr};
    double *d_qy_{nullptr};
    double *d_qz_{nullptr};
    double *d_vx_{nullptr};
    double *d_vy_{nullptr};
    double *d_vz_{nullptr};
    double *d_m_{nullptr};
    int *d_is_device_{nullptr};
    double *d_qx_backup_{nullptr};
    double *d_qy_backup_{nullptr};
    double *d_qz_backup_{nullptr};
    double *d_vx_backup_{nullptr};
    double *d_vy_backup_{nullptr};
    double *d_vz_backup_{nullptr};
    double *d_m_backup_{nullptr};
    bool has_backup_{false};
};

Problem3Result run_problem3_subset(NBodySimulatorGPU& simulator,
    const std::vector<int>& device_subset, int planet, int asteroid,
    const std::vector<double>& base_qx, const std::vector<double>& base_qy,
    const std::vector<double>& base_qz, const std::vector<double>& base_vx,
    const std::vector<double>& base_vy, const std::vector<double>& base_vz,
    const std::vector<double>& base_m, double planet_radius_sq, float& elapsed_ms) {
    Problem3Result result{-1, param::n_steps + 1, 0.0};
    if (device_subset.empty()) {
        elapsed_ms = 0.0f;
        return result;
    }

    simulator.set_state(base_qx, base_qy, base_qz, base_vx, base_vy, base_vz, base_m);
    simulator.backup_state();

    PlanetAsteroidStats* d_problem3_stats = nullptr;
    DeviceScenarioStats* d_device_stats = nullptr;
    CUDA_CHECK(hipMalloc(&d_problem3_stats, sizeof(PlanetAsteroidStats)));
    CUDA_CHECK(hipMalloc(&d_device_stats, sizeof(DeviceScenarioStats)));
    PlanetAsteroidStats* h_problem3_stats_pinned = nullptr;
    CUDA_CHECK(hipHostMalloc(&h_problem3_stats_pinned, sizeof(PlanetAsteroidStats)));

    hipStream_t stream;
    CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipEvent_t start, stop;
    CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(hipEventCreate(&stop));
    CUDA_CHECK(hipEventRecord(start, stream));

    for (int device_idx : device_subset) {
        simulator.restore_from_backup();
        PlanetAsteroidStats host_problem3_stats{};
        host_problem3_stats.planet_idx = planet;
        host_problem3_stats.asteroid_idx = asteroid;
        host_problem3_stats.track_min = 0;
        host_problem3_stats.track_collision = 1;
        host_problem3_stats.planet_radius_sq = planet_radius_sq;
        host_problem3_stats.min_dist_sq = std::numeric_limits<double>::infinity();
        host_problem3_stats.hit_step = -1;
        CUDA_CHECK(hipMemcpy(d_problem3_stats, &host_problem3_stats,
            sizeof(PlanetAsteroidStats), hipMemcpyHostToDevice));
        DeviceScenarioStats host_device_stats{};
        host_device_stats.planet_idx = planet;
        host_device_stats.device_idx = device_idx;
        host_device_stats.mass_zeroed = 0;
        host_device_stats.destroyed_step = -1;
        host_device_stats.missile_speed = param::missile_speed;
        host_device_stats.dt = param::dt;
        CUDA_CHECK(hipMemcpy(d_device_stats, &host_device_stats, sizeof(DeviceScenarioStats),
            hipMemcpyHostToDevice));

        simulator.record_distance_stats(d_problem3_stats, 0, stream);
        for (int step = 1; step <= param::n_steps; ++step) {
            simulator.run_step(step, d_problem3_stats, d_device_stats, stream);
            if (step % 100 == 0) {
                CUDA_CHECK(hipMemcpyAsync(h_problem3_stats_pinned, d_problem3_stats,
                    sizeof(PlanetAsteroidStats), hipMemcpyDeviceToHost, stream));
                CUDA_CHECK(hipStreamSynchronize(stream));
                if (h_problem3_stats_pinned->hit_step != -1) {
                    break;
                }
            }
        }
        CUDA_CHECK(hipStreamSynchronize(stream));
        CUDA_CHECK(hipMemcpy(h_problem3_stats_pinned, d_problem3_stats,
            sizeof(PlanetAsteroidStats), hipMemcpyDeviceToHost));
        DeviceScenarioStats final_device_stats{};
        CUDA_CHECK(hipMemcpy(&final_device_stats, d_device_stats,
            sizeof(DeviceScenarioStats), hipMemcpyDeviceToHost));

        bool destroyed = final_device_stats.destroyed_step != -1;
        bool safe = (h_problem3_stats_pinned->hit_step == -1);

        if (safe && destroyed) {
            if (final_device_stats.destroyed_step < result.best_hit_step ||
                (final_device_stats.destroyed_step == result.best_hit_step &&
                    (result.best_device == -1 || device_idx < result.best_device))) {
                result.best_device = device_idx;
                result.best_hit_step = final_device_stats.destroyed_step;
                result.best_cost = param::get_missile_cost(
                    final_device_stats.destroyed_step * param::dt);
            }
        }
    }

    CUDA_CHECK(hipEventRecord(stop, stream));
    CUDA_CHECK(hipEventSynchronize(stop));
    CUDA_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream);
    hipFree(d_problem3_stats);
    hipFree(d_device_stats);
    hipHostFree(h_problem3_stats_pinned);
    return result;
}

double solve_problem1(NBodySimulatorGPU& simulator, int n, int planet, int asteroid,
    const std::vector<int>& is_device, const std::vector<double>& base_qx,
    const std::vector<double>& base_qy, const std::vector<double>& base_qz,
    const std::vector<double>& base_vx, const std::vector<double>& base_vy,
    const std::vector<double>& base_vz, const std::vector<double>& base_m,
    float& elapsed_ms) {
    std::vector<double> qx = base_qx;
    std::vector<double> qy = base_qy;
    std::vector<double> qz = base_qz;
    std::vector<double> vx = base_vx;
    std::vector<double> vy = base_vy;
    std::vector<double> vz = base_vz;
    std::vector<double> m = base_m;
    for (int i = 0; i < n; ++i) {
        if (is_device[i]) {
            m[i] = 0.0;
        }
    }
    simulator.set_state(qx, qy, qz, vx, vy, vz, m);

    PlanetAsteroidStats host_stats{};
    host_stats.planet_idx = planet;
    host_stats.asteroid_idx = asteroid;
    host_stats.track_min = 1;
    host_stats.track_collision = 0;
    host_stats.planet_radius_sq = 0.0;
    host_stats.min_dist_sq = std::numeric_limits<double>::infinity();
    host_stats.hit_step = -1;

    PlanetAsteroidStats* d_stats = nullptr;
    CUDA_CHECK(hipMalloc(&d_stats, sizeof(PlanetAsteroidStats)));
    CUDA_CHECK(hipMemcpy(d_stats, &host_stats, sizeof(PlanetAsteroidStats),
        hipMemcpyHostToDevice));
    hipEvent_t start, stop;
    CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(hipEventCreate(&stop));
    CUDA_CHECK(hipEventRecord(start));
    simulator.record_distance_stats(d_stats, 0);
    for (int step = 1; step <= param::n_steps; ++step) {
        simulator.run_step(step, d_stats);
    }
    CUDA_CHECK(hipDeviceSynchronize());
    CUDA_CHECK(hipEventRecord(stop));
    CUDA_CHECK(hipEventSynchronize(stop));
    CUDA_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(hipMemcpy(&host_stats, d_stats, sizeof(PlanetAsteroidStats),
        hipMemcpyDeviceToHost));
    hipFree(d_stats);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return std::sqrt(host_stats.min_dist_sq);
}

int solve_problem2(NBodySimulatorGPU& simulator, int planet, int asteroid,
    double planet_radius_sq, const std::vector<double>& base_qx,
    const std::vector<double>& base_qy, const std::vector<double>& base_qz,
    const std::vector<double>& base_vx, const std::vector<double>& base_vy,
    const std::vector<double>& base_vz, const std::vector<double>& base_m,
    float& elapsed_ms) {
    std::vector<double> qx = base_qx;
    std::vector<double> qy = base_qy;
    std::vector<double> qz = base_qz;
    std::vector<double> vx = base_vx;
    std::vector<double> vy = base_vy;
    std::vector<double> vz = base_vz;
    std::vector<double> m = base_m;
    simulator.set_state(qx, qy, qz, vx, vy, vz, m);

    PlanetAsteroidStats host_stats{};
    host_stats.planet_idx = planet;
    host_stats.asteroid_idx = asteroid;
    host_stats.track_min = 0;
    host_stats.track_collision = 1;
    host_stats.planet_radius_sq = planet_radius_sq;
    host_stats.min_dist_sq = std::numeric_limits<double>::infinity();
    host_stats.hit_step = -1;
    PlanetAsteroidStats* d_stats = nullptr;
    CUDA_CHECK(hipMalloc(&d_stats, sizeof(PlanetAsteroidStats)));
    CUDA_CHECK(hipMemcpy(d_stats, &host_stats, sizeof(PlanetAsteroidStats),
        hipMemcpyHostToDevice));
    hipEvent_t start, stop;
    CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(hipEventCreate(&stop));
    CUDA_CHECK(hipEventRecord(start));
    simulator.record_distance_stats(d_stats, 0);
    for (int step = 1; step <= param::n_steps; ++step) {
        simulator.run_step(step, d_stats);
    }
    CUDA_CHECK(hipDeviceSynchronize());
    CUDA_CHECK(hipEventRecord(stop));
    CUDA_CHECK(hipEventSynchronize(stop));
    CUDA_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(hipMemcpy(&host_stats, d_stats, sizeof(PlanetAsteroidStats),
        hipMemcpyDeviceToHost));
    hipFree(d_stats);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return host_stats.hit_step;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> is_device;

    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);

    const std::vector<double> base_qx = qx;
    const std::vector<double> base_qy = qy;
    const std::vector<double> base_qz = qz;
    const std::vector<double> base_vx = vx;
    const std::vector<double> base_vy = vy;
    const std::vector<double> base_vz = vz;
    const std::vector<double> base_m = m;

    int device_count = 0;
    CUDA_CHECK(hipGetDeviceCount(&device_count));
    if (device_count < 2) {
        throw std::runtime_error("This refactored version requires at least 2 CUDA devices");
    }
    printf("Detected %d CUDA devices; running with GPUs 0 and 1\n", device_count);

    const double planet_radius_sq = param::planet_radius * param::planet_radius;

    std::vector<int> devices;
    for (int i = 0; i < n; i++) {
        if (is_device[i]) {
            devices.push_back(i);
        }
    }
    size_t mid = (devices.size() + 1) / 2;
    std::vector<int> first_devices(devices.begin(), devices.begin() + mid);
    std::vector<int> second_devices(devices.begin() + mid, devices.end());

    double min_dist = 0.0;
    int hit_time_step = -1;
    Problem3Result result_gpu0{-1, param::n_steps + 1, 0.0};
    Problem3Result result_gpu1{-1, param::n_steps + 1, 0.0};
    float problem1_ms = 0.0f;
    float problem2_ms = 0.0f;
    float problem3_ms_gpu0 = 0.0f;
    float problem3_ms_gpu1 = 0.0f;

#pragma omp parallel sections shared(min_dist, hit_time_step, result_gpu0, result_gpu1, problem1_ms, problem2_ms, problem3_ms_gpu0, problem3_ms_gpu1)
    {
#pragma omp section
        {
            CUDA_CHECK(hipSetDevice(0));
            NBodySimulatorGPU simulator(n, is_device);
            min_dist = solve_problem1(simulator, n, planet, asteroid, is_device, base_qx,
                base_qy, base_qz, base_vx, base_vy, base_vz, base_m, problem1_ms);
            if (!first_devices.empty()) {
                result_gpu0 = run_problem3_subset(simulator, first_devices, planet, asteroid,
                    base_qx, base_qy, base_qz, base_vx, base_vy, base_vz, base_m,
                    planet_radius_sq, problem3_ms_gpu0);
            } else {
                problem3_ms_gpu0 = 0.0f;
            }
        }
#pragma omp section
        {
            CUDA_CHECK(hipSetDevice(1));
            NBodySimulatorGPU simulator(n, is_device);
            hit_time_step = solve_problem2(simulator, planet, asteroid, planet_radius_sq,
                base_qx, base_qy, base_qz, base_vx, base_vy, base_vz, base_m, problem2_ms);
            if (!second_devices.empty()) {
                result_gpu1 = run_problem3_subset(simulator, second_devices, planet, asteroid,
                    base_qx, base_qy, base_qz, base_vx, base_vy, base_vz, base_m,
                    planet_radius_sq, problem3_ms_gpu1);
            } else {
                problem3_ms_gpu1 = 0.0f;
            }
        }
    }

    auto pick_best = [](const Problem3Result& best, const Problem3Result& candidate) {
        Problem3Result out = best;
        if (candidate.best_device == -1) {
            return out;
        }
        if (out.best_device == -1 ||
            candidate.best_hit_step < out.best_hit_step ||
            (candidate.best_hit_step == out.best_hit_step &&
                candidate.best_device < out.best_device)) {
            out = candidate;
        }
        return out;
    };
    Problem3Result global_result{-1, param::n_steps + 1, 0.0};
    global_result = pick_best(global_result, result_gpu0);
    global_result = pick_best(global_result, result_gpu1);

    int gravity_device_id = -1;
    double missile_cost = 0.0;
    if (global_result.best_device != -1) {
        gravity_device_id = global_result.best_device;
        missile_cost = global_result.best_cost;
    }

    float problem3_ms = std::max(problem3_ms_gpu0, problem3_ms_gpu1);
    printf("Problem 1 kernel time: %.3f ms\n", problem1_ms);
    printf("Problem 2 kernel time: %.3f ms\n", problem2_ms);
    printf("Problem 3 kernel time: %.3f ms\n", problem3_ms);

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    printf("Elapsed Time: %.6f s\n", elapsed_seconds.count());
}
