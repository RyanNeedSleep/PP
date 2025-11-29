#include "hip/hip_runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <fstream>
#include <iomanip>
#include <limits>
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
__host__ __device__ inline double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000.0));
}
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

__global__ void integrate_kernel(int n, double* qx, double* qy, double* qz, double* vx,
    double* vy, double* vz, double* m, const int* is_device, double current_time,
    double eps, double G, double dt) {
    extern __shared__ unsigned char shared_mem[];
    double* s_qx = reinterpret_cast<double*>(shared_mem);
    double* s_qy = s_qx + blockDim.x;
    double* s_qz = s_qy + blockDim.x;
    double* s_m = s_qz + blockDim.x;
    int* s_is_device = reinterpret_cast<int*>(s_m + blockDim.x);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double qxi = qx[i];
    double qyi = qy[i];
    double qzi = qz[i];
    double vxi = vx[i];
    double vyi = vy[i];
    double vzi = vz[i];
    double axi = 0.0;
    double ayi = 0.0;
    double azi = 0.0;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        int load_idx = tile + threadIdx.x;
        if (load_idx < n) {
            s_qx[threadIdx.x] = qx[load_idx];
            s_qy[threadIdx.x] = qy[load_idx];
            s_qz[threadIdx.x] = qz[load_idx];
            s_m[threadIdx.x] = m[load_idx];
            s_is_device[threadIdx.x] = is_device[load_idx];
        } else {
            s_qx[threadIdx.x] = 0.0;
            s_qy[threadIdx.x] = 0.0;
            s_qz[threadIdx.x] = 0.0;
            s_m[threadIdx.x] = 0.0;
            s_is_device[threadIdx.x] = 0;
        }
        __syncthreads();

        int tile_count = blockDim.x;
        if (tile + tile_count > n) {
            tile_count = n - tile;
        }
        for (int j = 0; j < tile_count; ++j) {
            int partner_idx = tile + j;
            if (partner_idx == i) continue;
            double mj = s_m[j];
            if (s_is_device[j]) {
                mj = param::gravity_device_mass(mj, current_time);
            }
            double dx = s_qx[j] - qxi;
            double dy = s_qy[j] - qyi;
            double dz = s_qz[j] - qzi;
            double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
            double inv_dist = rsqrt(dist2);
            double inv_dist3 = inv_dist * inv_dist * inv_dist;
            double scale = G * mj * inv_dist3;
            axi += scale * dx;
            ayi += scale * dy;
            azi += scale * dz;
        }
        __syncthreads();
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

__global__ void record_planet_asteroid_stats_kernel(const double* qx, const double* qy,
    const double* qz, PlanetAsteroidStats* stats, int step) {
    PlanetAsteroidStats local = stats[0];
    double dx = qx[local.planet_idx] - qx[local.asteroid_idx];
    double dy = qy[local.planet_idx] - qy[local.asteroid_idx];
    double dz = qz[local.planet_idx] - qz[local.asteroid_idx];
    double dist_sq = dx * dx + dy * dy + dz * dz;
    if (local.track_min && dist_sq < local.min_dist_sq) {
        local.min_dist_sq = dist_sq;
    }
    if (local.track_collision && local.hit_step == -1 && dist_sq < local.planet_radius_sq) {
        local.hit_step = step;
    }
    stats[0] = local;
}

__global__ void update_device_scenario_kernel(const double* qx, const double* qy,
    const double* qz, double* m, DeviceScenarioStats* scenario, int step) {
    DeviceScenarioStats local = scenario[0];
    if (local.device_idx < 0) {
        scenario[0] = local;
        return;
    }

    double dx = qx[local.planet_idx] - qx[local.device_idx];
    double dy = qy[local.planet_idx] - qy[local.device_idx];
    double dz = qz[local.planet_idx] - qz[local.device_idx];
    double dist_sq = dx * dx + dy * dy + dz * dz;
    if (local.destroyed_step == -1) {
        double missile_dist = step * local.dt * local.missile_speed;
        double missile_dist_sq = missile_dist * missile_dist;
        if (missile_dist_sq > dist_sq) {
            local.destroyed_step = step;
        }
    }
    if (local.destroyed_step != -1 && !local.mass_zeroed) {
        m[local.device_idx] = 0.0;
        local.mass_zeroed = 1;
    }
    scenario[0] = local;
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

    ~NBodySimulatorGPU() {
        hipFree(d_qx_);
        hipFree(d_qy_);
        hipFree(d_qz_);
        hipFree(d_vx_);
        hipFree(d_vy_);
        hipFree(d_vz_);
        hipFree(d_m_);
        hipFree(d_is_device_);
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
    }

    void run_step(int step, PlanetAsteroidStats* distance_stats = nullptr,
        DeviceScenarioStats* device_stats = nullptr) {
        double current_time = step * param::dt;
        int threads = 256;
        int blocks = (n_ + threads - 1) / threads;
        size_t shared_bytes =
            threads * (4 * sizeof(double)) + threads * sizeof(int);
        integrate_kernel<<<blocks, threads, shared_bytes>>>(
            n_, d_qx_, d_qy_, d_qz_, d_vx_, d_vy_, d_vz_, d_m_, d_is_device_,
            current_time, param::eps, param::G, param::dt);
        CUDA_CHECK(hipGetLastError());
        if (distance_stats) {
            record_planet_asteroid_stats_kernel<<<1, 1>>>(
                d_qx_, d_qy_, d_qz_, distance_stats, step);
            CUDA_CHECK(hipGetLastError());
        }
        if (device_stats) {
            update_device_scenario_kernel<<<1, 1>>>(
                d_qx_, d_qy_, d_qz_, d_m_, device_stats, step);
            CUDA_CHECK(hipGetLastError());
        }
    }

    void record_distance_stats(PlanetAsteroidStats* stats, int step) {
        record_planet_asteroid_stats_kernel<<<1, 1>>>(d_qx_, d_qy_, d_qz_, stats, step);
        CUDA_CHECK(hipGetLastError());
    }

private:
    int n_;
    double *d_qx_{nullptr};
    double *d_qy_{nullptr};
    double *d_qz_{nullptr};
    double *d_vx_{nullptr};
    double *d_vy_{nullptr};
    double *d_vz_{nullptr};
    double *d_m_{nullptr};
    int *d_is_device_{nullptr};
};

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

    NBodySimulatorGPU simulator(n, is_device);

    auto reset_state = [&](std::vector<double>& tqx, std::vector<double>& tqy,
                            std::vector<double>& tqz, std::vector<double>& tvx,
                            std::vector<double>& tvy, std::vector<double>& tvz,
                            std::vector<double>& tm) {
        tqx = base_qx;
        tqy = base_qy;
        tqz = base_qz;
        tvx = base_vx;
        tvy = base_vy;
        tvz = base_vz;
        tm = base_m;
    };

    // Problem 1: disable gravity devices and track min distance on the GPU.
    reset_state(qx, qy, qz, vx, vy, vz, m);
    for (int i = 0; i < n; i++) {
        if (is_device[i]) {
            m[i] = 0.0;
        }
    }
    simulator.set_state(qx, qy, qz, vx, vy, vz, m);
    PlanetAsteroidStats host_problem1_stats{};
    host_problem1_stats.planet_idx = planet;
    host_problem1_stats.asteroid_idx = asteroid;
    host_problem1_stats.track_min = 1;
    host_problem1_stats.track_collision = 0;
    host_problem1_stats.planet_radius_sq = 0.0;
    host_problem1_stats.min_dist_sq = std::numeric_limits<double>::infinity();
    host_problem1_stats.hit_step = -1;
    PlanetAsteroidStats* d_problem1_stats = nullptr;
    CUDA_CHECK(hipMalloc(&d_problem1_stats, sizeof(PlanetAsteroidStats)));
    CUDA_CHECK(hipMemcpy(d_problem1_stats, &host_problem1_stats, sizeof(PlanetAsteroidStats),
        hipMemcpyHostToDevice));
    simulator.record_distance_stats(d_problem1_stats, 0);
    for (int step = 1; step <= param::n_steps; ++step) {
        simulator.run_step(step, d_problem1_stats);
    }
    CUDA_CHECK(hipDeviceSynchronize());
    CUDA_CHECK(hipMemcpy(&host_problem1_stats, d_problem1_stats, sizeof(PlanetAsteroidStats),
        hipMemcpyDeviceToHost));
    double min_dist = std::sqrt(host_problem1_stats.min_dist_sq);
    hipFree(d_problem1_stats);

    // Problem 2: detect the first collision step using GPU distances.
    reset_state(qx, qy, qz, vx, vy, vz, m);
    simulator.set_state(qx, qy, qz, vx, vy, vz, m);
    PlanetAsteroidStats host_problem2_stats{};
    host_problem2_stats.planet_idx = planet;
    host_problem2_stats.asteroid_idx = asteroid;
    host_problem2_stats.track_min = 0;
    host_problem2_stats.track_collision = 1;
    const double planet_radius_sq = param::planet_radius * param::planet_radius;
    host_problem2_stats.planet_radius_sq = planet_radius_sq;
    host_problem2_stats.min_dist_sq = std::numeric_limits<double>::infinity();
    host_problem2_stats.hit_step = -1;
    PlanetAsteroidStats* d_problem2_stats = nullptr;
    CUDA_CHECK(hipMalloc(&d_problem2_stats, sizeof(PlanetAsteroidStats)));
    CUDA_CHECK(hipMemcpy(d_problem2_stats, &host_problem2_stats, sizeof(PlanetAsteroidStats),
        hipMemcpyHostToDevice));
    simulator.record_distance_stats(d_problem2_stats, 0);
    for (int step = 1; step <= param::n_steps; ++step) {
        simulator.run_step(step, d_problem2_stats);
    }
    CUDA_CHECK(hipDeviceSynchronize());
    CUDA_CHECK(hipMemcpy(&host_problem2_stats, d_problem2_stats, sizeof(PlanetAsteroidStats),
        hipMemcpyDeviceToHost));
    int hit_time_step = host_problem2_stats.hit_step;
    hipFree(d_problem2_stats);

    // Problem 3: brute-force missile targeting with CUDA-updated trajectories.
    std::vector<int> devices;
    for (int i = 0; i < n; i++) {
        if (is_device[i]) {
            devices.push_back(i);
        }
    }

    int best_device = -1;
    int best_hit_step = param::n_steps + 1;
    double best_cost = 0.0;
    PlanetAsteroidStats* d_problem3_stats = nullptr;
    DeviceScenarioStats* d_device_stats = nullptr;
    CUDA_CHECK(hipMalloc(&d_problem3_stats, sizeof(PlanetAsteroidStats)));
    CUDA_CHECK(hipMalloc(&d_device_stats, sizeof(DeviceScenarioStats)));

    for (int device_idx : devices) {
        reset_state(qx, qy, qz, vx, vy, vz, m);
        simulator.set_state(qx, qy, qz, vx, vy, vz, m);
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

        simulator.record_distance_stats(d_problem3_stats, 0);
        for (int step = 1; step <= param::n_steps; ++step) {
            simulator.run_step(step, d_problem3_stats, d_device_stats);
        }
        CUDA_CHECK(hipDeviceSynchronize());
        CUDA_CHECK(hipMemcpy(&host_problem3_stats, d_problem3_stats,
            sizeof(PlanetAsteroidStats), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipMemcpy(&host_device_stats, d_device_stats, sizeof(DeviceScenarioStats),
            hipMemcpyDeviceToHost));

        bool destroyed = host_device_stats.destroyed_step != -1;
        bool safe = (host_problem3_stats.hit_step == -1);

        if (safe && destroyed) {
            if (host_device_stats.destroyed_step < best_hit_step ||
                (host_device_stats.destroyed_step == best_hit_step &&
                    (best_device == -1 || device_idx < best_device))) {
                best_device = device_idx;
                best_hit_step = host_device_stats.destroyed_step;
                best_cost = param::get_missile_cost(
                    host_device_stats.destroyed_step * param::dt);
            }
        }
    }
    hipFree(d_problem3_stats);
    hipFree(d_device_stats);

    int gravity_device_id = -1;
    double missile_cost = 0.0;
    if (best_device != -1) {
        gravity_device_id = best_device;
        missile_cost = best_cost;
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    printf("Elapsed Time: %.6f s\n", elapsed_seconds.count());
}
