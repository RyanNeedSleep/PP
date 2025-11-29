#include <cuda_runtime.h>
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
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
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

__global__ void integrate_kernel(int n, double* qx, double* qy, double* qz, double* vx,
    double* vy, double* vz, double* m, const int* is_device, double current_time,
    double eps, double G, double dt, double device_mass_scale) {
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
                mj *= device_mass_scale;
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
        CUDA_CHECK(cudaMalloc(&d_qx_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_qy_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_qz_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vx_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vy_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vz_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_m_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_is_device_, n_ * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_is_device_, is_device.data(), n_ * sizeof(int),
            cudaMemcpyHostToDevice));
    }

    ~NBodySimulatorGPU() {
        cudaFree(d_qx_);
        cudaFree(d_qy_);
        cudaFree(d_qz_);
        cudaFree(d_vx_);
        cudaFree(d_vy_);
        cudaFree(d_vz_);
        cudaFree(d_m_);
        cudaFree(d_is_device_);
        cudaFree(d_qx_backup_);
        cudaFree(d_qy_backup_);
        cudaFree(d_qz_backup_);
        cudaFree(d_vx_backup_);
        cudaFree(d_vy_backup_);
        cudaFree(d_vz_backup_);
        cudaFree(d_m_backup_);
    }

    void set_state(const std::vector<double>& qx, const std::vector<double>& qy,
        const std::vector<double>& qz, const std::vector<double>& vx,
        const std::vector<double>& vy, const std::vector<double>& vz,
        const std::vector<double>& m) {
        CUDA_CHECK(cudaMemcpy(d_qx_, qx.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qy_, qy.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qz_, qz.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx_, vx.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy_, vy.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vz_, vz.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m_, m.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        has_backup_ = false;
    }

    void run_step(int step, PlanetAsteroidStats* distance_stats = nullptr,
        DeviceScenarioStats* device_stats = nullptr, cudaStream_t stream = 0) {
        double current_time = step * param::dt;
        double device_mass_scale = 1.0 + 0.5 * fabs(sin(current_time / 6000.0));
        int threads = 256;
        int blocks = (n_ + threads - 1) / threads;
        size_t shared_bytes =
            threads * (4 * sizeof(double)) + threads * sizeof(int);
        integrate_kernel<<<blocks, threads, shared_bytes, stream>>>(
            n_, d_qx_, d_qy_, d_qz_, d_vx_, d_vy_, d_vz_, d_m_, d_is_device_,
            current_time, param::eps, param::G, param::dt, device_mass_scale);
        CUDA_CHECK(cudaGetLastError());
        if (distance_stats) {
            record_planet_asteroid_stats_kernel<<<1, 1, 0, stream>>>(
                d_qx_, d_qy_, d_qz_, distance_stats, step);
            CUDA_CHECK(cudaGetLastError());
        }
        if (device_stats) {
            update_device_scenario_kernel<<<1, 1, 0, stream>>>(
                d_qx_, d_qy_, d_qz_, d_m_, device_stats, step);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    void record_distance_stats(PlanetAsteroidStats* stats, int step, cudaStream_t stream = 0) {
        record_planet_asteroid_stats_kernel<<<1, 1, 0, stream>>>(
            d_qx_, d_qy_, d_qz_, stats, step);
        CUDA_CHECK(cudaGetLastError());
    }

    void backup_state() {
        allocate_backups();
        copy_state(d_qx_backup_, d_qy_backup_, d_qz_backup_, d_vx_backup_, d_vy_backup_,
            d_vz_backup_, d_m_backup_, cudaMemcpyDeviceToDevice);
        has_backup_ = true;
    }

    void restore_from_backup() {
        if (!has_backup_) {
            throw std::runtime_error("Simulation backup requested before initialization");
        }
        copy_state(d_qx_, d_qy_, d_qz_, d_vx_, d_vy_, d_vz_, d_m_, cudaMemcpyDeviceToDevice,
            true);
    }

private:
    void allocate_backups() {
        if (d_qx_backup_) {
            return;
        }
        CUDA_CHECK(cudaMalloc(&d_qx_backup_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_qy_backup_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_qz_backup_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vx_backup_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vy_backup_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vz_backup_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_m_backup_, n_ * sizeof(double)));
    }

    void copy_state(double* qx_dst, double* qy_dst, double* qz_dst, double* vx_dst,
        double* vy_dst, double* vz_dst, double* m_dst, cudaMemcpyKind kind,
        bool from_backup = false) {
        const double* qx_src = from_backup ? d_qx_backup_ : d_qx_;
        const double* qy_src = from_backup ? d_qy_backup_ : d_qy_;
        const double* qz_src = from_backup ? d_qz_backup_ : d_qz_;
        const double* vx_src = from_backup ? d_vx_backup_ : d_vx_;
        const double* vy_src = from_backup ? d_vy_backup_ : d_vy_;
        const double* vz_src = from_backup ? d_vz_backup_ : d_vz_;
        const double* m_src = from_backup ? d_m_backup_ : d_m_;
        CUDA_CHECK(cudaMemcpy(qx_dst, qx_src, n_ * sizeof(double), kind));
        CUDA_CHECK(cudaMemcpy(qy_dst, qy_src, n_ * sizeof(double), kind));
        CUDA_CHECK(cudaMemcpy(qz_dst, qz_src, n_ * sizeof(double), kind));
        CUDA_CHECK(cudaMemcpy(vx_dst, vx_src, n_ * sizeof(double), kind));
        CUDA_CHECK(cudaMemcpy(vy_dst, vy_src, n_ * sizeof(double), kind));
        CUDA_CHECK(cudaMemcpy(vz_dst, vz_src, n_ * sizeof(double), kind));
        CUDA_CHECK(cudaMemcpy(m_dst, m_src, n_ * sizeof(double), kind));
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
    CUDA_CHECK(cudaMalloc(&d_problem3_stats, sizeof(PlanetAsteroidStats)));
    CUDA_CHECK(cudaMalloc(&d_device_stats, sizeof(DeviceScenarioStats)));
    PlanetAsteroidStats* h_problem3_stats_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_problem3_stats_pinned, sizeof(PlanetAsteroidStats)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

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
        CUDA_CHECK(cudaMemcpy(d_problem3_stats, &host_problem3_stats,
            sizeof(PlanetAsteroidStats), cudaMemcpyHostToDevice));
        DeviceScenarioStats host_device_stats{};
        host_device_stats.planet_idx = planet;
        host_device_stats.device_idx = device_idx;
        host_device_stats.mass_zeroed = 0;
        host_device_stats.destroyed_step = -1;
        host_device_stats.missile_speed = param::missile_speed;
        host_device_stats.dt = param::dt;
        CUDA_CHECK(cudaMemcpy(d_device_stats, &host_device_stats, sizeof(DeviceScenarioStats),
            cudaMemcpyHostToDevice));

        simulator.record_distance_stats(d_problem3_stats, 0, stream);
        for (int step = 1; step <= param::n_steps; ++step) {
            simulator.run_step(step, d_problem3_stats, d_device_stats, stream);
            if (step % 100 == 0) {
                CUDA_CHECK(cudaMemcpyAsync(h_problem3_stats_pinned, d_problem3_stats,
                    sizeof(PlanetAsteroidStats), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (h_problem3_stats_pinned->hit_step != -1) {
                    break;
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(h_problem3_stats_pinned, d_problem3_stats,
            sizeof(PlanetAsteroidStats), cudaMemcpyDeviceToHost));
        DeviceScenarioStats final_device_stats{};
        CUDA_CHECK(cudaMemcpy(&final_device_stats, d_device_stats,
            sizeof(DeviceScenarioStats), cudaMemcpyDeviceToHost));

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

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_problem3_stats);
    cudaFree(d_device_stats);
    cudaFreeHost(h_problem3_stats_pinned);
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
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(PlanetAsteroidStats)));
    CUDA_CHECK(cudaMemcpy(d_stats, &host_stats, sizeof(PlanetAsteroidStats),
        cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    simulator.record_distance_stats(d_stats, 0);
    for (int step = 1; step <= param::n_steps; ++step) {
        simulator.run_step(step, d_stats);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(&host_stats, d_stats, sizeof(PlanetAsteroidStats),
        cudaMemcpyDeviceToHost));
    cudaFree(d_stats);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(PlanetAsteroidStats)));
    CUDA_CHECK(cudaMemcpy(d_stats, &host_stats, sizeof(PlanetAsteroidStats),
        cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    simulator.record_distance_stats(d_stats, 0);
    for (int step = 1; step <= param::n_steps; ++step) {
        simulator.run_step(step, d_stats);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(&host_stats, d_stats, sizeof(PlanetAsteroidStats),
        cudaMemcpyDeviceToHost));
    cudaFree(d_stats);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
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
            CUDA_CHECK(cudaSetDevice(0));
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
            CUDA_CHECK(cudaSetDevice(1));
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
