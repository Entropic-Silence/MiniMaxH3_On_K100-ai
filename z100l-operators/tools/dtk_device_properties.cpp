// Host-only metadata adapter. No device code or computation.
#include <hip/hip_runtime_api.h>
#include <cstring>
struct DtkProperties {
    int max_shared_mem, max_num_regs, multiprocessor_count, sm_clock_rate;
    int mem_clock_rate, mem_bus_width, warpSize, max_threads_per_sm, cooperativeLaunch;
    char arch[256];
};
extern "C" int h3_dtk_properties(int device, DtkProperties *out) {
    hipDeviceProp_t p{};
    auto status = hipGetDeviceProperties(&p, device);
    if (status != hipSuccess) return static_cast<int>(status);
    out->max_shared_mem = p.sharedMemPerBlock;
    out->max_num_regs = p.regsPerBlock;
    out->multiprocessor_count = p.multiProcessorCount;
    out->sm_clock_rate = p.clockRate;
    out->mem_clock_rate = p.memoryClockRate;
    out->mem_bus_width = p.memoryBusWidth;
    out->warpSize = p.warpSize;
    out->max_threads_per_sm = p.maxThreadsPerMultiProcessor;
    out->cooperativeLaunch = p.cooperativeLaunch;
    std::strncpy(out->arch, p.gcnArchName, sizeof(out->arch)-1);
    return 0;
}
