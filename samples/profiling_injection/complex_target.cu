// Copyright 2021 NVIDIA Corporation. All rights reserved
//
// This is a sample CUDA application with several different kernel launch
// patterns - launching on the default stream, multple streams, and multiple
// threads on different devices, if more than one device is present.
//
// The injection sample shared library can be used on this sample application,
// demonstrating that the injection code handles multple streams and multiple
// threads.

// Standard CUDA
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"

// Standard STL headers
#include <chrono>
#include <cstdint>
#include <iostream>
using ::std::cerr;
using ::std::cout;
using ::std::endl;

#include <string>
using ::std::string;

#include <thread>
using ::std::thread;

#include <vector>
using ::std::vector;

#include <stdlib.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Helpful error handlers for standard CUDA runtime calls
#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define MEMORY_ALLOCATION_CALL(var)                                             \
do {                                                                            \
    if (var == NULL) {                                                          \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",            \
                __FILE__, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// Per-device configuration, buffers, stream and device information, and device pointers
typedef struct {
    int deviceID;
    CUcontext context;         //!< CUDA driver context, or NULL if default context has already been initialized
    vector<cudaStream_t> streams;           // Each device needs its own streams
    vector<double *> d_x;                   // And device memory allocation
    vector<double *> d_y;                   // ..
} perDeviceData;

#define DAXPY_REPEAT 32768
// Loop over array of elements performing daxpy multiple times
// To be launched with only one block (artificially increasing serial time to better demonstrate overlapping replay)
__global__ void daxpyKernel(int elements, double a, double * x, double * y)
{
    for (int i = threadIdx.x; i < elements; i += blockDim.x)
        // Artificially increase kernel runtime to emphasize concurrency
        for (int j = 0; j < DAXPY_REPEAT; j++)
            y[i] = a * x[i] + y[i]; // daxpy
}

// Initialize kernel values
double a = 2.5;

// Normally you would want multiple warps, but to emphasize concurrency with streams and multiple devices
// we run the kernels on a single warp.
int threadsPerBlock = 32;
int threadBlocks = 1;

// Configurable number of kernels (streams, when running concurrently)
int const numKernels = 4;
int const numStreams = numKernels;
vector<size_t> elements(numKernels);

// Each kernel call allocates and computes (call number) * (blockSize) elements
// For 4 calls, this is 4k elements * 2 arrays * (1 + 2 + 3 + 4 stream mul) * 8B/elem =~ 640KB
int const blockSize = 4 * 1024;

// Wrapper which will launch numKernel kernel calls on a single device
// The device streams vector is used to control which stream each call is made on
// If 'serial' is non-zero, the device streams are ignored and instead the default stream is used
void launchKernels(perDeviceData &d, char const * const rangeName, bool serial)
{
    // Switch to desired device
    RUNTIME_API_CALL(cudaSetDevice(d.deviceID));
    DRIVER_API_CALL(cuCtxSetCurrent(d.context));

    for (unsigned int stream = 0; stream < d.streams.size(); stream++)
    {
        cudaStream_t streamId = (serial ? 0 : d.streams[stream]);
        daxpyKernel <<<threadBlocks, threadsPerBlock, 0, streamId>>> (elements[stream], a, d.d_x[stream], d.d_y[stream]);
        RUNTIME_API_CALL(cudaGetLastError());
    }

    // After launching all work, synchronize all streams
    if (serial == false)
    {
        for (unsigned int stream = 0; stream < d.streams.size(); stream++)
        {
            RUNTIME_API_CALL(cudaStreamSynchronize(d.streams[stream]));
        }
    }
    else
    {
        RUNTIME_API_CALL(cudaStreamSynchronize(0));
    }
}


int main(int argc, char * argv[])
{
    int numDevices;
    RUNTIME_API_CALL(cudaGetDeviceCount(&numDevices));

    // Per-device information
    vector<int> device_ids;

    // Find all devices
    for (int i = 0; i < numDevices; i++)
    {
        // Record device number
        device_ids.push_back(i);
    }

    numDevices = device_ids.size();
    cout << "Found " << numDevices << " devices" << endl;

    // Ensure we found at least one device
    if (numDevices == 0)
    {
        cerr << "No devices detected" << endl;
        exit(EXIT_WAIVED);
    }

    // Initialize kernel input to some known numbers
    vector<double> h_x(blockSize * numKernels);
    vector<double> h_y(blockSize * numKernels);
    for (size_t i = 0; i < blockSize * numKernels; i++)
    {
        h_x[i] = 1.5 * i;
        h_y[i] = 2.0 * (i - 3000);
    }

    // Initialize a vector of 'default stream' values to demonstrate serialized kernels
    vector<cudaStream_t> defaultStreams(numStreams);
    for (int stream = 0; stream < numStreams; stream++)
    {
        defaultStreams[stream] = 0;
    }

    // Scale per-kernel work by stream number
    for (int stream = 0; stream < numStreams; stream++)
    {
        elements[stream] = blockSize * (stream + 1);
    }

    // For each device, configure profiling, set up buffers, copy kernel data
    vector<perDeviceData> deviceData(numDevices);

    for (int device = 0; device < numDevices; device++)
    {
        RUNTIME_API_CALL(cudaSetDevice(device_ids[device]));
        cout << "Configuring device " << device_ids[device] << endl;

        // For simplicity's sake, in this sample, a single config struct is created per device
        deviceData[device].deviceID = device_ids[device];// GPU device ID

        DRIVER_API_CALL(cuCtxCreate(&(deviceData[device].context), 0, device_ids[device])); // Either set to a context, or may be NULL if a default context has been created

        // Per-stream initialization & memory allocation - copy from constant host array to each device array
        deviceData[device].streams.resize(numStreams);
        deviceData[device].d_x.resize(numStreams);
        deviceData[device].d_y.resize(numStreams);
        for (int stream = 0; stream < numStreams; stream++)
        {
            RUNTIME_API_CALL(cudaStreamCreate(&(deviceData[device].streams[stream])));

            // Each kernel does (stream #) * blockSize work on doubles
            size_t size = elements[stream] * sizeof(double);

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].d_x[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].d_x[stream]); // Validate pointer
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].d_x[stream], h_x.data(), size, cudaMemcpyHostToDevice));

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].d_y[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].d_y[stream]); // Validate pointer
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].d_y[stream], h_x.data(), size, cudaMemcpyHostToDevice));
        }
    }

    //
    // First version - single device, kernel calls serialized on default stream
    //

    // Use wallclock time to measure performance
    auto begin_time = ::std::chrono::high_resolution_clock::now();

    // Run on first device and use default streams - will show runtime without any concurrency
    launchKernels(deviceData[0], "single_gpu_serial", true);

    auto end_time = ::std::chrono::high_resolution_clock::now();
    auto elapsed_serial_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
    cout << "It took " << elapsed_serial_ms.count() << "ms on the host to launch " << numKernels << " kernels in serial" << endl;

    //
    // Second version - same kernel calls as before on the same device, but now using separate streams for concurrency
    // (Should be limited by the longest running kernel)
    //

    begin_time = ::std::chrono::high_resolution_clock::now();

    // Still only use first device, but this time use its allocated streams for parallelism
    launchKernels(deviceData[0], "single_gpu_async", false);

    end_time = ::std::chrono::high_resolution_clock::now();
    auto elapsed_single_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
    cout << "It took " << elapsed_single_device_ms.count() << "ms on the host to launch " << numKernels << " kernels on a single device on separate streams" << endl;

    //
    // Third version - same as the second case, but duplicate the work across devices to show cross-device concurrency
    // This is done using threads so no serialization is needed between devices
    // (Should have roughly the same runtime as second case)
    //

    // Time creation of the same multiple streams * multiple devices
    vector<::std::thread> threads;
    begin_time = ::std::chrono::high_resolution_clock::now();

    // Now launch parallel thread work, duplicated on one thread per gpu
    for (int device = 0; device < numDevices; device++)
    {
        threads.push_back(::std::thread(launchKernels, ::std::ref(deviceData[device]), "multi_gpu_async", false));
    }

    // Wait for all threads to finish
    for (auto &t: threads)
    {
        t.join();
    }

    // Record time used when launching on multiple devices
    end_time = ::std::chrono::high_resolution_clock::now();
    auto elapsed_multiple_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
    cout << "It took " << elapsed_multiple_device_ms.count() << "ms on the host to launch the same " << numKernels << " kernels on each of the " << numDevices << " devices in parallel" << endl;

    // Free stream memory for each device
    for (int i = 0; i < numDevices; i++)
    {
        for (int j = 0; j < numKernels; j++)
        {
            RUNTIME_API_CALL(cudaFree(deviceData[i].d_x[j]));
            RUNTIME_API_CALL(cudaFree(deviceData[i].d_y[j]));
        }
    }

    exit(EXIT_SUCCESS);
}
