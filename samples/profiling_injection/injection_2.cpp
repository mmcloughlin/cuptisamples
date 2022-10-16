// Copyright 2021 NVIDIA Corporation. All rights reserved
//
// This sample demostrates using the profiler API in injection mode.
// Build this file as a shared object, and set environment variable
// CUDA_INJECTION64_PATH to the full path to the .so.
//
// CUDA will load the object during initialization and will run
// the function called 'InitializeInjection'.
//
// After the initialization routine  returns, the application resumes running,
// with the registered callbacks triggering as expected.  These callbacks
// are used to start a Profiler API session using Kernel Replay and
// Auto Range modes.
//
// A configurable number of kernel launches (default 10) are run
// under one session.  Before the 11th kernel launch, the callback
// ends the session, prints metrics, and starts a new session.
//
// An atexit callback is also used to ensure that any partial sessions
// are handled when the target application exits.
//
// This code supports multiple contexts and multithreading through
// locking shared data structures.

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cupti_callbacks.h"
#include "cupti_profiler_target.h"
#include "cupti_driver_cbid.h"
#include "cupti_target.h"
#include "cupti_activity.h"
#include "nvperf_host.h"

#include <Eval.h>
using ::NV::Metric::Eval::PrintMetricValues;

#include <Metric.h>
using ::NV::Metric::Config::GetConfigImage;
using ::NV::Metric::Config::GetCounterDataPrefixImage;

#include <Utils.h>
using ::NV::Metric::Utils::GetNVPWResultString;

#include <iostream>
using ::std::cerr;
using ::std::cout;
using ::std::endl;

#include <mutex>
using ::std::mutex;

#include <string>
using ::std::string;

#include <unordered_map>
using ::std::unordered_map;

#include <unordered_set>
using ::std::unordered_set;

#include <vector>
using ::std::vector;

#include <stdlib.h>

#include "dlfcn.h" // dlsym, RTLD_NEXT
extern "C"
{
    extern decltype(dlsym) __libc_dlsym;
    extern decltype(dlopen) __libc_dlopen_mode;
}

// Export InitializeInjection symbol
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define HIDDEN
#else
#define DLLEXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Helpful error handlers for standard CUPTI and CUDA runtime calls
#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, errstr);                     \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define MEMORY_ALLOCATION_CALL(var)                                            \
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

#define NVPW_API_CALL(apiFuncCall)                                             \
do {                                                                           \
    NVPA_Status _status = apiFuncCall;                                         \
    if (_status != NVPA_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
            __FILE__, __LINE__, #apiFuncCall, GetNVPWResultString(_status));   \
    exit(EXIT_FAILURE);                                                        \
    }                                                                          \
} while (0)

// Profiler API configuration data, per-context
struct ctxProfilerData
{
    CUcontext       ctx;
    int             dev_id;
    cudaDeviceProp  dev_prop;
    vector<uint8_t> counterAvailabilityImage;
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    vector<uint8_t> counterDataImage;
    vector<uint8_t> counterDataPrefixImage;
    vector<uint8_t> counterDataScratchBufferImage;
    vector<uint8_t> configImage;
    int             maxNumRanges;
    int             curRanges;
    int             maxRangeNameLength;
    int             iterations; // Count of sessions

    // Initialize fields, with env var overrides
    ctxProfilerData() : curRanges(), maxRangeNameLength(64), iterations()
    {
        char * env_var = getenv("INJECTION_KERNEL_COUNT");
        if (env_var != NULL)
        {
            int val = atoi(env_var);
            if (val < 1)
            {
                cerr << "Read " << val << " kernels from INJECTION_KERNEL_COUNT, but must be >= 1; defaulting to 10." << endl;
                val = 10;
            }
            maxNumRanges = val;
        }
        else
        {
            maxNumRanges = 10;
        }
    };
};

// Track per-context profiler API data in a shared map
mutex ctx_data_mutex;
unordered_map<CUcontext, ctxProfilerData> ctx_data;

// List of metrics to collect
vector<string> metricNames;

// Initialize state
void initialize_state()
{
    static int profiler_initialized = 0;

    if (profiler_initialized == 0)
    {
        // CUPTI Profiler API initialization
        CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

        // NVPW required initialization
        NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
        NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

        profiler_initialized = 1;
    }
}

// Initialize profiler for a context
void initialize_ctx_data(ctxProfilerData &ctx_data)
{
    initialize_state();

    // Get size of counterAvailabilityImage - in first pass, GetCounterAvailability return size needed for data
    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = { CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE };
    getCounterAvailabilityParams.ctx = ctx_data.ctx;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Allocate sized counterAvailabilityImage
    ctx_data.counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);

    // Initialize counterAvailabilityImage
    getCounterAvailabilityParams.pCounterAvailabilityImage = ctx_data.counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Fill in configImage - can be run on host or target
    if (!GetConfigImage(ctx_data.dev_prop.name, metricNames, ctx_data.configImage, ctx_data.counterAvailabilityImage.data()))
    {
        cerr << "Failed to create configImage for context " << ctx_data.ctx << endl;
        exit(EXIT_FAILURE);
    }

    // Fill in counterDataPrefixImage - can be run on host or target
    if (!GetCounterDataPrefixImage(ctx_data.dev_prop.name, metricNames, ctx_data.counterDataPrefixImage, ctx_data.counterAvailabilityImage.data()))
    {
        cerr << "Failed to create counterDataPrefixImage for context " << ctx_data.ctx << endl;
        exit(EXIT_FAILURE);
    }

    // Record counterDataPrefixImage info and other options for sizing the counterDataImage
    ctx_data.counterDataImageOptions.pCounterDataPrefix = ctx_data.counterDataPrefixImage.data();
    ctx_data.counterDataImageOptions.counterDataPrefixSize = ctx_data.counterDataPrefixImage.size();
    ctx_data.counterDataImageOptions.maxNumRanges = ctx_data.maxNumRanges;
    ctx_data.counterDataImageOptions.maxNumRangeTreeNodes = ctx_data.maxNumRanges;
    ctx_data.counterDataImageOptions.maxRangeNameLength = ctx_data.maxRangeNameLength;

    // Calculate size of counterDataImage based on counterDataPrefixImage and options
    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = { CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
    calculateSizeParams.pOptions = &(ctx_data.counterDataImageOptions);
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));
    // Create counterDataImage
    ctx_data.counterDataImage.resize(calculateSizeParams.counterDataImageSize);

    // Initialize counterDataImage inside start_session
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.pOptions = &(ctx_data.counterDataImageOptions);
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.counterDataImageSize = ctx_data.counterDataImage.size();
    initializeParams.pCounterDataImage = ctx_data.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    // Calculate scratchBuffer size based on counterDataImage size and counterDataImage
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = { CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
    scratchBufferSizeParams.counterDataImageSize = ctx_data.counterDataImage.size();
    scratchBufferSizeParams.pCounterDataImage = ctx_data.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));
    // Create counterDataScratchBuffer
    ctx_data.counterDataScratchBufferImage.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    // Initialize counterDataScratchBuffer
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
    initScratchBufferParams.counterDataImageSize = ctx_data.counterDataImage.size();
    initScratchBufferParams.pCounterDataImage = ctx_data.counterDataImage.data();
    initScratchBufferParams.counterDataScratchBufferSize = ctx_data.counterDataScratchBufferImage.size();;
    initScratchBufferParams.pCounterDataScratchBuffer = ctx_data.counterDataScratchBufferImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

}

// Start a session
void start_session(ctxProfilerData &ctx_data)
{
    CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
    beginSessionParams.counterDataImageSize = ctx_data.counterDataImage.size();
    beginSessionParams.pCounterDataImage = ctx_data.counterDataImage.data();
    beginSessionParams.counterDataScratchBufferSize = ctx_data.counterDataScratchBufferImage.size();
    beginSessionParams.pCounterDataScratchBuffer = ctx_data.counterDataScratchBufferImage.data();
    beginSessionParams.ctx = ctx_data.ctx;
    beginSessionParams.maxLaunchesPerPass = ctx_data.maxNumRanges;
    beginSessionParams.maxRangesPerPass = ctx_data.maxNumRanges;
    beginSessionParams.pPriv = NULL;
    beginSessionParams.range = CUPTI_AutoRange;
    beginSessionParams.replayMode = CUPTI_KernelReplay;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
    setConfigParams.pConfig = ctx_data.configImage.data();
    setConfigParams.configSize = ctx_data.configImage.size();
    setConfigParams.passIndex = 0; // Only set for Application Replay mode
    setConfigParams.minNestingLevel = 1;
    setConfigParams.numNestingLevels = 1;
    setConfigParams.targetNestingLevel = 1;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
    enableProfilingParams.ctx = ctx_data.ctx;
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));

    ctx_data.iterations++;
}

// Print session data
static void print_data(ctxProfilerData &ctx_data)
{
    cout << endl << "Context " << ctx_data.ctx << ", device " << ctx_data.dev_id << " (" << ctx_data.dev_prop.name << ") session " << ctx_data.iterations << ":" << endl;
    PrintMetricValues(ctx_data.dev_prop.name, ctx_data.counterDataImage, metricNames, ctx_data.counterAvailabilityImage.data());
}

// End a session during execution
void end_session(ctxProfilerData &ctx_data)
{
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
    disableProfilingParams.ctx = ctx_data.ctx;
    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
    unsetConfigParams.ctx = ctx_data.ctx;
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

    CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
    endSessionParams.ctx = ctx_data.ctx;
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

    print_data(ctx_data);

    // Clear counterDataImage (otherwise it maintains previous records when it is reused)
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.pOptions = &(ctx_data.counterDataImageOptions);
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.counterDataImageSize = ctx_data.counterDataImage.size();
    initializeParams.pCounterDataImage = ctx_data.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));
}

// Clean up at end of execution
static void end_execution()
{
    CUPTI_API_CALL(cuptiGetLastError());
    ctx_data_mutex.lock();

    for (auto itr = ctx_data.begin(); itr != ctx_data.end(); ++itr)
    {
        ctxProfilerData &data = itr->second;

        if (data.curRanges > 0)
        {
            print_data(data);
            data.curRanges = 0;
        }
    }

    ctx_data_mutex.unlock();
}

// Callback handler
void callback(void * userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const * cbdata)
{
    static int initialized = 0;

    CUptiResult res;
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API)
    {
        // For a driver call to launch a kernel:
        if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)
        {
            CUpti_CallbackData const * data = static_cast<CUpti_CallbackData const *>(cbdata);
            CUcontext ctx = data->context;

            // On entry, enable / update profiling as needed
            if (data->callbackSite == CUPTI_API_ENTER)
            {
                // Check for this context in the configured contexts
                // If not configured, it isn't compatible with profiling
                ctx_data_mutex.lock();
                if (ctx_data.count(ctx) > 0)
                {
                    // If at maximum number of ranges, end session and reset
                    if (ctx_data[ctx].curRanges == ctx_data[ctx].maxNumRanges)
                    {
                        end_session(ctx_data[ctx]);
                        ctx_data[ctx].curRanges = 0;
                    }

                    // If no currently enabled session on this context, start one
                    if (ctx_data[ctx].curRanges == 0)
                    {
                        initialize_ctx_data(ctx_data[ctx]);
                        start_session(ctx_data[ctx]);
                    }

                    // Increment curRanges
                    ctx_data[ctx].curRanges++;
                }
                ctx_data_mutex.unlock();
            }
        }
    }
    else if (domain == CUPTI_CB_DOMAIN_RESOURCE)
    {
        // When a context is created, check to see whether the device is compatible with the Profiler API
        if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)
        {
            CUpti_ResourceData const * res_data = static_cast<CUpti_ResourceData const *>(cbdata);
            CUcontext ctx = res_data->context;

            // Configure handler for new context under lock
            ctxProfilerData data = { };

            data.ctx = ctx;

            RUNTIME_API_CALL(cudaGetDevice(&(data.dev_id)));

            // Initialize profiler API and test device compatibility
            initialize_state();
            CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
            params.cuDevice = data.dev_id;
            CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

            // If valid for profiling, set up profiler and save to shared structure
            ctx_data_mutex.lock();
            if (params.isSupported == CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
            {
                // Update shared structures
                ctx_data[ctx] = data;
                initialize_ctx_data(ctx_data[ctx]);
            }
            else
            {
                if (ctx_data.count(ctx))
                {
                    // Update shared structures
                    ctx_data.erase(ctx);
                }

                cerr << "libinjection_2: Unable to profile context on device " << data.dev_id << endl;

                if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice architecture is not supported" << endl;
                }

                if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice sli configuration is not supported" << endl;
                }

                if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice vgpu configuration is not supported" << endl;
                }
                else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
                {
                    cerr << "\tdevice vgpu configuration disabled profiling support" << endl;
                }

                if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice confidential compute configuration is not supported" << endl;
                }

                if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
                }
            }
            ctx_data_mutex.unlock();
        }
    }

    return;
}

// Register callbacks for several points in target application execution
void register_callbacks()
{
    // One subscriber is used to register multiple callback domains
    CUpti_SubscriberHandle subscriber;
    CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callback, NULL));
    // Runtime callback domain is needed for kernel launch callbacks
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    // Resource callback domain is needed for context creation callbacks
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));

    // Register callback for application exit
    atexit(end_execution);
}

static bool injectionInitialized = false;

// InitializeInjection will be called by the driver when the tool is loaded
// by CUDA_INJECTION64_PATH, - or -
// InitializeInjection should be called before the first CUDA function in the
// target application.  It cannot call any CUDA runtime or driver code, but
// the CUPTI Callback API is supported at this point.
extern "C" DLLEXPORT int InitializeInjection()
{
    if (injectionInitialized == false)
    {
        injectionInitialized = true;

        // Read in optional list of metrics to gather
        char * metrics_env = getenv("INJECTION_METRICS");
        if (metrics_env != NULL)
        {
            char * tok = strtok(metrics_env, " ;,");
            do
            {
                cout << "Read " << tok << endl;
                metricNames.push_back(string(tok));
                tok = strtok(NULL, " ;,");
            } while (tok != NULL);
        }
        else
        {
            metricNames.push_back("sm__cycles_elapsed.avg");
            metricNames.push_back("smsp__sass_thread_inst_executed_op_dadd_pred_on.avg");
            metricNames.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.avg");
        }

        // Subscribe to some callbacks
        register_callbacks();
    }
    return 1;
}

// Whether the application calls the runtime or driver CUDA API, dynamic
// linking will likely use dlsym - intercept this call with LD_PRELOAD to
// have a convenient place to initialize Cupti Callback API.
// Note that there are possible timing issues if this dlsym call occurs
// before all constructors have run.
extern "C" DLLEXPORT void * dlsym(void * handle, char const * symbol)
{
    InitializeInjection();

    typedef void * (*dlsym_fn)(void *, char const *);
    static dlsym_fn real_dlsym = NULL;
    if (real_dlsym == NULL)
    {
        // Use libc internal names to avoid recursive call
        real_dlsym = (dlsym_fn)(__libc_dlsym(__libc_dlopen_mode("libdl.so", RTLD_LAZY), "dlsym"));
    }
    if (real_dlsym == NULL)
    {
        cerr << "Error finding real dlsym symbol" << endl;
        return NULL;
    }
    return real_dlsym(handle, symbol);
}
