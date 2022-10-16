Copyright 2021 NVIDIA Corporation. All rights reserved

Profiling API injection sample code

Build this sample with

make CUDA_INSTALL_PATH=/path/to/cuda

This x86 linux-only sample contains 4 build targets:

libinjection_1.so
    * Minimal injection sample code showing how to write an injection library using
      either LD_PRELOAD to perform dlsym() interception, or CUDA's injection support with
      CUDA_INJECTION64_PATH.
    * When CUDA_INJECTION64_PATH is set to a shared library, at initialization, CUDA
      will load the shared object and call the function named 'InitializeInjection'.
    * When LD_PRELOAD is set to a shared library, its symbols will be preferrentially
      used to resolve dynamic linking.  When an application dynamically links in the
      dlsym() call, this version of dlsym() is provided instead of the default system
      version.  In this case, dlsym() is used to call CUPTI initialization code, then
      call an internal name for the system dlsym(), ensuring that the original functionality
      of dlsym() is preserved.
    *** While this sample shows potential use of LD_PRELOAD, CUPTI does not currently
      recommend using this means of injecting a tool into a process - CUPTI's initialization
      may run before other objects are constructed, causing potential undefined behavior.
      For this reason we only recommend using CUDA_INJECTION64_PATH to guarantee
      correct behavior. ***

libinjection_2.so
    * Expands on the injection_1 sample to add CUPTI Callback and Profiler API calls
    * Registers callbacks for cuLaunchKernel and context creation.  This will be
      sufficient for many target applications, but others may require other launches
      to be matched, eg cuLaunchCoooperativeKernel or cuLaunchGrid.  See the Callback
      API for all possible kernel launch callbacks.
    * Creates a Profiler API configuration for each context in the target (using the
      context creation callback).  The Profiler API is configured using Kernel Replay
      and Auto Range modes with a configurable number of kernel launches within a pass.
    * The kernel launch callback is used to track how many kernels have launched in
      a given context's current pass, and if the pass reached its maximum count, it
      prints the metrics and starts a new pass.
    * At exit, any context with an unprocessed metrics (any which had partially
      completed a pass) print their data.
    * This library links in the profilerHostUtils library which may be built from the
      cuda/extras/CUPTI/samples/extensions/src/profilerhost_util/ directory

simple_target
    * Very simple executable which calls a kernel several times with increasing amount
      of work per call.

complex_target
    * More complicated example (similar to the concurrent_profiling sample) which
      launches several patterns of kernels - using default stream, multiple streams,
      and multiple devices if there are more than one device.

To use the injection library, set CUDA_INJECTION64_PATH to point to that library
when you launch the target application:

env CUDA_INJECTION64_PATH=./libinjection_2.so ./simple_target
