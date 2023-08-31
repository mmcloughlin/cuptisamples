#
# Copyright 2023 NVIDIA Corporation. All rights reserved
#
ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

CUDA_INSTALL_PATH ?= ../../../..
CUPTI_INSTALL_PATH ?= $(CUDA_INSTALL_PATH)/extras/CUPTI
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I"$(CUPTI_INSTALL_PATH)/include" -I"$(CUPTI_INSTALL_PATH)/samples/common"

ifeq ($(OS),Windows_NT)
    LIB_PATH ?= "$(CUPTI_INSTALL_PATH)\lib64"
else
    EXTRAS_LIB_PATH := $(CUPTI_INSTALL_PATH)/lib64
    LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):$(LIB_PATH)
    LIBS= -lcuda -L $(LIB_PATH) -lcupti
    OBJ = obj
else
    export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
    LIBS = -L $(EXTRAS_LIB_PATH) -lcuda -L $(LIB_PATH) -lcupti
    OBJ = o
endif

# Gencode arguments
SMS ?= 70 72 75 80 86 87 89 90
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

sass_metrics: sass_metrics.$(OBJ)
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) -o $@ sass_metrics.$(OBJ) $(LIBS)

sass_metrics.$(OBJ): sass_metrics.cu
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) $(GENCODE_FLAGS) -lineinfo  -c $(INCLUDES) $<

run: sass_metrics
	./$<

clean:
	rm -f sass_metrics sass_metrics.$(OBJ)

