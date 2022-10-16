/*
 * Copyright 2015 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI library for OpenACC data collection.
 */

#include <stdlib.h>
#include <stdio.h>
#include <cupti.h>
#include <cuda.h>
#include <openacc.h>

// helper macros

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static size_t openacc_records = 0;

static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_OPENACC_DATA:
      case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH:
      case CUPTI_ACTIVITY_KIND_OPENACC_OTHER:
          {
              CUpti_ActivityOpenAcc *oacc =
                     (CUpti_ActivityOpenAcc *)record;

              if (oacc->deviceType != acc_device_nvidia) {
                  printf("Error: OpenACC device type is %u, not %u (acc_device_nvidia)\n",
                         oacc->deviceType, acc_device_nvidia);
                  exit(EXIT_FAILURE);
              }

              openacc_records++;
          }
          break;

      default:
          ;
  }
}

// CUPTI buffer handling

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(EXIT_FAILURE);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}

void finalize()
{
  CUPTI_CALL(cuptiGetLastError());
  CUPTI_CALL(cuptiActivityFlushAll(0));
  printf("Found %llu OpenACC records\n", (long long unsigned) openacc_records);
}

// acc_register_library is defined by the OpenACC tools interface
// and allows to register this library with the OpenACC runtime.

extern "C" void
acc_register_library(void *profRegister, void *profUnregister, void *profLookup)
{
  // once connected to the OpenACC runtime, initialize CUPTI's OpenACC interface
  if (cuptiOpenACCInitialize(profRegister, profUnregister, profLookup) != CUPTI_SUCCESS) {
    printf("Error: Failed to initialize CUPTI OpenACC support\n");
    exit(EXIT_FAILURE);
  }

  printf("Initialized CUPTI OpenACC\n");

  // only interested in OpenACC activity records
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_DATA));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_OTHER));

  // setup CUPTI buffer handling
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  // at program exit, flush CUPTI buffers and print results
  atexit(finalize);
}

