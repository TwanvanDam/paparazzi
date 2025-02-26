#ifndef INFERENCE_H
#define INFERENCE_H

#include "model_loader.h"

typedef struct {
    float* depth_map;
    int width;
    int height;
} DepthMapResult;

ModelContext* init_inference(const char* model_path);
DepthMapResult* run_inference(ModelContext* model_ctx, float* preprocessed_data);
void free_depth_map_result(DepthMapResult* result);
void cleanup_inference(ModelContext* model_ctx);
int save_depth_map(const char* filename, const DepthMapResult* depth_map);

#endif // INFERENCE_H