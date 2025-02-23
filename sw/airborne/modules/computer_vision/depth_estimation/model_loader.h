#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <tensorflow/lite/c/c_api.h>

// Model context structure
typedef struct {
    TfLiteModel* model;
    TfLiteInterpreterOptions* options;
    TfLiteInterpreter* interpreter;
} ModelContext;

// Function declarations
ModelContext* load_model(const char* model_path);
void print_model_info(ModelContext* ctx);
void free_model_context(ModelContext* ctx);

#endif // MODEL_LOADER_H