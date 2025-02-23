#include "model_loader.h"
#include <stdio.h>
#include <stdlib.h>

ModelContext* load_model(const char* model_path) {
    ModelContext* ctx = (ModelContext*)malloc(sizeof(ModelContext));
    if (!ctx) {
        printf("Failed to allocate model context\n");
        return NULL;
    }

    // Load model
    ctx->model = TfLiteModelCreateFromFile(model_path);
    if (ctx->model == NULL) {
        printf("Failed to load model: %s\n", model_path);
        free(ctx);
        return NULL;
    }

    // Create interpreter options
    ctx->options = TfLiteInterpreterOptionsCreate();
    if (ctx->options == NULL) {
        printf("Failed to create interpreter options\n");
        TfLiteModelDelete(ctx->model);
        free(ctx);
        return NULL;
    }
    TfLiteInterpreterOptionsSetNumThreads(ctx->options, 2);

    // Create interpreter
    ctx->interpreter = TfLiteInterpreterCreate(ctx->model, ctx->options);
    if (ctx->interpreter == NULL) {
        printf("Failed to create interpreter\n");
        TfLiteInterpreterOptionsDelete(ctx->options);
        TfLiteModelDelete(ctx->model);
        free(ctx);
        return NULL;
    }

    // Allocate tensors
    if (TfLiteInterpreterAllocateTensors(ctx->interpreter) != kTfLiteOk) {
        printf("Failed to allocate tensors\n");
        free_model_context(ctx);
        return NULL;
    }

    return ctx;
}

void print_model_info(ModelContext* ctx) {
    if (!ctx || !ctx->interpreter) return;

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(ctx->interpreter, 0);
    if (!input_tensor) return;

    printf("\nModel loaded successfully!\n");
    printf("Input tensor dimensions: %d x %d x %d x %d\n",
           TfLiteTensorDim(input_tensor, 0),
           TfLiteTensorDim(input_tensor, 1),
           TfLiteTensorDim(input_tensor, 2),
           TfLiteTensorDim(input_tensor, 3));
    printf("Input tensor type: %d\n", TfLiteTensorType(input_tensor));

    int input_tensors_size = TfLiteInterpreterGetInputTensorCount(ctx->interpreter);
    int output_tensors_size = TfLiteInterpreterGetOutputTensorCount(ctx->interpreter);
    printf("Number of input tensors: %d\n", input_tensors_size);
    printf("Number of output tensors: %d\n", output_tensors_size);
}

void free_model_context(ModelContext* ctx) {
    if (ctx) {
        if (ctx->interpreter) TfLiteInterpreterDelete(ctx->interpreter);
        if (ctx->options) TfLiteInterpreterOptionsDelete(ctx->options);
        if (ctx->model) TfLiteModelDelete(ctx->model);
        free(ctx);
    }
}