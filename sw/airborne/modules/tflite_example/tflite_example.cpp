/**
 * @file "modules/tflite_example/tflite_example.cpp"
 * @author Daniel Rugge
 * A simple module that demonstrates the usage of TensorFlow Lite in Paparazzi.
 */

 #include "tflite_example.h"
 #include <cstdio>
 
 // Include TFLite headers (assumes the include path is set via the build system)
 #include "tensorflow/lite/interpreter.h"
 #include "tensorflow/lite/model.h"
 #include "tensorflow/lite/kernels/register.h"
 
 // Example function to test TFLite.
 // This function follows a similar style as opencv_example, but here we load a TFLite model,
 // allocate tensors, run a dummy inference, and print the output.
 int tflite_example(void) {
   printf("TFLite Example: Starting module\n");
 
   // Use a define (from the XML or default) for the model path.
 #ifdef TFLITE_MODEL_PATH
   const char* model_path = TFLITE_MODEL_PATH;
 #else
   // Fallback model path if not defined via xml
   const char* model_path = "/path/to/model.tflite";
 #endif
 
   // Load the TFLite model
   auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
   if (!model) {
     printf("TFLite Example: Failed to load TFLite model from %s\n", model_path);
     return -1;
   }
 
   // Create the built-in op resolver.
   tflite::ops::builtin::BuiltinOpResolver resolver;
 
   // Build the TFLite interpreter.
   tflite::Interpreter* interpreter = nullptr;
   tflite::InterpreterBuilder builder(*model, resolver);
   if (builder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
     printf("TFLite Example: Failed to construct interpreter\n");
     return -1;
   }
 
   // Allocate memory for tensors.
   if (interpreter->AllocateTensors() != kTfLiteOk) {
     printf("TFLite Example: Failed to allocate tensors\n");
     return -1;
   }
 
   // Assume the model expects a float input; write a test value.
   if (interpreter->inputs().size() > 0) {
     float* input = interpreter->typed_input_tensor<float>(0);
     input[0] = 1.0f;  // Dummy input value.
   } else {
     printf("TFLite Example: No input tensor available\n");
     return -1;
   }
 
   // Run inference.
   if (interpreter->Invoke() != kTfLiteOk) {
     printf("TFLite Example: Inference failed\n");
     return -1;
   }
 
   // Retrieve and print the first output value.
   if (interpreter->outputs().size() > 0) {
     float* output = interpreter->typed_output_tensor<float>(0);
     printf("TFLite Example: Inference output: %f\n", output[0]);
   } else {
     printf("TFLite Example: No output tensor available\n");
   }
 
   printf("TFLite Example: Module complete\n");
   return 0;
 }
 
 // A simple initialization function
 void tflite_example_init(void) {
   tflite_example();
 }