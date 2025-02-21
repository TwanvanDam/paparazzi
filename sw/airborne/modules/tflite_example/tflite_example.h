/**
 * @file "modules/tflite_example/tflite_example.h"
 * @author Daniel Rugge
 * Header for the TFLite example module.
 */

 #ifndef TFLITE_EXAMPLE_H
 #define TFLITE_EXAMPLE_H
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 // Function prototypes. This module provides an initialization function and a demo function.
 int tflite_example(void);
 void tflite_example_init(void);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // TFLITE_EXAMPLE_H