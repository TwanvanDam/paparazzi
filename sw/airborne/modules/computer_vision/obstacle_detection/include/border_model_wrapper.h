// border_model_wrapper.h
#ifndef BORDER_MODEL_WRAPPER_H
#define BORDER_MODEL_WRAPPER_H

// Declare the entry function
void entry_border(const float tensor_input[1][3][240][240], float tensor_output[1][1]);

enum DebugLevel {
  DEBUG_NONE = 0,
  DEBUG_MINIMAL = 1,   // Just final outputs
  DEBUG_MODERATE = 2,  // Key layer statistics
  DEBUG_VERBOSE = 3    // All layer details
};

extern int frame_counter;
extern enum DebugLevel current_debug_level;

#endif // BORDER_MODEL_WRAPPER_H