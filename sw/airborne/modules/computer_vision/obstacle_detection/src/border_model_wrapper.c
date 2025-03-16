// border_model_wrapper.c
#include "border_model_wrapper.h"

// Include the generated model code with entry renamed to entry_border
#define entry entry_border
#include "models/border_detector.c"
#undef entry