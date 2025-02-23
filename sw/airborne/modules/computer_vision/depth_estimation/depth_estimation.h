#ifndef DEPTH_ESTIMATION_H
#define DEPTH_ESTIMATION_H

#include <stdbool.h>

extern bool depth_estimation_init(void);
extern void depth_estimation_periodic(void);
extern void depth_estimation_cleanup(void);

#endif