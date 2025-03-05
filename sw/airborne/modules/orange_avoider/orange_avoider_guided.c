/*
 * Copyright (C) Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider_guided.c"
 * @author Kirk Scheper
 * This module is an example module for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 *
 * The avoidance strategy is now improved by estimating the best turn direction instead of choosing randomly.
 */

#include "modules/orange_avoider/orange_avoider_guided.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <stdio.h>
#include <time.h>

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider_guided->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// Define settings
float oag_color_count_frac = 0.15f;
float oag_floor_count_frac = 0.05f;
float oag_max_speed = 1.0f;
float oag_heading_rate = RadOfDeg(70.f);

// Define and initialize global variables
enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS,
  REENTER_ARENA
} navigation_state = SEARCH_FOR_SAFE_HEADING;

int32_t color_count = 0;
int32_t color_count_left = 0;
int32_t color_count_right = 0;
int32_t floor_count = 0;
int32_t floor_centroid = 0;
float avoidance_heading_direction = 0;
int16_t obstacle_free_confidence = 0;
const int16_t max_trajectory_confidence = 5;

static abi_event color_detection_ev;
static void color_detection_cb(uint8_t sender_id,
                               int16_t pixel_x, int16_t pixel_y,
                               int16_t pixel_width, int16_t pixel_height,
                               int32_t quality, int16_t extra) {
  color_count = quality;
  if (navigation_state == SAFE || navigation_state == OBSTACLE_FOUND){
	  if (pixel_x > 0){
	  	avoidance_heading_direction = 1.f;
		}else {
		avoidance_heading_direction = -1.f;
		}
	}	
}

static abi_event floor_detection_ev;
static void floor_detection_cb(uint8_t sender_id,
                               int16_t pixel_x, int16_t pixel_y,
                               int16_t pixel_width, int16_t pixel_height,
                               int32_t quality, int16_t extra) {
  floor_count = quality;
  floor_centroid = pixel_y;
}

uint8_t chooseBestAvoidanceDirection(void) {
  if (color_count_left < color_count_right) {
    avoidance_heading_direction = -1.f;
  } else {
    avoidance_heading_direction = 1.f;
  }
  VERBOSE_PRINT("Set avoidance direction to: %f\n", avoidance_heading_direction * oag_heading_rate);
  color_count_left = 0;
  color_count_right = 0;
  return false;
}

void orange_avoider_guided_init(void) {
  srand(time(NULL));
  chooseBestAvoidanceDirection();
  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);
  AbiBindMsgVISUAL_DETECTION(FLOOR_VISUAL_DETECTION_ID, &floor_detection_ev, floor_detection_cb);
}

void orange_avoider_guided_periodic(void) {
  if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
    navigation_state = SEARCH_FOR_SAFE_HEADING;
    obstacle_free_confidence = 0;
    return;
  }

  int32_t color_count_threshold = oag_color_count_frac * front_camera.output_size.w * front_camera.output_size.h;
  int32_t floor_count_threshold = oag_floor_count_frac * front_camera.output_size.w * front_camera.output_size.h;
  float floor_centroid_frac = floor_centroid / (float)front_camera.output_size.h / 2.f;

  VERBOSE_PRINT("direction: %f confidence: %d state: %d \n", avoidance_heading_direction, obstacle_free_confidence, navigation_state);

  if (color_count < color_count_threshold) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  float speed_sp = fminf(oag_max_speed, 0.2f * obstacle_free_confidence);

  switch (navigation_state) {
    case SAFE:
      if (floor_count < floor_count_threshold || fabsf(floor_centroid_frac) > 0.12) {
        navigation_state = OUT_OF_BOUNDS;
      } else if (obstacle_free_confidence <= 2) {
        navigation_state = OBSTACLE_FOUND;
      } else {
        guidance_h_set_body_vel(speed_sp, 0);
      }
      break;

    case OBSTACLE_FOUND:
      guidance_h_set_body_vel(0.8f*oag_max_speed, 0);
      guidance_h_set_heading_rate(avoidance_heading_direction * oag_heading_rate);
      if (obstacle_free_confidence >= 2) {
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
        navigation_state = SAFE;
      }
      break;

    case SEARCH_FOR_SAFE_HEADING:
      guidance_h_set_heading_rate(avoidance_heading_direction * oag_heading_rate * (1.0f - obstacle_free_confidence / 5.0f));
      if (obstacle_free_confidence >= 2) {
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
        navigation_state = SAFE;
      }
      break;

    case OUT_OF_BOUNDS:
      guidance_h_set_body_vel(0, 0);
      guidance_h_set_heading_rate(avoidance_heading_direction * RadOfDeg(90));
      navigation_state = REENTER_ARENA;
      break;

    case REENTER_ARENA:
      if (floor_count >= floor_count_threshold && avoidance_heading_direction * floor_centroid_frac >= 0.f) {
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
        obstacle_free_confidence = 0;
        navigation_state = SAFE;
      }
      break;
  }
  return;
}

