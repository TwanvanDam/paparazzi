// stream_funcs.h
#ifndef STREAM_FUNCS_H
#define STREAM_FUNCS_H

#include "modules/computer_vision/lib/vision/image.h"
#include "mcu_periph/udp.h"

struct stream_context_t {
    struct UdpSocket video_sock;
    uint16_t rtp_packet_nr;
    uint32_t rtp_frame_time;
    struct image_t img_jpeg;
    uint8_t colormap[256 * 3];
    int port;
};

void init_stream(struct stream_context_t* ctx, const char* ip, int port);
void cleanup_stream(struct stream_context_t* ctx);
void test_jpeg_conversion(struct stream_context_t* ctx);
void test_stream_setup(struct stream_context_t* ctx, const char* ip, int port);
void test_full_streaming(struct stream_context_t* ctx);
void stream_frame(struct stream_context_t* ctx, float* rgb_data, int width, int height);
void stream_depth(struct stream_context_t* ctx, float* depth_map, int width, int height, 
                 float min_depth, float max_depth);
void test_color_pattern(struct stream_context_t* ctx, int width, int height);

#endif