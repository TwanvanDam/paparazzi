#include <stdio.h>
#include <stdlib.h>
#include "stream_funcs.h"
#include "mcu_periph/udp.h"
#include "modules/computer_vision/lib/encoding/rtp.h"
#include <jpeglib.h>
#include "udp_socket.h"

#ifndef VIEWVIDEO_QUALITY_FACTOR
#define VIEWVIDEO_QUALITY_FACTOR 50
#endif


static void init_colormap(struct stream_context_t* ctx) {
    printf("[Stream] Initializing colormap\n");
    for (int i = 0; i < 256; i++) {
        ctx->colormap[i*3] = i;     // R
        ctx->colormap[i*3+1] = i;   // G
        ctx->colormap[i*3+2] = i;   // B
    }
}

static bool rgb_to_jpeg(float* rgb_data, int width, int height, struct image_t* out) {
    // printf("[Stream] Starting JPEG conversion %dx%d\n", width, height);
    
    if (!rgb_data || !out) {
        printf("[Stream] Invalid input parameters\n");
        return false;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    
    // Convert float [0,1] to uint8 [0,255]
    uint8_t* rgb_bytes = malloc(width * height * 3);
    if (!rgb_bytes) {
        printf("[Stream] Failed to allocate RGB buffer\n");
        return false;
    }

    // printf("[Stream] Converting float to uint8\n");
    for(int i = 0; i < width * height * 3; i++) {
        float val = rgb_data[i];
        val = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
        rgb_bytes[i] = (uint8_t)(val * 255.0f);
    }

    // printf("[Stream] Setting up JPEG compression\n");
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    unsigned char* outbuffer = NULL;
    unsigned long outsize = 0;
    jpeg_mem_dest(&cinfo, &outbuffer, &outsize);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    // printf("[Stream] Setting JPEG parameters\n");
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, VIEWVIDEO_QUALITY_FACTOR, TRUE);
    
    // printf("[Stream] Starting compression\n");
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &rgb_bytes[cinfo.next_scanline * width * 3];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // printf("[Stream] Finishing compression\n");
    jpeg_finish_compress(&cinfo);
    
    out->buf = outbuffer;
    out->buf_size = outsize;
    out->type = IMAGE_JPEG;
    out->w = width;
    out->h = height;

    free(rgb_bytes);
    jpeg_destroy_compress(&cinfo);
    
    // printf("[Stream] JPEG conversion complete. Output size: %lu bytes\n", outsize);
    return true;
}

static void create_sdp_file(int width, int height, int port) {
    FILE* fp;
    fp = fopen("./sw/airborne/modules/computer_vision/depth_estimation/streams/depth_stream.sdp", "w");
    if (fp == NULL) {
        printf("[Stream] Failed to create SDP file\n");
        return;
    }

    fprintf(fp, "v=0\n");
    fprintf(fp, "m=video %d RTP/AVP 26\n", port);
    fprintf(fp, "c=IN IP4 127.0.0.1\n");
    fprintf(fp, "a=rtpmap:26 JPEG/90000\n");
    fprintf(fp, "a=framerate:10.0\n");
    fprintf(fp, "a=framesize:26 %d-%d\n", width, height);
    fprintf(fp, "a=title:PPRZ Depth Stream\n");
    
    fclose(fp);
    printf("[Stream] Created SDP file with resolution %dx%d\n", width, height);
}

void init_stream(struct stream_context_t* ctx, const char* ip, int port) {
    printf("[Stream] Initializing stream context\n");
    ctx->rtp_packet_nr = 0;
    ctx->rtp_frame_time = 0;
    ctx->img_jpeg.buf = NULL;
    ctx->img_jpeg.buf_size = 0;
    ctx->port = port;  
    
    init_colormap(ctx);
    
    char ip_copy[128];
    strncpy(ip_copy, ip, sizeof(ip_copy) - 1);
    ip_copy[sizeof(ip_copy) - 1] = '\0';
    
    int ret = udp_socket_create(&ctx->video_sock, ip_copy, port, -1, FALSE);
    if (ret < 0) {
        printf("[Stream] Failed to create UDP socket (error: %d)\n", ret);
        return;
    }
    printf("[Stream] UDP socket created successfully\n");

    // Create SDP file with initial resolution
    create_sdp_file(640, 480, port);
}


void cleanup_stream(struct stream_context_t* ctx) {
    printf("[Stream] Cleaning up stream context\n");
    if (ctx->img_jpeg.buf) {
        free(ctx->img_jpeg.buf);
        ctx->img_jpeg.buf = NULL;
    }
    close(ctx->video_sock.sockfd);
}

void test_jpeg_conversion(struct stream_context_t* ctx) {
    printf("[Stream] Testing JPEG conversion\n");
    
    // Create a small test image (2x2 pixels)
    float test_rgb[12] = {
        1.0f, 0.0f, 0.0f,  // Red
        0.0f, 1.0f, 0.0f,  // Green
        0.0f, 0.0f, 1.0f,  // Blue
        1.0f, 1.0f, 1.0f   // White
    };
    
    struct image_t test_output;
    if (rgb_to_jpeg(test_rgb, 2, 2, &test_output)) {
        printf("[Stream] Test JPEG conversion successful\n");
        free(test_output.buf);
    } else {
        printf("[Stream] Test JPEG conversion failed\n");
    }
}

void test_stream_setup(struct stream_context_t* ctx, const char* ip, int port) {
    printf("[Stream] Testing stream setup\n");
    init_stream(ctx, ip, port);
    test_jpeg_conversion(ctx);
    cleanup_stream(ctx);
    printf("[Stream] Stream setup test complete\n");
}

void test_full_streaming(struct stream_context_t* ctx) {
    printf("[Stream] Testing full streaming functionality\n");
    
    // Create a test gradient image
    int width = 64, height = 48;
    float* test_depth = malloc(width * height * sizeof(float));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            test_depth[y * width + x] = (float)x / width;
        }
    }
    
    // Test depth streaming
    stream_depth(ctx, test_depth, width, height, 0.0f, 1.0f);
    
    free(test_depth);
    printf("[Stream] Full streaming test complete\n");
}

void test_color_pattern(struct stream_context_t* ctx, int width, int height) {
    printf("[Stream] Creating test color pattern\n");
    
    // Create initial YUV422 (YUY2) image
    struct image_t yuv_input = {
        .w = width,
        .h = height,
        .type = IMAGE_YUV422,
        .buf = malloc(width * height * 2),  // YUV422 is 2 bytes per pixel
        .buf_size = width * height * 2
    };

    // Fill with a simple color pattern in YUV422 format
    // YUY2 format: [Y0 U Y1 V] [Y2 U Y3 V] ...
    uint8_t* yuv_buf = (uint8_t*)yuv_input.buf;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 2) {
            int idx = (y * width + x) * 2;
            if (x < width/3) {
                // Red in YUV
                yuv_buf[idx] = 76;     // Y0
                yuv_buf[idx+1] = 84;   // U
                yuv_buf[idx+2] = 76;   // Y1
                yuv_buf[idx+3] = 255;  // V
            } else if (x < 2*width/3) {
                // Green in YUV
                yuv_buf[idx] = 149;    // Y0
                yuv_buf[idx+1] = 43;   // U
                yuv_buf[idx+2] = 149;  // Y1
                yuv_buf[idx+3] = 21;   // V
            } else {
                // Blue in YUV
                yuv_buf[idx] = 29;     // Y0
                yuv_buf[idx+1] = 255;  // U
                yuv_buf[idx+2] = 29;   // Y1
                yuv_buf[idx+3] = 107;  // V
            }
        }
    }

    // Create output structure for RGB result
    struct image_t rgb_output = {0};
    
    // Process the YUV image through existing pipeline
    float* preprocessed_input = NULL;
    if (!preprocess_image(&yuv_input, &rgb_output, &preprocessed_input, width, height)) {
        printf("[Stream] Test pattern preprocessing failed\n");
        free(yuv_input.buf);
        return;
    }

    // Stream the preprocessed input
    stream_frame(ctx, preprocessed_input, width, height);

    // Cleanup
    free(yuv_input.buf);
    free(preprocessed_input);
    if (rgb_output.buf) free(rgb_output.buf);
}

void stream_frame(struct stream_context_t* ctx, float* rgb_data, int width, int height) {
    static int last_width = 0;
    static int last_height = 0;
    
    // Update SDP if resolution changes
    if (width != last_width || height != last_height) {
        create_sdp_file(width, height, ctx->port);  // Use stored port
        last_width = width;
        last_height = height;
    }

    // printf("[Stream] Streaming frame %dx%d\n", width, height);
    
    if (ctx->img_jpeg.buf != NULL) {
        free(ctx->img_jpeg.buf);
        ctx->img_jpeg.buf = NULL;
    }

    // Convert RGB float data to JPEG
    if (!rgb_to_jpeg(rgb_data, width, height, &ctx->img_jpeg)) {
        printf("[Stream] Failed to convert RGB to JPEG\n");
        return;
    }

    // Stream using RTP
    // printf("[Stream] Sending RTP frame of size %u bytes\n", ctx->img_jpeg.buf_size);
    rtp_frame_send(
        &ctx->video_sock,
        &ctx->img_jpeg,
        0,
        VIEWVIDEO_QUALITY_FACTOR,
        0,
        10,  // FPS
        &ctx->rtp_packet_nr,
        &ctx->rtp_frame_time
    );
    // printf("[Stream] Frame sent\n");
}

void stream_depth(struct stream_context_t* ctx, float* depth_map, int width, int height, 
                 float min_depth, float max_depth) {
    printf("[Stream] Converting depth map %dx%d\n", width, height);
    
    // Convert depth map to RGB using colormap
    float* rgb_data = malloc(width * height * 3 * sizeof(float));
    if (!rgb_data) {
        printf("[Stream] Failed to allocate RGB buffer for depth map\n");
        return;
    }
    
    for (int i = 0; i < width * height; i++) {
        // Normalize depth to [0,1] range
        float d = depth_map[i];
        float normalized = (d - min_depth) / (max_depth - min_depth);
        normalized = normalized < 0 ? 0 : (normalized > 1 ? 1 : normalized);
        
        // Map to colormap index
        int idx = (int)(normalized * 255);
        
        // Copy RGB values
        rgb_data[i*3] = ctx->colormap[idx*3] / 255.0f;
        rgb_data[i*3+1] = ctx->colormap[idx*3+1] / 255.0f;
        rgb_data[i*3+2] = ctx->colormap[idx*3+2] / 255.0f;
    }

    // Use the existing stream_frame function to send the colorized depth map
    stream_frame(ctx, rgb_data, width, height);
    free(rgb_data);
}