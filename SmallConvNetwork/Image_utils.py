import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from pathlib import Path

def jpeg_to_uyvy(jpg_path, target_width, target_height):
    """Convert JPEG to UYVY format"""
    img = cv2.imread(jpg_path)
    if img is None:
        print(f"Failed to read {jpg_path}")
        return None

    aspect = img.shape[1] / img.shape[0]
    target_aspect = target_width / target_height

    if aspect > target_aspect:
        new_width = int(target_height * aspect)
        img = cv2.resize(img, (new_width, target_height))
        start = (new_width - target_width) // 2
        img = img[:, start:start+target_width]
    else:
        new_height = int(target_width / aspect)
        img = cv2.resize(img, (target_width, new_height))
        start = (new_height - target_height) // 2
        img = img[start:start+target_height, :]

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    uyvy = np.zeros((target_height, target_width * 2), dtype=np.uint8)

    uyvy[:, 1::2] = yuv[:, :, 0]  # Y values

    u = yuv[:, :, 1]
    v = yuv[:, :, 2]

    u_sub = u[:, ::2]
    v_sub = v[:, ::2]

    uyvy[:, 0::4] = u_sub
    uyvy[:, 2::4] = v_sub

    return uyvy

def crop_yuv(yuv_data, width, height):
    center_w = yuv_data.shape[1] // 2
    center_h = yuv_data.shape[2] // 2
    left_w = center_w - width // 2
    right_w = center_w + width // 2
    top_h = center_h + height // 2
    bottom_h = center_h - height // 2

    # print(left_w, right_w, bottom_h, top_h)
    return yuv_data[:,left_w:right_w,bottom_h:top_h]

def reshape_uyvy_to_yuv(uyvy_data, downscale_factor=1):
    """
    Reshape 120x120 UYVY data into YUV channels with optional downscaling.
    Input: raw UYVY data (240x480 uint8 array, since each pixel needs 2 bytes)
    Output: (3, output_height, output_width) float32 array with separate Y, U, V channels, normalized to [0,1]
    """
    if downscale_factor not in [1, 2, 4]:
        raise ValueError("Downscale factor must be 1, 2, or 4")

    height = uyvy_data.shape[0]

    # When downscale_factor is 1, we decode the UYVY data directly.
    if downscale_factor == 1:
        # Each row length in bytes
        row_bytes = uyvy_data.shape[1]
        # Each pair of pixels occupies 4 bytes, so number of pixels per row:
        pixel_count = row_bytes // 2
        y = np.zeros((height, pixel_count), dtype=np.float32)
        u = np.zeros((height, pixel_count), dtype=np.float32)
        v = np.zeros((height, pixel_count), dtype=np.float32)
        for i in range(height):
            pixel_idx = 0
            # Process two pixels at a time: [U, Y0, V, Y1]
            for j in range(0, row_bytes, 4):
                U_val = uyvy_data[i, j]
                Y0 = uyvy_data[i, j + 1]
                V_val = uyvy_data[i, j + 2]
                Y1 = uyvy_data[i, j + 3]
                # First pixel of the pair
                y[i, pixel_idx] = Y0 / 255.0
                u[i, pixel_idx] = U_val / 255.0 - 0.5
                v[i, pixel_idx] = V_val / 255.0 - 0.5
                pixel_idx += 1
                # Second pixel of the pair
                y[i, pixel_idx] = Y1 / 255.0
                u[i, pixel_idx] = U_val / 255.0 - 0.5
                v[i, pixel_idx] = V_val / 255.0 - 0.5
                pixel_idx += 1
        yuv = np.stack([y, u, v])
        return yuv

    # For downscale_factor of 2 or 4, use the block-averaging method.
    # In this branch, we assume the full resolution is 240x240 pixels.
    full_width = 240
    full_height = 240
    out_height = full_height // downscale_factor
    out_width = full_width // downscale_factor

    # Reshape input to 2D array where each row represents byte values in UYVY format.
    # Here the expected row length is full_width * 2.
    uyvy = uyvy_data.reshape(full_height, full_width * 2)

    y = np.zeros((out_height, out_width), dtype=np.float32)
    u = np.zeros((out_height, out_width), dtype=np.float32)
    v = np.zeros((out_height, out_width), dtype=np.float32)

    for out_y in range(out_height):
        for out_x in range(out_width):
            y_sum = 0
            u_sum = 0
            v_sum = 0
            count = 0
            # Process block of size downscale_factor x downscale_factor pixels
            for dy in range(downscale_factor):
                in_y = out_y * downscale_factor + dy
                for dx in range(downscale_factor):
                    in_x = out_x * downscale_factor + dx
                    byte_idx = in_x * 2
                    # Determine if current pixel is even or odd (in pair)
                    if in_x % 2 == 0:  # even index in pair
                        y_val = uyvy[in_y, byte_idx + 1]
                        u_val = uyvy[in_y, byte_idx]
                        v_val = uyvy[in_y, byte_idx + 2]
                    else:  # odd pixel in pair gets U and V from previous even pixel
                        y_val = uyvy[in_y, byte_idx + 1]
                        u_val = uyvy[in_y, byte_idx - 2]
                        v_val = uyvy[in_y, byte_idx]
                    y_sum += y_val
                    u_sum += u_val
                    v_sum += v_val
                    count += 1
            y[out_y, out_x] = (y_sum / count) / 255.0
            u[out_y, out_x] = (u_sum / count) / 255.0 - 0.5
            v[out_y, out_x] = (v_sum / count) / 255.0 - 0.5

    yuv = np.stack([y, u, v])
    return yuv

def read_jpg_to_yuv(jpg_path):
    uyvy = jpeg_to_uyvy(jpg_path, 240, 520) # image is rotated by 90 degrees
    yuv = reshape_uyvy_to_yuv(uyvy)
    if uyvy is None:
        return None
    return crop_yuv(yuv, 240, 240)

def adjust_brightness_contrast(image, brightness, contrast):
    """
    Adjust brightness and contrast for float32 image
    image shape: (3, 240, 240) with values in [0,1]
    """
    # Only adjust Y channel (luminance)
    y_channel = image[0].copy()

    # Apply contrast
    contrast_factor = (1 + contrast)
    y_channel = (y_channel - 0.5) * contrast_factor + 0.5

    # Apply brightness (scaled to [-0.5, 0.5] range for float32)
    brightness_factor = brightness / 255.0
    y_channel = y_channel + brightness_factor

    # Clip values to [0,1]
    y_channel = np.clip(y_channel, 0, 1)

    # Update Y channel
    image[0] = y_channel

    return image

def process_directory(jpg_dir, output_dir):
    """Process all JPEGs in a directory"""
    jpg_dir = Path(jpg_dir).resolve()
    print(f"Processing JPEGs from: {jpg_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for jpg_file in jpg_dir.glob("*.jpg"):
        yuv_data = read_jpg_to_yuv(str(jpg_file)).astype(np.float32)
        print(yuv_data.shape)
        if yuv_data is not None:
            output_path = output_dir / (jpg_file.stem + '.raw')  # Changed to .raw
            yuv_data.tofile(output_path)
            print(f"Converted {jpg_file} to {output_path}")

if __name__ == "__main__":
    start = time.time()
    image_path = "SmallConvNetwork/dataset/images/val/177250905.jpg"
    yuv = read_jpg_to_yuv(image_path)
    process_directory("./SmallConvNetwork/dataset/images/train", "./SmallConvNetwork/dataset_raw/images/train")
    print(time.time()-start)
    with open(image_path.replace(".jpg", ".raw").replace("dataset", "dataset_raw"), 'rb') as f:
        yuv = np.frombuffer(f.read(), dtype=np.float32).reshape((3,240, 240))

    plt.imshow(yuv[0,:,:], cmap='gray')
    plt.show()