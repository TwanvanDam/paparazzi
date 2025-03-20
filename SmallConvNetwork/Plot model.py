import torchvision.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import torch
from CNN import ObjectDetectionModel
import glob
from Image_utils import read_jpg_to_yuv

width = 520
height = 240
# This defines how wide the columns are
columns = [140/520, 220/520, 300/520, 380/520]

configs = [
    {"name" : "Baseline", "grid_lines": columns,"channels": [8, 16, 32],"kernel_size": [3, 3, 3],"padding": [1, 1, 1],
     "stride": [2, 2, 2], "pool_size": [2, 2, 2],"hidden_units": [128],"dropout": 0.2, "lr" : 0.00005},
    {"name" : "Baseline more fc", "grid_lines": columns,"channels": [8, 16, 32],"kernel_size": [3, 3, 3],"padding": [1, 1, 1],
     "stride": [2, 2, 2], "pool_size": [2, 2, 2],"hidden_units": [128,128],"dropout": 0.2, "lr" : 0.00005},
    {"name" : "More Channels","grid_lines": columns,"channels": [8, 16, 64],"kernel_size": [3, 3, 3],"padding": [1, 1, 1],
     "stride": [2, 1, 2], "pool_size": [2, 2, 2],"hidden_units": [128],"dropout": 0.1, "lr" : 0.00005},
    {"name" : "Bigger kernel","grid_lines": columns,"channels": [8, 16, 16],"kernel_size": [5, 3, 3],"padding": [2, 1, 1],
     "stride": [2, 1, 1], "pool_size": [2, 2, 2],"hidden_units": [128],"dropout": 0.1, "lr" : 0.00005},
    {"name": "Compact", "grid_lines": columns, "channels": [4, 8, 16], "kernel_size": [3, 3, 3], "padding": [1, 1, 1],
     "stride": [2, 2, 2], "pool_size": [2, 2, 2], "hidden_units": [64], "dropout": 0.1, "lr": 0.0001},

    {"name": "Shallow", "grid_lines": columns, "channels": [8, 16], "kernel_size": [3, 3], "padding": [1, 1],
     "stride": [2, 2], "pool_size": [2, 2], "hidden_units": [64], "dropout": 0.1, "lr": 0.0001},

    {"name": "Minimal", "grid_lines": columns, "channels": [4, 8], "kernel_size": [3, 3], "padding": [1, 1],
     "stride": [2, 2], "pool_size": [2, 2], "hidden_units": [32], "dropout": 0.1, "lr": 0.0001},

    {"name": "FastStride", "grid_lines": columns, "channels": [8, 16, 16], "kernel_size": [3, 3, 3], "padding": [1, 1, 1],
     "stride": [3, 2, 2], "pool_size": [2, 2, 2], "hidden_units": [64], "dropout": 0.1, "lr": 0.0001},

    {"name": "TinyKernels", "grid_lines": columns, "channels": [8, 16, 16], "kernel_size": [1, 3, 1], "padding": [0, 1, 0],
     "stride": [2, 2, 1], "pool_size": [2, 2, 2], "hidden_units": [64], "dropout": 0.1, "lr": 0.0001},
    {"name": "FastStrideMoreChannels", "grid_lines": columns, "channels": [8, 16, 32], "kernel_size": [3, 3, 3], "padding": [1, 1, 1],
     "stride": [3, 2, 2], "pool_size": [2, 2, 2], "hidden_units": [64], "dropout": 0.1, "lr": 0.0001}
]
config = configs[-1]

def plot_image(sample_image, sample_danger, grid_lines, grid=False):
    colors = [(0, "green"), (0.5, "orange"), (1, "red")]
    cmap = LinearSegmentedColormap.from_list("traffic_light", colors)

    sample_danger = sample_danger.detach().numpy()

    # Rotate image to display it correctly. Divide by 255 to scale pixel values to [0, 1].
    plt.imshow(torch.rot90(sample_image.squeeze(0), k=1, dims=[1,2]).permute(1, 2, 0)/255)

    # Overlay predicted danger levels.
    for i in range(len(grid_lines) - 1):
        plt.fill_between([grid_lines[i]*width, grid_lines[i + 1]*width],
                         [0, 0],
                         [height, height],
                         color=cmap(sample_danger[i]), alpha=0.5)
        plt.text((grid_lines[i]*width +grid_lines[i + 1]*width)/2 , height/2, f"{sample_danger[i]:.2f}", ha='center', va='center')


    plt.xlim(0, width)
    plt.ylim(height, 0)

model = ObjectDetectionModel.load_from_checkpoint("checkpoints/FastStrideMoreChannelsfold0-epoch=98-val_loss=0.0977.ckpt",
                                                  grid_lines=config["grid_lines"], channels=config["channels"],
                                                  kernel_size=config["kernel_size"], padding=config["padding"],
                                                  stride=config["stride"], pool_size=config["pool_size"],
                                                  hidden_units=config["hidden_units"], dropout=config["dropout"], lr=config["lr"])

# save the model to onnx


model.to_onnx("./SmallConvNetwork/model_small_clip.onnx", torch.randn(1, 3, height, height))

# plot a video to test the predictions
model.to("cpu")
model.eval()
torch.set_num_threads(1)

# Use the images from this folder
test_images = './SmallConvNetwork/Test_video/*.jpg'
image_paths = sorted(glob.glob(test_images))

fig, ax = plt.subplots()

def update(frame):
    ax.clear()  # clear the axes for the new frame
    image = torchvision.io.read_image(image_paths[frame]).float()

    image_model = torch.tensor(read_jpg_to_yuv(image_paths[frame]))

    # Record frame processing time
    # start_frame = time.time()
    frame_danger = model(image_model.unsqueeze(0)).squeeze(0)
    # print(f"{(time.time() - start_frame)*1000:.3f} ms")

    # Call the plot_image function to update the plot
    plot_image(sample_image=image, sample_danger=frame_danger, grid_lines=columns, grid=False)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    return ax

# Make the animation
animation_fps = 100
ani = animation.FuncAnimation(fig, update, frames=len(image_paths), interval=1000/animation_fps)

# if save_video:
#     # Save the animation to an MP4 file using ffmpeg writer
#     ani.save('./SmallConvNetwork/output.mp4', writer='ffmpeg', fps=animation_fps)

plt.show()