import numpy as np
import torchvision.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches


def read_bboxes(path:str)->np.ndarray:
    bboxes = open(path, "r").readlines()
    bboxes_processed = np.zeros((len(bboxes), 5))

    for i, bbox in enumerate(bboxes):
        bboxes_split = bbox.split(" ")
        x = float(bboxes_split[1])
        y = float(bboxes_split[2])
        w = float(bboxes_split[3])
        h = float(bboxes_split[4])
        label = int(bboxes_split[0])
        bboxes_processed[i,:] = [x, y, w, h, label]
    return bboxes_processed

def convert_coordinates(bbox:np.ndarray)->tuple[float, float, float, float, int]:
    # Convert from x, y, w, h to x1, y1, x2, y2
    x1 = bbox[0] - bbox[2]/2
    y1 = bbox[1] - bbox[3]/2
    x2 = bbox[0] + bbox[2]/2
    y2 = bbox[1] + bbox[3]/2
    label = int(bbox[4])
    return x1, x2, y1, y2, label

def danger_level(x1:float, x2:float, y1:float, y2:float, label:int, min_danger=0.0)->float:
    """
    Calculate the danger level of a bounding box based on its coordinates and label.
    :param x1: left x coordinate
    :param x2: right x coordinate
    :param y1: bottom y coordinate
    :param y2: top y coordinate
    :param label: type of object
        0 : Panel
        1 : Plant
        2 : Pole
        3 : blocks
    :return: danger level (0.0 - 1.0)
    """

    rel_height = y2 - y1
    rel_width = x2 - x1

    # Panel
    if label == 0:
        width_distance = rel_width / 0.25
        height_distance = rel_height / 0.9
        danger = min(max([width_distance, height_distance]), 1)

    # Plant
    if label == 1: # Plant
        area = rel_width*rel_height*8
        #width_distance = rel_width / 0.4
        #height_distance = rel_height / 0.7
        danger = min(area, 1)

    # Pole or blocks
    if (label == 2) or (label == 3):
        width_distance = rel_width / 0.15
        height_distance = min(rel_height / 1.1, 0.7)
        danger = min(max([width_distance, height_distance]), 1)

    return max(min_danger, danger)

def generate_danger_level_list(bboxes:np.ndarray, columns:list[float], augment_bboxes=True)->list[float]:
    """
    Generate a list of danger levels for each column in the grid.
    :param bboxes: objects in the image
    :param columns: location of the grid lines
    :param augment_bboxes: if True, augment the bboxes to make sure boxes only exist within the grid
    :return: list of danger levels for each column
    """

    danger_list = [0 for _ in range(len(columns) - 1)]

    # loop over all bboxes
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1, x2, y1, y2, label = convert_coordinates(bbox)
        if augment_bboxes:
            # augment the bbox to make sure boxes only exist within the grid
            x1 = max(columns[0], x1)
            x2 = min(columns[-1], x2)

        # calculate how dangerous given box is
        box_danger = danger_level(x1, x2, y1, y2, label)
        for j in range(len(columns) - 1):
            left = columns[j]
            right = columns[j + 1]
            # update the danger_list iff the box is in the given column and danger value is greater than current for that column
            if ((left <= x1 <= right) or (left <= x2 <= right)) or (x1 <= left and x2 >= right):
                danger_list[j] = max(danger_list[j], box_danger)
    return danger_list


if __name__ == "__main__":
    # Test the danger_level function
    test_images_dir = "./SmallConvNetwork/SyntheticData/dataset/images/train/122215661.jpg"

    colors = [(0, "green"), (0.5, "orange"), (1.0, "red")]
    cmap = LinearSegmentedColormap.from_list("traffic_light", colors)

    grid_lines = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    width = 520
    height = 240

    # Create a figure with 2 subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))

    # First subplot - bounding boxes with danger levels
    ax1.set_title("Bounding Boxes with Danger Levels")
    ax1.imshow(torchvision.io.read_image(test_images_dir).permute(1, 2, 0) / 255)
    labels = read_bboxes(test_images_dir.replace("images", "labels").replace(".jpg", ".txt"))
    for i in range(labels.shape[0]):
        bbox = labels[i]
        x1, x2, y1, y2, label = convert_coordinates(bbox)
        danger = danger_level(x1, x2, y1, y2, label)
        print(bbox, danger)
        rect = patches.Rectangle((x1 * 520, y1 * 240), (x2 - x1) * 520, (y2 - y1) * 240,
                                 facecolor=cmap(danger - 0.001), linewidth=2, alpha=0.5)
        ax1.add_patch(rect)
        ax1.text((x1 + x2) * 260, (y1 + y2) * 120, f"{danger:.2f}", ha='center', va='center')
    ax1.set_xlim(0, 520)
    ax1.set_ylim(240, 0)

    # Second subplot - column-wise danger levels
    ax2.set_title("Column Danger Levels")
    ax2.imshow(torchvision.io.read_image(test_images_dir).permute(1, 2, 0) / 255)
    labels = read_bboxes(test_images_dir.replace("images", "labels").replace(".jpg", ".txt"))
    sample_danger = generate_danger_level_list(labels, grid_lines)
    for i in range(len(grid_lines) - 1):
        ax2.fill_between([grid_lines[i] * width, grid_lines[i + 1] * width],
                         [0, 0],
                         [height, height],
                         color=cmap(sample_danger[i] - 0.0001), alpha=0.5)
        ax2.text((grid_lines[i] * width + grid_lines[i + 1] * width) / 2, height / 2,
                 f"{sample_danger[i]:.2f}", ha='center', va='center')
    ax2.set_xlim(0, 520)
    ax2.set_ylim(240, 0)

    # First apply tight_layout to properly position the subplots
    plt.tight_layout()

    # Get position of the bottom subplot to align colorbar width
    pos = ax2.get_position()

    # Add a colorbar at the bottom with same width as plots
    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([pos.x0, 0.05, pos.width, 0.02])  # Match subplot width
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Danger Level")

    # Final adjustment for margins
    fig.subplots_adjust(bottom=0.08, top=0.95)
    plt.savefig("./SmallConvNetwork/danger_levels.pdf", dpi=300, bbox_inches='tight')
    plt.show()