import matplotlib.pyplot as plt

from Danger_Calculation import generate_danger_level_list, read_bboxes
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import time
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import random
from Image_utils import adjust_brightness_contrast

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, width, height, grid_lines):
        self.image_paths = sorted(glob.glob(image_dir + "/images_raw/*.raw"))
        self.label_paths = sorted(glob.glob(image_dir + "/labels/*.txt"))
        self.width = width
        self.height = height
        self.grid_lines = grid_lines

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with open(self.image_paths[idx], 'rb') as f:
            image = np.frombuffer(f.read(), dtype=np.float32).reshape((3,240, 240))
        bboxes = read_bboxes(self.label_paths[idx])
        cell_danger_levels = generate_danger_level_list(bboxes, self.grid_lines)

        if random.random() < 0.1:
            image = adjust_brightness_contrast(image, random.randint(-10, 10)/50, random.randint(-10, 10)/50)

        image = torch.tensor(image).float()
        # Add random flipping with probability 0.1
        if random.random() < 0.1:
            image = torch.flip(image, dims=[1])  # Flip over x-axis (horizontal)
            cell_danger_levels = cell_danger_levels[::-1]  # Adjust danger levels accordingly


        return image, torch.tensor(cell_danger_levels).float()

class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, grid_lines, channels, kernel_size, padding, stride, pool_size, hidden_units, dropout, lr, width=520, heigth=240, flip=True, brightness_contrast=True):
        super(ObjectDetectionModel, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.verbose = True
        self.grid_lines = grid_lines
        self.xlim_left = int(self.grid_lines[0] * width)
        self.xlim_right = int(self.grid_lines[-1] * width)

        layers_conv = []
        for i in range(len(channels)):
            if i == 0:
                layers_conv.append(torch.nn.Conv2d(3, channels[i], kernel_size[i], padding=padding[i], stride=stride[i]))
            else:
                layers_conv.append(torch.nn.Conv2d(channels[i - 1], channels[i], kernel_size[i], padding=padding[i], stride=stride[i]))
            layers_conv.append(torch.nn.MaxPool2d(pool_size[i]))
            layers_conv.append(torch.nn.ReLU())

        layers_fc = []
        for i in range(len(hidden_units)):
            if i == 0:
                layers_fc.append(torch.nn.Linear(128,hidden_units[i]))

            else:
                if len(hidden_units) > 1:
                    layers_fc.append(torch.nn.Linear(hidden_units[i - 1], hidden_units[i]))
            layers_fc.append(torch.nn.ReLU())
            layers_fc.append(torch.nn.Dropout(dropout))
            if i == len(hidden_units) - 1:
                layers_fc.append(torch.nn.Linear(hidden_units[i], len(grid_lines)-1))


        self.conv = torch.nn.Sequential(*layers_conv)

        self.fc = torch.nn.Sequential(*layers_fc)

        self.loss_fn = torch.nn.MSELoss()

    def custom_relu(self, x):
        x = torch.maximum(x, torch.tensor(0.0))
        x = torch.minimum(x, torch.tensor(1.0))
        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = x.view(x.size(0), len(self.grid_lines) - 1)
        x = torch.clip(x, 0.0, 1.0)
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    training_dir = "/home/twan/YOLO_dataset_generated"

    # these are the dimensions of the image when it is rotated by 90 degrees, so it is displayed correctly
    width = 520
    height = 240

    # Option for running the file
    train = True
    save_model = False
    save_video = False

    # Parameters
    n_splits = 5
    batch_size = 8
    max_epochs = 100
    random_state = 42
    num_tests = 100

    # This defines how wide the columns are
    columns = [140/520, 220/520, 300/520, 380/520]

    configs = [
        #  {"name" : "Baseline", "grid_lines": columns,"channels": [8, 16, 32],"kernel_size": [3, 3, 3],"padding": [1, 1, 1],
        #           "stride": [2, 2, 2], "pool_size": [2, 2, 2],"hidden_units": [128],"dropout": 0.2, "lr" : 0.0001},
        # {"name" : "Baseline more fc", "grid_lines": columns,"channels": [8, 16, 32],"kernel_size": [3, 3, 3],"padding": [1, 1, 1],
        #  "stride": [2, 2, 2], "pool_size": [2, 2, 2],"hidden_units": [128,128],"dropout": 0.2, "lr" : 0.0001},
        #  {"name" : "More Channels","grid_lines": columns,"channels": [8, 16, 64],"kernel_size": [3, 3, 3],"padding": [1, 1, 1],
        #   "stride": [2, 1, 2], "pool_size": [2, 2, 2],"hidden_units": [128],"dropout": 0.1, "lr" : 0.0001},
        #  {"name" : "Bigger kernel","grid_lines": columns,"channels": [8, 16, 16],"kernel_size": [5, 3, 3],"padding": [2, 1, 1],
        #   "stride": [2, 1, 1], "pool_size": [2, 2, 2],"hidden_units": [128],"dropout": 0.1, "lr" : 0.0001},
        # {"name": "Compact", "grid_lines": columns, "channels": [4, 8, 16], "kernel_size": [3, 3, 3], "padding": [1, 1, 1],
        #  "stride": [2, 2, 2], "pool_size": [2, 2, 2], "hidden_units": [64], "dropout": 0.1, "lr": 0.0001},
        #
        # {"name": "Shallow", "grid_lines": columns, "channels": [8, 16], "kernel_size": [3, 3], "padding": [1, 1],
        #  "stride": [2, 2], "pool_size": [2, 2], "hidden_units": [64], "dropout": 0.1, "lr": 0.0001},
        #
        # {"name": "Minimal", "grid_lines": columns, "channels": [4, 8], "kernel_size": [3, 3], "padding": [1, 1],
        #  "stride": [2, 2], "pool_size": [2, 2], "hidden_units": [32], "dropout": 0.1, "lr": 0.0001},

        # {"name": "FastStride", "grid_lines": columns, "channels": [8, 16, 16], "kernel_size": [3, 3, 3], "padding": [1, 1, 1],
        #  "stride": [3, 2, 2], "pool_size": [2, 2, 2], "hidden_units": [64], "dropout": 0.1, "lr": 0.0001},
        {"name": "FastStrideMoreChannels", "grid_lines": columns, "channels": [8, 16, 32], "kernel_size": [3, 3, 3], "padding": [1, 1, 1],
         "stride": [3, 2, 2], "pool_size": [2, 2, 2], "hidden_units": [128], "dropout": 0.1, "lr": 0.0001},
        # {"name": "FastStrideMoreMoreChannels", "grid_lines": columns, "channels": [8, 16, 64], "kernel_size": [3, 3, 3], "padding": [1, 1, 1],
        #  "stride": [3, 2, 2], "pool_size": [2, 2, 2], "hidden_units": [64], "dropout": 0.1, "lr": 0.0001},
        #{"name": "FastStrideBigKernel", "grid_lines": columns, "channels": [8, 16, 32], "kernel_size": [3, 5, 5], "padding": [1, 2, 2],
        #  "stride": [3, 2, 2], "pool_size": [2, 2, 2], "hidden_units": [128], "dropout": 0.1, "lr": 0.0001},

        # {"name": "TinyKernels", "grid_lines": columns, "channels": [8, 16, 16], "kernel_size": [1, 3, 1], "padding": [0, 1, 0],
        #  "stride": [2, 2, 1], "pool_size": [2, 2, 2], "hidden_units": [64], "dropout": 0.1, "lr": 0.0001},
        # {"name": "ExtendedModel", "grid_lines": columns, "channels": [8, 16, 32, 64], "kernel_size": [3, 3, 3, 3],
        #  "padding": [1, 1, 1, 1], "stride": [2, 2, 2, 2], "pool_size": [2, 2, 2, 2],
        #  "hidden_units": [128, 64], "dropout": 0.2, "lr": 0.00005},
        #
        # {"name": "ExtraDeep", "grid_lines": columns, "channels": [8, 16, 32, 64, 128], "kernel_size": [3, 3, 3, 3, 3],
        #  "padding": [1, 1, 1, 1, 1], "stride": [2, 2, 2, 2, 2], "pool_size": [2, 2, 2, 2, 2],
        #  "hidden_units": [256, 128, 64], "dropout": 0.3, "lr": 0.00005},
        #
        # {"name": "LiteModel", "grid_lines": columns, "channels": [4, 8, 16], "kernel_size": [3, 3, 3],
        #  "padding": [1, 1, 1], "stride": [2, 2, 2], "pool_size": [2, 2, 2],
        #  "hidden_units": [32], "dropout": 0.1, "lr": 0.0002},
    ]


    full_dataset = CustomImageDataset(training_dir, width, height, columns)
    # for i in range(len(full_dataset)):
    #     print(full_dataset[i][1])
    #     plt.imshow(full_dataset[i][0][0,:,:], cmap='gray')
    #     plt.show()

    # Initialize KFold configuration
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Store results for each fold
    results = [{"name" : configs[i]["name"], "val_loss" : [], "epochs" : [], "inference" : []} for i in range(len(configs))]

    for i,config in enumerate(configs):
        for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(full_dataset)))):
            print(f"Training fold {fold + 1}/{n_splits}")

            # Create subset datasets for training and validation
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',         # metric to monitor
                dirpath='checkpoints/',     # directory to save checkpoints
                filename=f'{config["name"]}fold{fold}_synthethic'+ '-{epoch:02d}-{val_loss:.4f}',  # checkpoint filename format
                save_top_k=1,              # save only the best checkpoint
                mode='min',                # minimize the monitored metric
            )

            # fit the model

            print(f"Training model {i}")
            print(f"Model configuration: {config}")
            model = ObjectDetectionModel(grid_lines=config["grid_lines"], channels=config["channels"],
                                         kernel_size=config["kernel_size"], padding=config["padding"],
                                         stride=config["stride"], pool_size=config["pool_size"],
                                         hidden_units=config["hidden_units"], dropout=config["dropout"], lr=config["lr"],
                                         width=width, heigth=height, flip=True, brightness_contrast=True)
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                check_val_every_n_epoch=1,
                enable_progress_bar=True,
                callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=5)],
            )
            # Run training for this fold
            trainer.fit(model, DataLoader(train_subset, batch_size=batch_size), DataLoader(val_subset, batch_size=batch_size))
            results[i]["val_loss"].append(checkpoint_callback.best_model_score.cpu().item())
            results[i]["epochs"].append(trainer.current_epoch)
            print("Epochs used: ", trainer.current_epoch)
            print("Best Validation Loss:", checkpoint_callback.best_model_score)

            # test inference time:
            print("Testing inference time")
            start = time.time()
            model.eval()
            model.to("cpu")
            torch.set_num_threads(1)
            for n in range(num_tests):
                sample_image = val_subset[n][0].unsqueeze(0)
                output = model(sample_image)
            results[i]["inference"].append((time.time() - start) *1000 / num_tests)

        print(results[i])

    for i, config in enumerate(configs):
        print(f"Configuration: {config['name']}")
        print(f"Validation Loss: {np.mean(results[i]['val_loss'])} +- {np.std(results[i]['val_loss'])}")
        print(f"Epochs: {np.mean(results[i]['epochs'])} +- {np.std(results[i]['epochs'])}")
        print(f"Inference time: {np.mean(results[i]['inference'])} ms +- {np.std(results[i]['inference'])}")





