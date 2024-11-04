# Import necessary packages
import torch
from torchvision import transforms
from LymphoMNIST.LymphoMNIST import LymphoMNIST
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
from pathlib import Path
import datetime, os
from logger import logging
from poutyne.framework import Model
from poutyne.framework.callbacks import Callback
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from comet_ml import Experiment
from torchsummary import summary
from torch import optim

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


params = {"lr": 1e-5, 
          "batch_size": 16,
          "epochs": 10000, 
          "model": "Teacher_imsize-no-normalize",
          "im_size":120
          }

# Define comet experiment
experiment = Experiment(
    api_key="2iwTpjYhUb3dGr4yIiVtt1oRA",
    project_name="KD4FPGA",
    workspace="khayrulbuet13"
)
experiment.log_parameters(params)
experiment.set_name(params['model'])

# Check LymphoMNIST version
import LymphoMNIST as info
print(f"LymphoMNIST v{info.__version__} @ {info.HOMEPAGE}")

# Project class to manage directories
class Project:
    base_dir: Path = Path(__file__).resolve().parent
    data_dir = base_dir / 'dataset'
    checkpoint_dir = base_dir / 'checkpoint'

    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

# Comet.ml callback class
class CometCallback(Callback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs):
        self.experiment.log_metrics(logs, step=epoch)

# Dataset and data augmentation classes
class ConvertToRGB:
    def __call__(self, tensor):
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.8, iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)])),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.8, iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        if img.mode == 'L':
            img = img.convert("RGB")
        img = np.array(img)
        img = self.aug.augment_image(img)
        img = Image.fromarray(img).convert('L')
        return img

# Dataset class to filter by labels
class FilteredLymphoMNIST(Dataset):
    def __init__(self, original_dataset, labels_to_keep):
        self.original_dataset = original_dataset
        self.labels_to_keep = labels_to_keep
        self.label_map = {label: i for i, label in enumerate(labels_to_keep)}
        self.indices = [i for i, (_, label) in enumerate(original_dataset) if label in labels_to_keep]

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, label = self.original_dataset[original_index]
        return image, self.label_map[label.item()]

    def __len__(self):
        return len(self.indices)

# Balanced weights function for weighted sampling
def balanced_weights(dataset, nclasses):
    count = [0] * nclasses
    for _, label in dataset:
        count[label] += 1
    N = float(sum(count))
    weight_per_class = [N / float(count[i]) for i in range(nclasses)]
    return [weight_per_class[label] for _, label in dataset]

# Function to get dataloaders
def get_dataloaders(train_ds, val_ds, split=(0.5, 0.5), batch_size=64, sampler=None, *args, **kwargs):
    lengths = [int(len(val_ds) * frac) for frac in split]
    lengths[1] += len(val_ds) - sum(lengths)  # Correct split length sum
    val_ds, test_ds = torch.utils.data.random_split(val_ds, lengths)

    shuffle = False if sampler else True
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl

# Define transforms
im_size = params['im_size']
val_transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.4819], [0.1484]),
    ConvertToRGB()
])

train_transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    ImgAugTransform(),
    transforms.ToTensor(),
    transforms.Normalize([0.4819], [0.1484]),
    ConvertToRGB()
])

# The main execution logic
def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize dataset
    original_train_ds = LymphoMNIST(root='./dataset', train=True, download=True, transform=train_transform, num_classes=3)
    original_test_ds = LymphoMNIST(root='./dataset', train=False, download=True, transform=val_transform, num_classes=3)

    # Specify labels to keep
    labels_to_keep = [0, 1]  # Example: Keep two classes

    # Initialize filtered dataset
    train_ds = FilteredLymphoMNIST(original_train_ds, labels_to_keep)
    test_ds = FilteredLymphoMNIST(original_test_ds, labels_to_keep)

    # Compute balanced weights and create sampler
    weights = balanced_weights(train_ds, len(labels_to_keep))
    sampler = WeightedRandomSampler(weights, len(weights))

    # Create dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(train_ds, test_ds, split=(0.5, 0.5), batch_size=params["batch_size"], sampler=sampler, num_workers=4)

    # Load pre-trained ResNet50 model and modify it for single-channel input
    resnet50 = models.resnet50(weights='IMAGENET1K_V1')
    # resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, len(labels_to_keep))  # Adjust for the number of classes
    resnet50 = resnet50.to(device)

    # Print model summary
    summary(resnet50, (3, params["im_size"], params["im_size"]))

    # Define optimizer, model, and callbacks
    optimizer = optim.Adam(resnet50.parameters(), lr=params["lr"])
    model = Model(resnet50, optimizer, "cross_entropy", batch_metrics=["accuracy"]).to(device)

    # Define callbacks
    project = Project()
    checkpoint_dir = project.checkpoint_dir / f"{params['model']}_{datetime.datetime.now().strftime('%d %B %H:%M')}.pt"
    callbacks = [
        ReduceLROnPlateau(monitor="val_acc", patience=100, verbose=True),
        ModelCheckpoint(str(checkpoint_dir), save_best_only=True, verbose=True),
        EarlyStopping(monitor="val_acc", patience=100, mode='max'),
        CometCallback(experiment) 
    ]
    
    # Train model
    model.fit_generator(train_dl, val_dl, epochs=params["epochs"], callbacks=callbacks)

    # Save final model weights
    final_weights_path = project.checkpoint_dir / f"{params['model']}_final_weights.pt"
    model.save_weights(final_weights_path)
    logging.info(f"Model weights saved to {final_weights_path}")

    # Evaluate the model on the test set
    loss, test_acc = model.evaluate_generator(test_dl)
    logging.info(f'Test Accuracy: {test_acc}')
    experiment.log_metric('test_acc', test_acc)
    
    # End experiment
    experiment.end()

# Run the main function only if this script is executed directly
if __name__ == "__main__":
    main()
