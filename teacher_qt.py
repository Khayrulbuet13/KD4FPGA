# %%

# Lets import the packages
import torch
from torchvision import transforms
from LymphoMNIST.LymphoMNIST import LymphoMNIST

import torch.nn as nn
from torchvision import models
from torchsummary import summary


import time
from comet_ml import Experiment
import torch.optim as optim
from torchsummary import summary

from poutyne.framework import Model
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from logger import logging
import datetime, os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# check LymphoMNIST virsion
import LymphoMNIST as info
print(f"LymphoMNIST v{info.__version__} @ {info.HOMEPAGE}")
from torch.utils.data import WeightedRandomSampler

#%%
from dataclasses import dataclass
from pathlib import Path
class Project:
    """
    This class represents our project. It stores useful information about the structure, e.g. paths.
    """
    base_dir: Path = Path(__file__).parents[0]
    data_dir = base_dir / 'dataset'
    checkpoint_dir = base_dir / 'checkpoint'

    def __post_init__(self):
        # create the directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)


from poutyne.framework.callbacks import Callback
class CometCallback(Callback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs):
        self.experiment.log_metrics(logs, step=epoch)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import numpy as np
import torchvision.transforms as T
from imgaug import augmenters as iaa
from PIL import Image



class ConvertToRGB:
    """
    Convert 1-channel tensors to 3-channel tensors by duplicating the channel 3 times.
    """
    def __call__(self, tensor):
        # Check if the tensor is 1-channel (C, H, W) where C == 1
        if tensor.shape[0] == 1:
            # Duplicate the channel 3 times
            tensor = tensor.repeat(3, 1, 1)
        return tensor



class ImgAugTransform:
    """
    Wrapper to allow imgaug to work with Pytorch transformation pipeline, modified for 1-channel images.
    """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.8, iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])),
            iaa.Sometimes(0.5, iaa.Sequential([
                iaa.Crop(percent=(0.1, 0.2))
            ])),
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
        # Convert 1-channel images to 3-channel images for augmentation
        if img.mode == 'L':
            img = img.convert("RGB")
        img = np.array(img)
        img = self.aug.augment_image(img)
        # Convert back to 1-channel image if originally it was
        img = Image.fromarray(img).convert('L')
        return img

im_size = 28

val_transform = T.Compose([
    T.Resize((im_size, im_size)),
    T.ToTensor(),
    T.Normalize([0.4819], [0.1484]),  # Adjusted for 1-channel
    # ConvertToRGB()
])

train_transform = T.Compose([
    T.Resize((im_size, im_size)),
    ImgAugTransform(),
    T.ToTensor(),
    T.Normalize([0.4819], [0.1484]),  # Adjusted for 1-channel
    # ConvertToRGB([])
])



from torch.utils.data import Dataset


class FilteredLymphoMNIST(Dataset):
    def __init__(self, original_dataset, labels_to_keep):
        self.original_dataset = original_dataset
        self.labels_to_keep = labels_to_keep
        self.label_map = {label: i for i, label in enumerate(labels_to_keep)}  # Map original labels to new labels
        self.indices = [i for i, (_, label) in enumerate(original_dataset) if label in labels_to_keep]

    def __getitem__(self, index):
        if index >= len(self.indices):
            raise IndexError("Index out of range")
        original_index = self.indices[index]
        image, label = self.original_dataset[original_index]
        # Remap the label
        return image, self.label_map[label.item()]

    def __len__(self):
        return len(self.indices)





def balanced_weights(dataset, nclasses):
    # Count each class's occurrences in the dataset
    count = [0] * nclasses
    for _, label in dataset:
        count[label] += 1
    # Calculate the weight for each class based on their occurrences
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    # Assign weight to each sample based on its class
    weights = [weight_per_class[label] for _, label in dataset]
    return weights


import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from logger import logging


def get_dataloaders(
        train_ds,
        val_ds,
        split=(0.5, 0.5),
        batch_size=64,
        sampler=None, 
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """
    lengths = [int(len(val_ds) * frac) for frac in split]
    lengths[1] += len(val_ds) - sum(lengths)  # Correct split length sum
    val_ds, test_ds = random_split(val_ds, lengths)
    
    # print the lengths of the datasets
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    shuffle = False if sampler else True

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,  *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl




# our hyperparameters
params = {
    'lr': 1e-5,
    'batch_size': 16,
    'epochs': 10000,
    'model': "Teacher_MNIST"
}


# Load MNIST dataset
from torchvision import datasets, transforms
train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=val_transform)

# Split validation and test data
val_size = int(len(test_dataset)*0.5)
test_size = len(test_dataset) - val_size
test_dataset, val_dataset = random_split(test_dataset, [test_size, val_size])

# Create data loaders
train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_dl = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_dl = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)



# print one image shape
for x, y in train_dl:
    print("train",x.shape)
    break
for x, y in val_dl:
    print("val",x.shape)
    break
#%%


# Load a pre-trained ResNet50 model

# Define your QuantizedCNN model
class QuantizedCNN(nn.Module):
    def __init__(self, num_classes=2, input_size=(1, 28, 28)):
        super(QuantizedCNN, self).__init__()
        self.num_classes = num_classes

        # Define the convolutional layers and pooling layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        # Initialize the features to pass a dummy input through to find number of feature outputs
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            dummy_output = self.features(dummy_input)
            num_ftrs = dummy_output.numel() // dummy_output.size(0)  # Calculate total feature number dynamically

        # Redefine the classifier part of the network
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

num_classes =10
data_iter = iter(train_dl)
samples, targets = next(data_iter)
input_size = samples.shape[1:]
cnn =  QuantizedCNN(num_classes=num_classes, input_size=input_size).to(device)


# Adjust the input size if necessary
summary(cnn, input_size=input_size)


project = Project()

logging.info(f'Using device={device} 🚀')

model_name = params['model']

# define our comet experiment
experiment = Experiment(
    api_key="2iwTpjYhUb3dGr4yIiVtt1oRA",
    project_name="TvsB-ablation",
    # project_name="",
    workspace="khayrulbuet13")

experiment.log_parameters(params)
experiment.set_name(model_name)




# define custom optimizer and instantiace the trainer `Model`
optimizer = optim.Adam(cnn.parameters(), lr=params['lr'])
model = Model(cnn, optimizer, "cross_entropy",
                batch_metrics=["accuracy"]).to(device)
# usually you want to reduce the lr on plateau and store the best model
callbacks = [
    ReduceLROnPlateau(monitor="val_acc", patience=100, verbose=True),
    ModelCheckpoint(str(project.checkpoint_dir /
                        f"""{model_name}_{datetime.datetime.now().strftime('%d %B %H:%M')}.pt"""), save_best_only="True", verbose=True),
    EarlyStopping(monitor="val_acc", patience=100, mode='max'),
    CometCallback(experiment)
]

model.fit_generator(
    train_dl,
    val_dl,
    epochs=params['epochs'],
    callbacks=callbacks,
)


# Save the final model weights after training with Poutyne
model_weights_path = os.path.join(project.checkpoint_dir, f"{datetime.datetime.now().strftime('%d %B %H:%M')}_{model_name}_final_weights.pt")
model.save_weights(model_weights_path)
logging.info(f"Model weights saved to {model_weights_path}")



# get the results on the test set
loss, test_acc = model.evaluate_generator(test_dl)
logging.info(f'test_acc=({test_acc})')
experiment.log_metric('test_acc', test_acc)


# end the experiment
experiment.end()
