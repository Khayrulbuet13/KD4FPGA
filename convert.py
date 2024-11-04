
# Convert weight Lazy way

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Flatten, Dropout, Dense, Softmax, Input

# Define PyTorch model
class QuantizedCNN(nn.Module):
    def __init__(self, num_classes=2, input_size=(1, 48, 48)):
        super(QuantizedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        num_ftrs = 16 * 10 * 10
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

# Initialize PyTorch model
STUDENT_IM_SIZE = 48
num_classes = 2
pytorch_model = QuantizedCNN(num_classes=num_classes, input_size=(1, STUDENT_IM_SIZE, STUDENT_IM_SIZE))
pytorch_model.load_state_dict(torch.load('checkpoint/Final_models/KD_13 October 00:13_resnet50_qt-1channel-imsize-120-48.pt', map_location='cpu'))
pytorch_model.eval()

# Extract weights from PyTorch model
state_dict = pytorch_model.state_dict()
conv1_weight = state_dict['features.0.weight'].cpu().numpy()
conv1_bias = state_dict['features.0.bias'].cpu().numpy()
conv2_weight = state_dict['features.3.weight'].cpu().numpy()
conv2_bias = state_dict['features.3.bias'].cpu().numpy()
dense_weight = state_dict['classifier.2.weight'].cpu().numpy()
dense_bias = state_dict['classifier.2.bias'].cpu().numpy()

# Transpose weights for Keras
conv1_weight = np.transpose(conv1_weight, (2, 3, 1, 0))
conv2_weight = np.transpose(conv2_weight, (2, 3, 1, 0))
dense_weight = dense_weight.T

from tensorflow.keras.layers import Permute
# Define Keras model
def create_keras_model(input_shape=(48, 48, 1), num_classes=2):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(16, kernel_size=3, strides=1, padding='valid'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Conv2D(16, kernel_size=3, strides=1, padding='valid'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Permute((3, 1, 2)))  # Rearrange axes to (C, H, W)
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Softmax())
    return model


keras_model = create_keras_model(input_shape=(STUDENT_IM_SIZE, STUDENT_IM_SIZE, 1), num_classes=num_classes)
print(keras_model.summary())


# Print layer indices and names to confirm
for idx, layer in enumerate(keras_model.layers):
    print(f"Layer {idx}: {layer.name} ({layer.__class__.__name__})")

# Assign weights to the correct layers
keras_model.layers[0].set_weights([conv1_weight, conv1_bias])  # Conv1 (Layer 0)
keras_model.layers[3].set_weights([conv2_weight, conv2_bias])  # Conv2 (Layer 3)
keras_model.layers[-2].set_weights([dense_weight, dense_bias])  # Dense (Layer 9)




# Prepare input data
input_data = np.random.randn(1, 1, STUDENT_IM_SIZE, STUDENT_IM_SIZE).astype(np.float32)
pytorch_input = torch.from_numpy(input_data)
keras_input = np.transpose(input_data, (0, 2, 3, 1))  # Convert NCHW to NHWC

# Get outputs from both models
pytorch_output = pytorch_model(pytorch_input).detach().numpy()
keras_output = keras_model.predict(keras_input)

# Compare outputs
difference = np.abs(pytorch_output - keras_output)
print('PyTorch output:', pytorch_output)
print('Keras output:', keras_output)
print('Max difference:', difference.max())

keras_model.save('converted_keras_model_fixed.h5')


# Advanced way

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Flatten, Dropout, Dense, Softmax, Input

# Define PyTorch model
class QuantizedCNN(nn.Module):
    def __init__(self, num_classes=2, input_size=(1, 48, 48)):
        super(QuantizedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # Dynamically compute the number of features
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            dummy_output = self.features(dummy_input)
            num_ftrs = dummy_output.numel() // dummy_output.size(0)
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

# Initialize PyTorch model
STUDENT_IM_SIZE = 48
num_classes = 2
pytorch_model = QuantizedCNN(num_classes=num_classes, input_size=(1, STUDENT_IM_SIZE, STUDENT_IM_SIZE))
pytorch_model.load_state_dict(torch.load('checkpoint/Final_models/KD_13 October 00:13_resnet50_qt-1channel-imsize-120-48.pt', map_location='cpu'))
pytorch_model.eval()




# Extract weights from PyTorch model
state_dict = pytorch_model.state_dict()
conv1_weight = state_dict['features.0.weight'].cpu().numpy()
conv1_bias = state_dict['features.0.bias'].cpu().numpy()
conv2_weight = state_dict['features.3.weight'].cpu().numpy()
conv2_bias = state_dict['features.3.bias'].cpu().numpy()
dense_weight = state_dict['classifier.2.weight'].cpu().numpy()
dense_bias = state_dict['classifier.2.bias'].cpu().numpy()

# Transpose weights for Keras Conv2D layers
conv1_weight = np.transpose(conv1_weight, (2, 3, 1, 0))
conv2_weight = np.transpose(conv2_weight, (2, 3, 1, 0))

# Compute the output dimensions after the feature extractor
with torch.no_grad():
    dummy_input = torch.zeros(1, 1, STUDENT_IM_SIZE, STUDENT_IM_SIZE)
    dummy_output = pytorch_model.features(dummy_input)
    _, C, H, W = dummy_output.shape
print(f"Features output shape: C={C}, H={H}, W={W}")

# Adjust the Dense layer weights to match Keras flattening order
# In PyTorch, the flattening order is (C, H, W)
# In Keras, the flattening order is (H, W, C)
dense_weight = dense_weight.reshape(num_classes, C, H, W)
dense_weight = np.transpose(dense_weight, (0, 2, 3, 1))  # (num_classes, H, W, C)
dense_weight = dense_weight.reshape(num_classes, -1)       # Flatten to (num_classes, H*W*C)
dense_weight = dense_weight.T                              # Transpose to (H*W*C, num_classes)

# Define Keras model without Permute layer
def create_keras_model(input_shape=(48, 48, 1), num_classes=2):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(16, kernel_size=3, strides=1, padding='valid'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Conv2D(16, kernel_size=3, strides=1, padding='valid'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    # Removed Permute layer
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Softmax())
    return model

keras_model = create_keras_model(input_shape=(STUDENT_IM_SIZE, STUDENT_IM_SIZE, 1), num_classes=num_classes)
# Print layer indices and names to confirm
for idx, layer in enumerate(keras_model.layers):
    print(f"Layer {idx}: {layer.name} ({layer.__class__.__name__})")

# Assign weights to the correct layers
keras_model.layers[0].set_weights([conv1_weight, conv1_bias])  # Conv1 (Layer 1)
keras_model.layers[3].set_weights([conv2_weight, conv2_bias])  # Conv2 (Layer 4)
keras_model.layers[8].set_weights([dense_weight, dense_bias])   # Dense (Layer 9)

# Prepare input data
input_data = np.random.randn(1, 1, STUDENT_IM_SIZE, STUDENT_IM_SIZE).astype(np.float32)
pytorch_input = torch.from_numpy(input_data)
keras_input = np.transpose(input_data, (0, 2, 3, 1))  # Convert NCHW to NHWC

# Get outputs from both models
pytorch_output = pytorch_model(pytorch_input).detach().numpy()
keras_output = keras_model.predict(keras_input)


# Compare outputs
difference = np.abs(pytorch_output - keras_output)
print('PyTorch output:', pytorch_output)
print('Keras output:', keras_output)
print('Max difference:', difference.max())
keras_model.save('converted_keras_model_fixed.h5')


# %% [markdown]
# # Compare difference

# %%

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Softmax
from tensorflow.keras.models import Model
import torch
import torch.nn as nn

# --- Create and Load Keras Model ---
def create_keras_model():
    input_layer = Input(shape=(48, 48, 1), name='input_layer')
    x = Conv2D(16, (3, 3), padding='valid')(input_layer)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)
    x = Conv2D(16, (3, 3), padding='valid')(x)
    x = ReLU()(x)  # Add ReLU here
    x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)
    # x = Permute((3, 1, 2))(x)  # Add Permute layer here
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(2)(x)
    output_layer = Softmax()(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


keras_model = create_keras_model()
# keras_model.load_weights('converted_keras_model_fixed.h5')
keras_model.load_weights('converted_keras_model_fixed.h5')

# --- Prepare Input Data for Both Models ---
input_shape = (1, 48, 48, 1)  # Adjust this as necessary
dummy_input_keras = np.random.rand(*input_shape).astype('float32')

# Run the Keras model once to ensure it's fully initialized
keras_predictions = keras_model.predict(dummy_input_keras)

# Print model summary
keras_model.summary()

# %%
# --- Capture Activations from Keras Model ---
flatten_layer_name = 'flatten'  # Confirm this name matches your model summary
activation_model_keras = Model(inputs=keras_model.input, outputs=keras_model.get_layer(flatten_layer_name).output)
flatten_output_keras = activation_model_keras.predict(dummy_input_keras)
print("Keras Flatten output shape:", flatten_output_keras.shape)

# %%
import torch
import torch.nn as nn
from torchsummary import summary

# --- Load and Prepare PyTorch Model ---
class QuantizedCNN(nn.Module):
    def __init__(self, num_classes=2, input_size=(1, 48, 48)):
        super(QuantizedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            dummy_output = self.features(dummy_input)
            num_ftrs = dummy_output.numel() // dummy_output.size(0)
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

device = 'cpu'
pytorch_model = QuantizedCNN(num_classes=2, input_size=(1, 48, 48)).to(device)
pytorch_model.load_state_dict(torch.load('checkpoint/Final_models/KD_13 October 00:13_resnet50_qt-1channel-imsize-120-48.pt', map_location=device))
pytorch_model.eval()

# Use torchsummary to print the model summary, explicitly setting the device
# summary(pytorch_model, (1, 48, 48), device=device)


# %%

# Convert Keras input to match PyTorch's input format (N, C, H, W)
dummy_input_pytorch = torch.from_numpy(dummy_input_keras.transpose(0, 3, 1, 2)).float()

# --- Capture Activations from PyTorch Model ---
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hook
pytorch_model.classifier[0].register_forward_hook(get_activation('flatten_output'))

# Run the PyTorch model
with torch.no_grad():
    pytorch_predictions = pytorch_model(dummy_input_pytorch)

flatten_output_pytorch = activations['flatten_output'].numpy()
print("PyTorch Flatten output shape:", flatten_output_pytorch.shape)

# Reshape PyTorch output to match Keras convention (N, H, W, C)
# flatten_output_pytorch = flatten_output_pytorch.transpose(0, 2, 3, 1)
print("PyTorch Flatten output shape:", flatten_output_pytorch.shape)

# --- Compare the Activations ---
difference = np.abs(flatten_output_pytorch - flatten_output_keras)
print("Max difference:", np.max(difference))
print("Mean difference:", np.mean(difference))


# %% [markdown]
# # Do inferencing

# %% [markdown]
# 

# %%

# %%
import torch
from torchvision import models
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from LymphoMNIST.LymphoMNIST import LymphoMNIST
from torchvision import transforms
from torchsummary import summary

# %% [markdown]
# ### Define Dataset and Helper Functions

# %%

BATCH_SIZE = 64

class FilteredLymphoMNIST(Dataset):
    def __init__(self, original_dataset, labels_to_keep):
        self.original_dataset = original_dataset
        self.indices = [i for i, (_, label) in enumerate(original_dataset) if label in labels_to_keep]

    def __getitem__(self, index):
        return self.original_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

def get_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE, **kwargs):
    val_size = len(val_ds) // 2
    test_size = len(val_ds) - val_size
    val_ds, test_ds = random_split(val_ds, [val_size, test_size])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs),
    )
    

# Function to calculate accuracy
def calculate_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
    

# Hyperparameters
params = {
    'batch_size': 16,
    'im_size': 48,  # Resize dimension used during training
    'model_checkpoint': "../checkpoint/Final_models/KD_13 October 00:13_resnet50_qt-1channel-imsize-120-48.pt"  # Path to the saved model
}

BIGGER = 48

import torchvision.transforms as T
transform_student = T.Compose([
    T.Resize((BIGGER, BIGGER)),
    T.ToTensor(),
    T.Normalize([0.4819], [0.1484]),
])

# %% [markdown]
# ### Initialize the Dataset

# %%
# Initialize datasets
labels_to_keep = [0, 1]
original_train_ds = LymphoMNIST(root='../dataset', train=True, download=True, transform=transform_student, num_classes=3)
original_test_ds = LymphoMNIST(root='../dataset', train=False, download=True, transform=transform_student, num_classes=3)

# Filter datasets
train_ds = FilteredLymphoMNIST(original_train_ds, labels_to_keep)
test_ds = FilteredLymphoMNIST(original_test_ds, labels_to_keep)



# Get dataloaders
train_dl, val_dl, test_dl = get_dataloaders(train_ds, test_ds, batch_size=params['batch_size'], num_workers=4)

# %%

# Function to calculate accuracy using Keras model
def calculate_accuracy_keras(loader, model):
    correct = 0
    total = 0
    for images, labels in loader:
        # images: Tensor of shape (batch_size, channels, height, width)
        # labels: Tensor of shape (batch_size)
        
        # Convert images to numpy arrays and reshape to (batch_size, height, width, channels)
        images_np = images.numpy()
        images_np = images_np.transpose(0, 2, 3, 1)  # Convert from (N, C, H, W) to (N, H, W, C)
        
        # Perform inference with Keras model
        predictions = model.predict(images_np)
        
        # Get predicted labels
        predicted_labels = np.argmax(predictions, axis=1)
        labels_np = labels.numpy()
        
        total += labels_np.shape[0]
        correct += (predicted_labels == labels_np).sum()
        
    accuracy = 100 * correct / total
    return accuracy

# Calculate and print accuracy on the test dataset
test_accuracy = calculate_accuracy_keras(test_dl, keras_model)
print(f'Test Accuracy with Keras model: {test_accuracy:.2f}%')


