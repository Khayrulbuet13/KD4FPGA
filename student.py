import datetime
# from arrow import get
from comet_ml import Experiment
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from glob import glob
from PIL import Image
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler
import torch
from tqdm import tqdm
import torchvision.transforms as T

from LymphoMNIST.LymphoMNIST import LymphoMNIST
from torch.utils.data import DataLoader, Dataset, random_split
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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



# CONSTANTS
EPOCHS = 100000
TEMPERATURE = 1
INIT_LR = 0.001
WEIGHT_DECAY = .0001
CLIP_THRESHOLD = 1.0
ALPHA = 1
BATCH_SIZE = 28
RESIZE = 28
BIGGER = 28
MODEL = 'MNISTresnet18==>smaller_qt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
experiment = Experiment(
        api_key="2iwTpjYhUb3dGr4yIiVtt1oRA",
        project_name="KD4FPGA",
        # project_name="",
        workspace="khayrulbuet13")

experiment.set_name(MODEL)




transform_train = T.Compose([T.Resize((BIGGER, BIGGER)),
                                      T.ToTensor(),
                                      T.Normalize([0.4819], [0.1484]),
                                    #   ConvertToRGB()
                                      ])


transform_val = T.Compose([T.Resize((BIGGER, BIGGER)),
                                      T.ToTensor(),
                                      T.Normalize([0.4819], [0.1484]),
                                    #   ConvertToRGB()
                                      ])


class FilteredLymphoMNIST(Dataset):
    def __init__(self, original_dataset, labels_to_keep):
        self.original_dataset = original_dataset
        self.indices = [i for i, (_, label) in enumerate(original_dataset) if label in labels_to_keep]

    def __getitem__(self, index):
        return self.original_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

def get_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE, **kwargs):
    # Split val_ds to validation and test sets evenly
    val_size = len(val_ds) // 2
    test_size = len(val_ds) - val_size
    val_ds, test_ds = random_split(val_ds, [val_size, test_size])
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs),
    )




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
    



# Load MNIST dataset
from torchvision.datasets import MNIST
train_ds = MNIST(root='./data', train=True, download=True, transform=transform_train)
val_test_ds = MNIST(root='./data', train=False, download=True, transform=transform_train)


# Initialize dataloaders
train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_test_ds, num_workers=4)


def mixup(image):
    alpha = torch.rand(1)
    mixedup_images = (alpha * image +
                      (1 - alpha) * torch.flip(image, dims=[0]))
    return mixedup_images


def evaluate(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def mixup_data(x, y, alpha=1.0, device=device):
    '''Compute the mixup data. Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, num_classes):
    # print(f"y_a: {y_a}")
    # print(f"y_b: {y_b}")
    # print(f"Unique values in y_a: {torch.unique(y_a)}")
    # print(f"Unique values in y_b: {torch.unique(y_b)}")

    y_a_one_hot = F.one_hot(y_a, num_classes=num_classes).float()
    y_b_one_hot = F.one_hot(y_b, num_classes=num_classes).float()
    return lam * criterion(pred, y_a_one_hot) + (1 - lam) * criterion(pred, y_b_one_hot)



from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("fxmarty/resnet-tiny-mnist")
model = AutoModelForImageClassification.from_pretrained("fxmarty/resnet-tiny-mnist")

teacher_model = model.to(device)
teacher_model.eval()


num_classes =10
data_iter = iter(train_loader)
samples, targets = next(data_iter)
input_size = samples.shape[1:]
student_model =  QuantizedCNN(num_classes=num_classes, input_size=input_size).to(device)
summary(student_model, input_size=input_size)


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def forward(self, x):
        teacher_output = self.teacher(x)
        student_output = self.student(x)

        return student_output, teacher_output


# Create an instance of the Distiller class
distiller = Distiller(student=student_model, teacher=teacher_model)


# Define loss function and optimizer
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.AdamW(student_model.parameters(),
                        lr=INIT_LR, weight_decay=WEIGHT_DECAY)
# optimizer = optim.SGD(student_model.parameters(), lr=INIT_LR,
#                       momentum=0.9, weight_decay=WEIGHT_DECAY)


checkpoint_path = str("/checkpoint" +
                      f"{datetime.datetime.now().strftime('%d %B %H:%M')}-student_model.pt")



best_val_accuracy = 0.0
early_stopping_patience = 100
epochs_no_improve = 0

for epoch in range(EPOCHS):
    train_loss = 0.0
    train_total = 0
    train_correct = 0
    student_model.train()
    train_loader_progress = tqdm(train_loader, total=len(train_loader),
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    for inputs, labels in train_loader_progress:
        inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
        optimizer.zero_grad()
        
        mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, ALPHA, device)
        student_output, teacher_output = distiller(mixed_inputs)
        # teacher_output = teacher_output.detach()  # Ensure no gradient is computed for teacher
        teacher_output = teacher_output.logits.detach()
        student_output_log_prob = F.log_softmax(student_output / TEMPERATURE, dim=1)
        loss = mixup_criterion(criterion, student_output_log_prob, targets_a, targets_b, lam, num_classes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), CLIP_THRESHOLD)
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(student_output.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Update progress bar
        train_loader_progress.set_description(
            f"Epoch {epoch+1}/{EPOCHS}")
        train_loader_progress.set_postfix(
            loss=loss.item(), acc=100. * train_correct / train_total)
        
    train_accuracy = 100 * train_correct / train_total
    val_accuracy = evaluate(student_model, val_loader)
    experiment.log_metric("train_loss", train_loss / train_total, step=epoch)
    experiment.log_metric("train_accuracy", train_accuracy, step=epoch)
    experiment.log_metric("val_accuracy", val_accuracy, step=epoch)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
        print(f'Validation accuracy increased ({best_val_accuracy:.2f}%)')
        torch.save(student_model.state_dict(), f"./checkpoint/KD_{MODEL}.pt")
        
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs. No improvement in validation accuracy.')
            break

    print(f'Epoch {epoch+1}, Train Loss: {train_loss/train_total:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

experiment.end()

