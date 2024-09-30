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

# Constants
EPOCHS = 100000
TEMPERATURE = 1
INIT_LR = 0.001
WEIGHT_DECAY = .0001
CLIP_THRESHOLD = 1.0
ALPHA = 1
BATCH_SIZE = 64
RESIZE = 64
BIGGER = 64
MODEL = 'resnet50_qt-1channel'

# Set device (CUDA or CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvertToRGB:
    """ Convert 1-channel tensors to 3-channel tensors by duplicating the channel 3 times. """
    def __call__(self, tensor):
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

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

def mixup(image):
    alpha = torch.rand(1)
    mixedup_images = (alpha * image + (1 - alpha) * torch.flip(image, dims=[0]))
    return mixedup_images

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            # Ensure data is on the correct device
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def mixup_data(x, y, alpha=1.0, device='cuda'):
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
    y_a_one_hot = F.one_hot(y_a, num_classes=num_classes).float()
    y_b_one_hot = F.one_hot(y_b, num_classes=num_classes).float()
    return lam * criterion(pred, y_a_one_hot) + (1 - lam) * criterion(pred, y_b_one_hot)

class QuantizedCNN(nn.Module):
    def __init__(self, num_classes=2, input_size=(1, 28, 28)):
        super(QuantizedCNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
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

class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
    def forward(self, x):
        teacher_output = self.teacher(x)
        student_output = self.student(x)
        return student_output, teacher_output


# Transform for teacher model
transform_teacher = T.Compose([
    T.Resize((BIGGER, BIGGER)),
    T.ToTensor(),
    T.Normalize([0.4819], [0.1484]),
    ConvertToRGB()
])

# Transform for student model
transform_student = T.Compose([
    T.Resize((BIGGER, BIGGER)),
    T.ToTensor(),
    T.Normalize([0.4819], [0.1484]),
])

def main():
    experiment = Experiment(api_key="2iwTpjYhUb3dGr4yIiVtt1oRA", project_name="KD4FPGA", workspace="khayrulbuet13")
    experiment.set_name(MODEL)


    # Load datasets
    labels_to_keep = [0, 1]
    train_ds_teacher = FilteredLymphoMNIST(
        LymphoMNIST(root='./dataset', train=True, download=True, transform=transform_teacher, num_classes=3), labels_to_keep)
    val_test_ds_teacher = FilteredLymphoMNIST(
        LymphoMNIST(root='./dataset', train=False, download=True, transform=transform_teacher, num_classes=3), labels_to_keep)
    train_ds_student = FilteredLymphoMNIST(
        LymphoMNIST(root='./dataset', train=True, download=True, transform=transform_student, num_classes=3), labels_to_keep)
    val_test_ds_student = FilteredLymphoMNIST(
        LymphoMNIST(root='./dataset', train=False, download=True, transform=transform_student, num_classes=3), labels_to_keep)

    # Initialize dataloaders
    train_loader_teacher, val_loader_teacher, test_loader_teacher = get_dataloaders(train_ds_teacher, val_test_ds_teacher, num_workers=4)
    train_loader_student, val_loader_student, test_loader_student = get_dataloaders(train_ds_student, val_test_ds_student, num_workers=4)

    # Load models
    resnet50 = models.resnet50()
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, 2)
    teacher_model = resnet50
    teacher_model.load_state_dict(torch.load('checkpoint/Final_models/28 September 02:53_Teacher_final-control_final_weights.pt', map_location=device))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    num_classes = len(labels_to_keep)
    student_model = QuantizedCNN(num_classes=num_classes, input_size=(1, BIGGER, BIGGER)).to(device)
    summary(student_model, (1, BIGGER, BIGGER))

    distiller = Distiller(student=student_model, teacher=teacher_model)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(student_model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    checkpoint_path = str("./checkpoint/KD_" + f"{datetime.datetime.now().strftime('%d %B %H:%M')}_{MODEL}.pt")

    best_val_accuracy = 0.0
    early_stopping_patience = 100
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        student_model.train()
        teacher_model.eval()

        train_loader_progress = tqdm(zip(train_loader_student, train_loader_teacher),
                                 total=min(len(train_loader_student), len(train_loader_teacher)),
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        for (inputs_student, labels_student), (inputs_teacher, labels_teacher) in train_loader_progress:
            # Ensure inputs are on the correct device
            inputs_student, labels_student = inputs_student.to(device), labels_student.to(device)
            inputs_teacher, labels_teacher = inputs_teacher.to(device), labels_teacher.to(device)
            
            optimizer.zero_grad()

            mixed_inputs_student, targets_a, targets_b, lam = mixup_data(inputs_student, labels_student, ALPHA, device)
            student_output = student_model(mixed_inputs_student)
            teacher_output = teacher_model(inputs_teacher).detach()

            student_output_log_prob = F.log_softmax(student_output / TEMPERATURE, dim=1)
            loss = mixup_criterion(criterion, student_output_log_prob, targets_a, targets_b, lam, 2)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(student_model.parameters(), CLIP_THRESHOLD)
            optimizer.step()

            train_loss += loss.item() * inputs_student.size(0)
            _, predicted = torch.max(student_output.data, 1)
            train_total += labels_student.size(0)
            train_correct += (predicted == labels_student).sum().item()

        train_accuracy = 100 * train_correct / train_total
        val_accuracy = evaluate(student_model, val_loader_student)
        experiment.log_metric("train_loss", train_loss / train_total, step=epoch)
        experiment.log_metric("train_accuracy", train_accuracy, step=epoch)
        experiment.log_metric("val_accuracy", val_accuracy, step=epoch)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(student_model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                break
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/train_total:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')


    experiment.end()

if __name__ == "__main__":
    main()
