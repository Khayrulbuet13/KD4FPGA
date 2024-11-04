import datetime
from comet_ml import Experiment
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from tqdm import tqdm
import torchvision.transforms as T
from LymphoMNIST.LymphoMNIST import LymphoMNIST
import os

# Constants
EPOCHS = 100000
TEMPERATURE = 1
INIT_LR = 0.001
WEIGHT_DECAY = .0001
CLIP_THRESHOLD = 1.0
ALPHA = 1
BATCH_SIZE = 64
IM_SIZE = 120   # Image size for the teacher model
STUDENT_IM_SIZE = 48  # Image size for the student model after resizing
MODEL = 'resnet50_qt-1channel-imsize-120-120'

# Set device (CUDA or CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvertToRGB:
    """Convert 1-channel tensors to 3-channel tensors by duplicating the channel 3 times."""
    def __call__(self, tensor):
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor


from imgaug import augmenters as iaa
from PIL import Image
class ImgAugTransform:
    """
    Wrapper to allow imgaug to work with Pytorch transformation pipeline, modified for 1-channel images.
    """
    def __init__(self):
        self.aug = iaa.Sequential([
            # iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.8, iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])),
            iaa.Sometimes(0.5, iaa.Sequential([
                iaa.Crop(percent=(0.1, 0.2))
            ])),
            # iaa.LinearContrast((0.75, 1.5)),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.8, iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )),
            # iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
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
    
class FilteredLymphoMNIST(Dataset):
    """Filters the dataset to keep only specified labels."""
    def __init__(self, original_dataset, labels_to_keep):
        self.original_dataset = original_dataset
        self.indices = [i for i, (_, label) in enumerate(original_dataset) if label in labels_to_keep]

    def __getitem__(self, index):
        img, label = self.original_dataset[self.indices[index]]
        return img, label

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


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Applies Mixup augmentation to the inputs and targets."""
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
    """Calculates the Mixup loss."""
    y_a_one_hot = F.one_hot(y_a, num_classes=num_classes).float()
    y_b_one_hot = F.one_hot(y_b, num_classes=num_classes).float()
    return lam * criterion(pred, y_a_one_hot) + (1 - lam) * criterion(pred, y_b_one_hot)


class QuantizedCNN(nn.Module):
    """Defines the student model architecture."""
    def __init__(self, num_classes=2, input_size=(1, 64, 64)):
        super(QuantizedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # Calculate the number of features after convolution layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            dummy_output = self.features(dummy_input)
            num_ftrs = dummy_output.numel()
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


def evaluate(model, data_loader):
    """Evaluates the model on the validation or test dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            images_student = images[:, 0:1, :, :]  # Extract first channel
            images_student = F.interpolate(images_student, size=(STUDENT_IM_SIZE, STUDENT_IM_SIZE), mode='bilinear', align_corners=False)
            outputs = model(images_student)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def main():
    experiment = Experiment(api_key="2iwTpjYhUb3dGr4yIiVtt1oRA", project_name="KD4FPGA", workspace="khayrulbuet13")
    experiment.set_name(MODEL)

    # Unified transform for both teacher and student
    train_transform = T.Compose([
        T.Resize((IM_SIZE, IM_SIZE)),
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        T.ToTensor(),
        ConvertToRGB(),
        T.Normalize([0.4819, 0.4819, 0.4819], [0.1484, 0.1484, 0.1484]),
    ])
    
    val_transform = T.Compose([
        T.Resize((IM_SIZE, IM_SIZE)),
        T.ToTensor(),
        ConvertToRGB(),
        T.Normalize([0.4819, 0.4819, 0.4819], [0.1484, 0.1484, 0.1484]),
    ])

    # Load datasets with unified transform
    labels_to_keep = [0, 1]
    train_ds = FilteredLymphoMNIST(
        LymphoMNIST(root='./dataset', train=True, download=True, transform=train_transform, num_classes=3), labels_to_keep)
    val_test_ds = FilteredLymphoMNIST(
        LymphoMNIST(root='./dataset', train=False, download=True, transform=val_transform, num_classes=3), labels_to_keep)

    # Initialize dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_test_ds, num_workers=4)

    # Load teacher model
    resnet50 = models.resnet50()
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, 2)
    teacher_model = resnet50
    teacher_model.load_state_dict(torch.load('checkpoint/Final_models/Teacher_imsize-120_30 September 22:37.pt', map_location=device))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Initialize student model
    num_classes = len(labels_to_keep)
    student_model = QuantizedCNN(num_classes=num_classes, input_size=(1, STUDENT_IM_SIZE, STUDENT_IM_SIZE)).to(device)

    summary(student_model, (1, STUDENT_IM_SIZE, STUDENT_IM_SIZE))

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(student_model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    checkpoint_path = str("./checkpoint/KD_" + f"{datetime.datetime.now().strftime('%d %B %H:%M')}_{MODEL}.pt")

    best_val_accuracy = 0.0
    early_stopping_patience = 200
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        student_model.train()
        teacher_model.eval()

        for images, labels in tqdm(train_loader, total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Teacher model inference
            teacher_outputs = teacher_model(images)

            # Student model inference with single channel and resizing
            images_student = images[:, 0:1, :, :]  # Extract first channel
            images_student = F.interpolate(images_student, size=(STUDENT_IM_SIZE, STUDENT_IM_SIZE), mode='bilinear', align_corners=False)
            mixed_inputs_student, targets_a, targets_b, lam = mixup_data(images_student, labels, ALPHA, device)
            student_outputs = student_model(mixed_inputs_student)

            # Compute loss
            student_output_log_prob = F.log_softmax(student_outputs / TEMPERATURE, dim=1)
            teacher_output_soft = F.softmax(teacher_outputs / TEMPERATURE, dim=1).detach()
            loss = mixup_criterion(criterion, student_output_log_prob, targets_a, targets_b, lam, num_classes)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(student_model.parameters(), CLIP_THRESHOLD)
            optimizer.step()

            # Update statistics
            train_loss += loss.item() * images_student.size(0)
            _, predicted = torch.max(student_outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        val_accuracy = evaluate(student_model, val_loader)
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
                print("Early stopping due to no improvement in validation accuracy.")
                break

        print(f'Epoch {epoch+1}, Train Loss: {train_loss/train_total:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

    test_accuracy = evaluate(student_model, test_loader)
    experiment.log_metric("test_accuracy", test_accuracy)
    experiment.end()


if __name__ == "__main__":
    main()
