
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2
from skimage.measure import regionprops, label as sk_label
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Load LymphoMNIST dataset
from LymphoMNIST.LymphoMNIST import LymphoMNIST

# Import SAM modules
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Set device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Set debug flag
debug = True  # Set to False to disable plotting

# Function to plot images if debug is True
def plot_image(image, title='', cmap='gray'):
    if debug:
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()

# Initialize dataset without transformations
dataset = LymphoMNIST(root='./dataset', train=True, download=True, transform=None, num_classes=3)

debug = False  # Disable plotting for this cell

# Initialize lists to store properties
rotations = []
scales = []
translations_x = []
translations_y = []
shears = []
brightnesses = []
contrasts = []

# Function to compute brightness and contrast
def compute_brightness_contrast(image_gray, mask):
    masked_pixels = image_gray[mask]
    brightness = np.mean(masked_pixels)
    contrast = np.std(masked_pixels)
    return brightness, contrast

# Initialize SAM model
sam_checkpoint = "checkpoint/sam_vit_h_4b8939.pth"  # Use the correct checkpoint for 'vit_h'
model_type = "vit_h"  # Ensure this matches the checkpoint

# Load the model and initialize SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Iterate over the dataset
for idx in tqdm(range(len(dataset)), desc="Processing images"):
    image_pil, class_label = dataset[idx]

    # Convert PIL image to NumPy array (grayscale)
    image_np = np.array(image_pil)

    # Convert grayscale image to RGB (SAM expects 3-channel images)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Generate masks using SAM
    masks = mask_generator.generate(image_rgb)

    # If no masks are found, skip this image
    if not masks:
        continue

    # Find the largest mask based on area
    largest_mask_info = max(masks, key=lambda mask_info: np.sum(mask_info['segmentation']))
    largest_mask = largest_mask_info['segmentation'].astype(bool)

    # Label connected components within the mask
    labeled_img = sk_label(largest_mask)

    # Get region properties
    props = regionprops(labeled_img)

    for prop in props:
        # Skip small regions to avoid noise
        if prop.area < 50:
            continue

        # Bounding box dimensions
        minr, minc, maxr, maxc = prop.bbox
        width = maxc - minc
        height = maxr - minr

        # Calculate the eccentricity (how much the shape deviates from circular)
        # If eccentricity is too high (e.g., > 0.9), it deviates from circularity
        if prop.eccentricity > 0.7:
            # Plot even if skipping due to non-circularity
            plot_image(largest_mask, title=f"Non-circular Mask for Image {idx} (Skipped)") if debug else None
            continue

        # Scale (size of the cell relative to image size)
        scale_x = width / image_np.shape[1]
        scale_y = height / image_np.shape[0]
        scales.append((scale_x, scale_y))

        # Translation (position of the cell's centroid relative to image center)
        centroid_x = prop.centroid[1] / image_np.shape[1] - 0.5  # Normalize between -0.5 and 0.5
        centroid_y = prop.centroid[0] / image_np.shape[0] - 0.5
        translations_x.append(centroid_x)
        translations_y.append(centroid_y)

        # Orientation (rotation angle in degrees)
        rotation_angle = -prop.orientation * (180 / np.pi)  # Convert to degrees and invert sign
        rotations.append(rotation_angle)

        # Shear (approximated from eccentricity)
        shear_angle = np.arccos(prop.eccentricity) * (180 / np.pi)
        shears.append(shear_angle)

        # Brightness and Contrast within the cell
        brightness, contrast = compute_brightness_contrast(image_np, largest_mask)
        brightnesses.append(brightness)
        contrasts.append(contrast)

        # Visualization of the detected cell
        image_with_contour = np.copy(image_rgb)
        contours, _ = cv2.findContours(largest_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if debug:
            cv2.drawContours(image_with_contour, contours, -1, (255, 0, 0), 2)
            plot_image(image_with_contour, title=f"Image {idx}, Rotation: {rotation_angle:.2f}°, Scale: ({scale_x:.2f}, {scale_y:.2f})", cmap=None)
        
    # if idx >= 10:
    #     break

# Convert lists to numpy arrays
rotations = np.array(rotations)
scales = np.array(scales)
translations_x = np.array(translations_x)
translations_y = np.array(translations_y)
shears = np.array(shears)
brightnesses = np.array(brightnesses)
contrasts = np.array(contrasts)

# Compute mean and standard deviation
results = {
    "Rotation": (np.mean(rotations), np.std(rotations)),
    "Scale_X": (np.mean(scales[:, 0]), np.std(scales[:, 0])),
    "Scale_Y": (np.mean(scales[:, 1]), np.std(scales[:, 1])),
    "Translation_X": (np.mean(translations_x), np.std(translations_x)),
    "Translation_Y": (np.mean(translations_y), np.std(translations_y)),
    "Shear": (np.mean(shears), np.std(shears)),
    "Brightness": (np.mean(brightnesses), np.std(brightnesses)),
    "Contrast": (np.mean(contrasts), np.std(contrasts)),
}

# Print the results
print("\nComputed Properties:")
for prop, (mean_val, std_val) in results.items():
    print(f"{prop} - Mean: {mean_val:.2f}, Std: {std_val:.2f}")
