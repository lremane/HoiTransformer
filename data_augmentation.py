from PIL import Image
import torch

from datasets.hico import make_hico_transforms

# Load augmentation transforms
transforms = make_hico_transforms(image_set='train')

# Load your image
img = Image.open("path_to_your_image.jpg").convert('RGB')  # Replace with actual image path

# Create a sample annotation
target = {
    "human_boxes": torch.tensor([[50, 50, 100, 100]]),  # Example box coordinates
    "object_boxes": torch.tensor([[150, 150, 200, 200]]),
    "action_boxes": torch.tensor([[40, 40, 210, 210]]),
    "human_labels": torch.tensor([1]),  # Example labels
    "object_labels": torch.tensor([2]),
    "action_labels": torch.tensor([3]),
}

# Apply data augmentation
augmented_img, augmented_target = transforms(img, target, image_set='train')

# Display results
augmented_img.show()  # Show augmented image
print(augmented_target)  # Augmented annotations
