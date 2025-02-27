import torchvision.transforms as transforms
from PIL import Image

def apply_transformation(image_path):
    """
    Apply transformations to an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        transformed_image (PIL.Image): Transformed image.
    """ 

    # Define the augmentation transforms without resizing
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(5)
    ])

    # Open the image and convert to RGB
    image = Image.open(image_path).convert('RGB')

    # Apply the transformations to the image
    transformed_image = augmentation_transforms(image)

    return transformed_image