from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from PIL import Image
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def load_data(directory_path, transform=None):
    """
    Load data from the specified directory.

    Args:
        directory_path (str): Path to the directory containing the data.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.

    Returns:
        dataset (CustomImageFolder): Dataset object containing the images and their corresponding metadata.
    """
    return CustomImageFolder(root=directory_path, transform=transform)

def get_transformation(data_set, transforms_list=None):
    """
    Apply a list of transformations to each image in the dataset and append the transformed images to the dataset.

    Args:
        data_set (ImageFolder): The dataset containing images and labels.
        transforms_list (list, optional): A list of transformation functions to apply to the images. Each transformation function should take an image as input and return a transformed image.

    Returns:
        list: A new dataset with the original images and their transformed versions. Each element in the dataset is a tuple (image, label), where 'image' is the transformed image and 'label' is the corresponding label.
    """
    if transforms_list is None:
        return data_set
    
    augmented_data = []
    for image in data_set:
        
        # # Choose random transformations from the list 
        # include_indices = np.random.multinomial(1, [1/2.]*2, size=len(transforms_list))
        # filtered_transforms_list = [transforms_list[i] for i in range(len(transforms_list)) if (include_indices[i] == 1)]
        
        # Apply the randomly-selected transformations to the image
        transformed_image = image[0]
        label = image[1]
        for transform in transforms_list:
            r = np.random.rand()
            if r > 0.5:
                transformed_image = transform(transformed_image)
        augmented_data.append((transformed_image, label))

    return augmented_data


def split_dataset(dataset, train_ratio=0.8, random_state=None):
    """
    Split the dataset into training and testing sets.

    Args:
        dataset (ImageFolder): The dataset to split. Each element in the dataset is a tuple (image, label).
        train_ratio (float): The ratio of the dataset to include in the training set.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple: Two datasets containing the training and testing data. Each dataset is a Subset of the original dataset.
    """
    # Get the indices of the dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    #targets = [dataset[i][1] for i in indices]
    
    # Split the indices into training and testing sets
    train_indices, test_indices = train_test_split(indices, train_size=train_ratio, random_state=random_state, shuffle=False)
    
    # Create subsets for training and testing
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    
    
    return train_dataset, test_dataset
        
class CustomImageFolder(ImageFolder):
    """
    Custom ImageFolder class to load images from a directory.

    This class extends the torchvision.datasets.ImageFolder class to include additional functionality
    for loading images and their corresponding metadata from JSON files.

    Args:
        root (str): Root directory path.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        loader (callable, optional): A function to load an image given its path. Defaults to the default_loader.

    Methods:
        __getitem__(index): Returns the image and its label at the specified index.
    """
    
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        super(CustomImageFolder, self).__init__(root, transform=transform, target_transform=target_transform, loader=loader)
        self.root = root

    def __getitem__(self, index):
        """
        Get the image and its label at the specified index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: (sample, target) where sample is the transformed image and target is the label.
        """
        # Get the image and its label
        path, target = self.samples[index]
        sample = self.loader(path)

        # Get the corresponding JSON file
        json_path = path.replace('.jpg', '.json').replace('.png', '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
