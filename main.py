import utils as utils
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def train():
    pass

def main():

    # Loading the dataset, resizing
    dataset = utils.load_data("./small_dataset", transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))

    # applying transformations and adding them to dataset
    dataset_aug = utils.get_transformation(dataset, transforms_list=[
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=(0.5)),  # Adjust brightness (0.5 for darkness, 1.5 for brightness)
    ])


    train_dataset, test_dataset = utils.split_dataset(dataset_aug, train_ratio=0.8)


if __name__ == "__main__":
    main()



