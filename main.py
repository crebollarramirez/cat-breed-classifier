import utils as utils
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.utils.data import DataLoader
import multiprocessing
import sys
from sklearn.utils.class_weight import compute_class_weight
from model import FCN
import gc
from train import *


def train():
    best_iou_score = 0.0
    best_model_path = "best_model.pth"
    epochs_no_improve = 0
    early_stop_epoch = None

    train_losses = []

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        fcn_model.train()


        epoch_train_loss = 0.0

        for iter, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model(inputs)  # Forward pass

            loss = criterion(outputs, labels)

            loss.backward()  # Backward pass
    




epochs = 10
train_ratio = 0.8
batch_size = 16

num_workers = multiprocessing.cpu_count()
 
# # getting parameters from command line arguments
# epochs = int(sys.argv[1])
# train_ratio = float(sys.argv[2])
# batch_size = int(sys.argv[3])
# print(f"EPOCHS: {epochs}")        
# print(f"TRAIN/TEST RATIO: {train_ratio}")
# print(f"BATCH SIZE: {batch_size}")


# Loading the dataset, resizing
dataset = utils.load_data("./small_dataset", transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
]))

# applying transformations and adding them to dataset
dataset_aug = utils.get_transformation(dataset, transforms_list=[
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=(0.5)),  # Adjust brightness (0.5 for darkness, 1.5 for brightness)
])

train_dataset, test_dataset = utils.split_dataset(dataset, train_ratio=train_ratio)

# getting the targets
targets = []
for inx in range(len(train_dataset)):
    targets.append(int(train_dataset[inx][1]))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using Device: ", device)

class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights_tensor = torch.tensor(class_weights).to(device)


criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

fcn_model = FCN(num_classes=len(np.unique(targets)))

fcn_model = fcn_model.to(device)

if __name__ == "__main__":
    train()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
    

