import copy
import nibabel as nib
import numpy as np
import os
import tarfile
import json
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import random_split

#TODO: Normalization over the whole mean and std of the dataset (or for lazy programmers, mean and std of each ct)
#TODO: Cropping over the difference between maximum and minimum slice with label different from zero, run it over 
    #the whole dataset, get the maximum difference and crop this difference in all the volumes from the center label slice
#TODO: Remove outliers from the dataset that are bigger than 3 std from the mean

class MedicalSegmentationDecathlon(Dataset):
    """
    The base dataset class for Decathlon segmentation tasks
    -- __init__()
    :param task_number -> represent the organ dataset ID (see task_names above for hints)
    :param dir_path -> the dataset directory path to .tar files
    :param transform -> optional - transforms to be applied on each instance
    """
    def __init__(self,  split_ratios = [0.8, 0.1, 0.1], transforms = None, mode = None) -> None:
        super(MedicalSegmentationDecathlon, self).__init__()
        self.dir = "../Task08_HepaticVessel"
        self.meta = json.load(open(os.path.join(self.dir, "dataset.json")))
        self.splits = split_ratios
        self.transform = transforms
        #Calculating split number of images
        num_training_imgs =  self.meta["numTraining"]
        train_val_test = [int(x * num_training_imgs) for x in split_ratios]
        if(sum(train_val_test) != num_training_imgs): train_val_test[0] += (num_training_imgs - sum(train_val_test))
        train_val_test = [x for x in train_val_test if x!=0]
        # train_val_test = [(x-1) for x in train_val_test]
        self.mode = mode
        #Spliting dataset
        samples = self.meta["training"]
        shuffle(samples)
        self.train = samples[0:train_val_test[0]]
        self.val = samples[train_val_test[0]:train_val_test[0] + train_val_test[1]]
        self.test = samples[train_val_test[1]:train_val_test[1] + train_val_test[2]]


    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.train)
        elif self.mode == "val":
            return len(self.val)
        elif self.mode == "test":
            return len(self.test)
        return self.meta["numTraining"]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #Obtaining image name by given index and the mode using meta data
        if self.mode == "train":
            name = self.train[idx]['image'].split('/')[-1]
        elif self.mode == "val":
            name = self.val[idx]['image'].split('/')[-1]
        elif self.mode == "test":
            name = self.test[idx]['image'].split('/')[-1]
        else:
            name = self.meta["training"][idx]['image'].split('/')[-1]
        img_path = os.path.join(self.dir, "imagesTr", name)
        label_path = os.path.join(self.dir, "labelsTr", name)

        img_object = nib.load(img_path)
        label_object = nib.load(label_path)
        
        img_array = img_object.get_fdata()
        label_array = label_object.get_fdata()

        #Cropping the center of the label
        img_array, label_array = crop_from_center(img_array, label_array, 28)

        #Converting to channel-first numpy array
        img_array = np.moveaxis(img_array, -1, 0)
        label_array = np.moveaxis(label_array, -1, 0)

        proccessed_out = {'name': name, 'image': img_array, 'label': label_array} 

        
        #The output numpy array is in channel-first format
        return proccessed_out


def crop_from_center(img_array, label_array, crop_size):
    """
    The utility function to crop the center of the label
    """

    # Find the center of the non-zero label
    label_positions = np.argwhere(label_array)
    if len(label_positions) > 0:
        #check z position average
        z_center = label_positions[:, 2].mean().astype(int)
    else:
        z_center = label_array.shape[2] // 2

    # Define the crop range, ensuring it doesn't go out of bounds
    z_start = max(z_center - int(crop_size/2), 0)
    z_end = min(z_center + int(crop_size/2), label_array.shape[0])

    # Crop the image and label arrays
    img_array = img_array[:, :, z_start:z_end]
    label_array = label_array[:, :, z_start:z_end]

    print(f"Image shape: {img_array.shape}, Label shape: {label_array.shape}")
    #Assure that the image has 28 slices
    if img_array.shape[2] != crop_size:
        print(f"Error image slices is not {crop_size}")

    return img_array, label_array


def get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms):
    """
    The utility function to generate splitted train, validation and test dataloaders
    
    Note: all the configs to generate dataloaders in included in "config.py"
    """

    dataset = MedicalSegmentationDecathlon(split_ratios=[0.8, 0.1, 0.1], transforms=[train_transforms, val_transforms, test_transforms])

    #Spliting dataset and building their respective DataLoaders
    train_set, val_set, test_set = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_set.set_mode('train')
    val_set.set_mode('val')
    test_set.set_mode('test')
    train_dataloader = DataLoader(dataset= train_set, batch_size= 1, shuffle= False)
    val_dataloader = DataLoader(dataset= val_set, batch_size= 1, shuffle= False)
    test_dataloader = DataLoader(dataset= test_set, batch_size= 1, shuffle= False)
    
    return train_dataloader, val_dataloader, test_dataloader


def test_dataloaders():
    # Define your transformations here, for simplicity, let's assume they are None
    train_transforms = None
    val_transforms = None
    test_transforms = None

    # Get the dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms)
    print(f"Train Dataloader length: {train_dataloader}")
    # Test the train dataloader
    print("Testing Train Dataloader")
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i}: Image shape: {batch['image'].shape}, Label shape: {batch['label'].shape}")
        if i == 2:  # Just check a few batches to keep the output short
            break

    # Test the validation dataloader
    print("\nTesting Validation Dataloader")
    for i, batch in enumerate(val_dataloader):
        print(f"Batch {i}: Image shape: {batch['image'].shape}, Label shape: {batch['label'].shape}")
        if i == 2:
            break

    # Test the test dataloader
    print("\nTesting Test Dataloader")
    for i, batch in enumerate(test_dataloader):
        print(f"Batch {i}: Image shape: {batch['image'].shape}, Label shape: {batch['label'].shape}")
        if i == 2:
            break

if __name__ == "__main__":
    test_dataloaders()