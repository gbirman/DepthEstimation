import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *
from torch.utils.data.sampler import SubsetRandomSampler

class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None,nrows=None):
        self.frame = pd.read_csv(csv_file, header=None, nrows=nrows)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame.iloc[idx, 0]
        depth_name = self.frame.iloc[idx, 1]
        masks_name = self.frame.iloc[idx, 2]
        
#         # obtain random mask 
#         masks = np.load(masks_name)
#         Nmasks = np.size(masks,2) # size along 3rd dimension 
#         np.random.seed(0) # change this during actual training
#         randmask = np.random.randint(0,Nmasks)
#         mask = masks[:,:,randmask]
#         mask = Image.fromarray(np.uint8(255*mask))
        
        # obtain random mask 
        masks = np.load(masks_name)
        mask = Image.fromarray(np.uint8(255*masks))
        
        image = Image.open(image_name)
        depth = Image.open(depth_name)
        
        sample = {'image': image, 'depth': depth, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(batch_size=64):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    
    # these are calculated in MATLAB, shouldn't make a huge difference? 
    __nyu_stats = {'mean': [0.481,0.411,0.392],
                    'std': [0.289,0.296,0.309]}
    
    transformed_training = depthDataset(csv_file='./data/nyu2_train4_1percent.csv',
                                        transform=transforms.Compose([ Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(is_test=True),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std']),
                                            Binarize(1),
                                            #ImageAlphaChannel()
                                        ])
                                       )

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=False, num_workers=4, pin_memory=False)

    return dataloader_training



def getTestingData(batch_size=64,nrows=None):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # these are calculated in MATLAB, shouldn't make a huge difference? 
    __nyu_stats = {'mean': [0.481,0.411,0.392],
                    'std': [0.289,0.296,0.309]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(csv_file='./data/nyu2_test4_1percent.csv',
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [152, 114]),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std']),
                                           Binarize(0.5),
#                                            ImageAlphaChannel()
                                       ]),nrows=nrows)

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=4, pin_memory=False)

    dataloader = {}
    dataloader['val'] = dataloader_testing 
    
    return dataloader

def getTrainValData(batch_size=64,nrows=None):
    
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    
    __nyu_stats = {'mean': [0.481,0.411,0.392],
                    'std': [0.289,0.296,0.309]}
    
    dataset = depthDataset(csv_file='./data/nyu2_train4_1percent.csv',
                                   transform=transforms.Compose([
                                       Scale(240),
                                       CenterCrop([304, 228], [152, 114]),
                                       ToTensor(is_test=True),
                                       Normalize(__imagenet_stats['mean'],
                                                 __imagenet_stats['std']),
                                       Binarize(0.5),
#                                        ImageAlphaChannel()
                                   ]),nrows=nrows)
    validation_split = .2
    shuffle_dataset = True
#     random_seed= 42 ###### change during testing

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
#         np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    
    dataloader = {}
    dataloader['train'] = train_loader
    dataloader['val'] = validation_loader 
    
    return dataloader

def getTestingSingle(batch_size=1):
    
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # these are calculated in MATLAB, shouldn't make a huge difference? 
    __nyu_stats = {'mean': [0.481,0.411,0.392],
                    'std': [0.289,0.296,0.309]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(csv_file='./data/nyu2_test4_1percent.csv',
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [152, 114]),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std']),
                                           Binarize(0.8),
                                           #ImageAlphaChannel()
                                       ]),nrows=1)

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=4, pin_memory=False)
    
    dataloader = {}
    dataloader['val'] = dataloader_testing 
    
    return dataloader
