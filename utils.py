import torch
import numpy as np
from torchvision import models
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
import glob
import random
import torchvision.transforms as t

def calc_metrics(preds, targets):
    '''
    Calculates accuracies, balanced accuracies, precisions, recalls and auroc for predictions and targets
    '''
    binary_preds = 1 * (preds >= 0.0)
    
    # Calculate metrics.
    acc = accuracy_score(targets, binary_preds)
    bal_acc = balanced_accuracy_score(targets, binary_preds)
    neg_prec = precision_score(targets, binary_preds, 
                                       pos_label=0, zero_division=0)
    pos_prec = precision_score(targets, binary_preds, 
                                       pos_label=1, zero_division=0)
    neg_recall = recall_score(targets, binary_preds, 
                                      pos_label=0, zero_division=0)
    pos_recall = recall_score(targets, binary_preds, 
                                      pos_label=1, zero_division=0)
    if (targets.sum() > 0) and not (targets.sum() == len(targets)):
        auroc = roc_auc_score(targets, preds)
    else:
        auroc = 0
    
    results = {
        'acc': acc,
        'acc_bal': bal_acc,
        'prec_empty': neg_prec,
        'prec_ok': pos_prec,
        'rec_empty': neg_recall,
        'rec_ok': pos_recall,
        'auroc': auroc
    }
    return results


def load_binary_images_from_folder(path):
    classes = os.listdir(path)
    classes = sorted(classes)
    images_empty = np.array(glob.glob(path + classes[0] + '/*.jpg'))
    images_empty = images_empty.reshape((len(images_empty),1))
    images_empty = np.concatenate((images_empty, np.zeros((len(images_empty),1))) , axis=1) 
    images_ok = np.array(glob.glob(path + classes[1] + '/*.jpg'))
    images_ok = images_ok.reshape((len(images_ok),1))
    images_ok = np.concatenate((images_ok, np.ones((len(images_ok),1))) , axis=1) 
    all_images = np.concatenate((images_empty, images_ok) , axis=0) 
    
    return all_images

# A class for own augmentation
class OwnRandAug:
    """Rotate by one of the given angles."""

    def __init__(self, mean, std, img_size, inter):
        self.trans = ['identity', 'rotate', 'color', 'sharpness', 'blur', 'affine' ,'erasing']
        self.means = mean
        self.stds = std
        self.img_size = img_size
        self.inter = inter

    def __call__(self, img, choice = None):
        if choice == None:
            choice = random.choice(self.trans)
        if choice == 'identity':
            trans = t.Compose([
                            t.Resize((self.img_size,self.img_size), interpolation= self.inter),
                            t.ToTensor(),
                            t.Normalize(self.means, self.stds),
                        ])
            img = trans(img)
        elif choice == 'rotate':
            degrees = random.uniform(0, 180)
            trans = t.Compose([
                            t.Resize((self.img_size,self.img_size), interpolation= self.inter),
                            t.ToTensor(),
                            t.RandomRotation(degrees),
                            t.Normalize(self.means, self.stds),
                        ])
            img = trans(img)
        elif choice == 'color':
            brigthness = np.random.exponential()/2
            contrast = np.random.exponential()/2
            saturation = np.random.exponential()/2
            hue = random.uniform(0, 0.5)
            trans = t.Compose([
                            t.Resize((self.img_size,self.img_size), interpolation= self.inter),
                            t.ToTensor(),
                            t.ColorJitter(brightness=brigthness, contrast=contrast, saturation=saturation, hue=hue),
                            t.Normalize(self.means, self.stds),
                        ])
            img = trans(img)
        elif choice=='sharpness':
            sharpness = 1+(np.random.exponential()/2)
            trans = t.Compose([
                            t.Resize((self.img_size,self.img_size), interpolation= self.inter),
                            t.ToTensor(),
                            t.RandomAdjustSharpness(sharpness, p=1),
                            t.Normalize(self.means, self.stds),
                        ])
            img = trans(img)
        elif choice=='blur':
            kernel = random.choice([1,3,5,7])
            trans = t.Compose([
                            t.Resize((self.img_size,self.img_size), interpolation= self.inter),
                            t.ToTensor(),
                            t.GaussianBlur(kernel, sigma=(0.1, 2.0)),
                            t.Normalize(self.means, self.stds),
                        ])
            img = trans(img)
        elif choice=='affine':
            trans = t.Compose([
                            t.Resize((self.img_size,self.img_size), interpolation= self.inter),
                            t.ToTensor(),
                            t.Normalize(self.means, self.stds),
                            t.RandomAffine(0, translate=(0,0.25), scale=(0.75,1.25)),
                        ])
            img = trans(img)
        elif choice=='erasing':
            trans = t.Compose([
                            t.Resize((self.img_size,self.img_size), interpolation= self.inter),
                            t.ToTensor(),
                            t.Normalize(self.means, self.stds),
                            t.RandomErasing(p=1)
                        ])
            img = trans(img)
            
        return img
    
# FUNCTIONS FOR BUILDING FASTAI MODEL IN PLAIN PYTORCH

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * 4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out
    
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return x


def make_layer(block, planes, blocks, inplanes, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = torch.nn.Sequential(
            torch.nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return torch.nn.Sequential(*layers)

class AdaptiveConcatPool2d(torch.nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        self.size = size or 1
        self.ap = torch.nn.AdaptiveAvgPool2d(self.size)
        self.mp = torch.nn.AdaptiveMaxPool2d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def build_fastai_model():
    return torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False),
                              torch.nn.BatchNorm2d(64),
                              torch.nn.ReLU(inplace=True),
                              torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                              make_layer(BasicBlock, 64, 2, 64),
                              make_layer(BasicBlock, 128, 2, 64, stride=2),
                              make_layer(BasicBlock, 256, 2, 128, stride=2),
                              make_layer(BasicBlock, 512, 2, 256, stride=2)
                              ),
                              torch.nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  torch.nn.Dropout(p=0.25),
                                  torch.nn.Linear(in_features=1024, out_features=512, bias=True),
                                  torch.nn.ReLU(inplace=True),
                                  torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  torch.nn.Dropout(p=0.5),
                                  torch.nn.Linear(in_features=512, out_features=2, bias=True)
                                 )
                             )

def initialize_model(num_classes, model_path, freeze=True, use_pretrained=True,
                     use_fai_classifier = True, model = 'resnet18'):
    if model == 'resnet18':
        #build fastai model in plain pytorch
        model_ft = build_fastai_model()
        if use_pretrained:
            model_ft.load_state_dict(torch.load(model_path), strict=False)
        if freeze:
            for param in model_ft[0].parameters():
                    param.requires_grad = False
    else:
        if use_pretrained:
            model_ft = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
            if freeze:
                for param in model_ft.parameters():
                    param.requires_grad = False
        else:
            model_ft = models.densenet121()

        #create a classifier head
        num_ftrs = model_ft.classifier.in_features
        if use_fai_classifier:
            model_ft.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.25),
                                torch.nn.Linear(in_features=num_ftrs, out_features=512, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=512, out_features=num_classes, bias=True))
        else:
            model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
    
    return model_ft