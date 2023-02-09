import argparse
import torch
from torch import nn
from torchvision import models


def get_input_args():
    
    parser = argparse.ArgumentParser(description = 'Train the model.')
    
    parser.add_argument('data_dir', type = str,
                       help = 'Path to directory with Image Datasets.'+
                       'Keep the training data inside the "train" folder' +
                       ' and validiation data inside "valid" folder in the directory.')
    
    parser.add_argument('--save_dir', type = str,
                       help = 'Path to directory to store checkpoints along with the filename'+
                       ' ex : directory_name/file_name.pth',
                       default = "checkpoint.pth")
    
    parser.add_argument('--arch', type = str,
                       help = 'Model Architecture (resnet34, densenet121), default: resnet34',
                       default = 'resnet34')
    
    parser.add_argument('--learning_rate', type = float,
                        help = 'Learning Rate, default: 0.01',
                        default = 0.01)
    
    parser.add_argument('--hidden_units', type = int,
                        help = 'Number of hidden units, default: 512',
                        default = 512)
    
    parser.add_argument('--epochs', type = int,
                        help = 'Number of epochs, default: 5',
                        default = 5)
    
    parser.add_argument('--gpu', action = 'store_true',
                        help = 'Use GPU for Training')
    
    return parser.parse_args()


def create_model(args):
    
    # dictionary with model definitions
    model_dic = {'resnet34': models.resnet34,
                 'densenet121': models.densenet121}
    
    model = model_dic[args.arch](pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    in_features = model.fc.in_features if args.arch == 'resnet34' \
                  else model.classifier.in_features
    
    if args.arch == 'densenet121':
        model.classifier = nn.Sequential(nn.Linear(in_features, args.hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(p = 0.2),
                                        nn.Linear(args.hidden_units, 102),
                                        nn.LogSoftmax(dim = 1))
        
    else:
        model.fc = nn.Sequential(nn.Linear(in_features, args.hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(p = 0.2),
                                        nn.Linear(args.hidden_units, 102),
                                        nn.LogSoftmax(dim = 1))

    
    return model