import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import pandas as pd

def get_input_args():
    
    parser = argparse.ArgumentParser(description = 'Make predictions based on the checkpoint of a pretrained model')
    
    parser.add_argument('input', type = str,
                       help = 'Path to Image.')
    
    parser.add_argument('checkpoint', type = str,
                       help = 'Path to checkpoint file.')
    
    parser.add_argument('--top_k', type = int,
                        help = 'Results must include top k classes',
                        default = 3)
    
    parser.add_argument('--category_names', type = str,
                        help = 'File that contains mapping of categories to real names.')
    
    parser.add_argument('--gpu', action = 'store_true',
                        help = 'Use GPU for Training')
    
    return parser.parse_args()


def load_checkpoint(checkpoint):
    
    model = checkpoint['model']()
    
    if checkpoint['model'] == models.resnet34:
        model.fc = checkpoint['classifier']
    else:
        model.classifier = checkpoint['classifier']
        
    model.idx_to_class = checkpoint['idx_to_class']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a tensor
    '''
    im = Image.open(image)
    
    # resize the image where the shortest side is 256 pixels
    width, height = im.size
    
    if width < height:
        new_width = 256
        new_height = int(256 * height/width)
    else:
        new_height = 256
        new_width = int(256 * width/height)
        
    im1 = im.resize((new_width, new_height))
    
    # center crop - 224 x 224 pixels
    left = (new_width - 224)/2
    right = (new_width + 224)/2
    top = (new_height - 224)/2
    bottom = (new_height + 224)/2
    
    im1 = im1.crop((left, top, right, bottom))
    
    # convert pil image to numpy array for normalization
    np_image = np.array(im1)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    np_image = (np_image/255 - mean) / std
    
    # make the color channel the first dimension
    np_image = np.transpose(np_image, (2,0,1))
    
    
    tensor_image = torch.from_numpy(np_image)
    
    return tensor_image

def print_result(probs, classes):
    result = pd.DataFrame({'Category': classes, 'Probability':probs[0]},
                          index = range(1, len(classes)+1),
                          columns = ['Category', 'Probability'])
    print(result)
    

