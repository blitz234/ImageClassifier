from predict_utils import *
import torch
from torchvision import models
import json


def main():
    
    # get input arguments
    args = get_input_args()
    
    # check if gpu is to be used for training
    device = torch.device('cuda' if args.gpu else 'cpu')
    if device == torch.device('cuda') and not(torch.cuda.is_available()):
        print("GPU not available, cpu will be used for prediction.")
        device = torch.device('cpu')
    
    # select location to load checkpoint
    if torch.cuda.is_available() and device == torch.device('cuda'):
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    checkpoint = torch.load(args.checkpoint, map_location = map_location)
    
    # create the model
    model = load_checkpoint(checkpoint)
    
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # generate predictions
    inputs = process_image(args.input)
    
    # add the dimension for batch size
    inputs = inputs.unsqueeze(0).type(torch.FloatTensor)
    inputs = inputs.to(device)
    
    model.to(device);
    model.eval()
    
    with torch.no_grad():
        logps = model.forward(inputs)
        ps = torch.exp(logps)

        top_k, top_class = ps.topk(args.top_k, dim = 1)
        
        
        if args.gpu:
            top_class = top_class.cpu()
            top_k = top_k.cpu()
        
    probs, classes = top_k.numpy(), top_class.numpy()
    classes = [cat_to_name[model.idx_to_class[key]] for key in classes[0]]
    
    print_result(probs, classes)
        
if __name__ == "__main__":
    main()