from train_utils import *
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


def main():
    
    # get the command line arguments
    args = get_input_args()
    
    # load the data
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_transform = transforms.Compose([transforms.RandomRotation(35),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    validation_transform = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    validation_data = datasets.ImageFolder(valid_dir, transform = validation_transform)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size = 32, shuffle = True)
    
    # create the model
    model = create_model(args)
    
    device = torch.device('cuda' if args.gpu else 'cpu')
    if device == torch.device('cuda') and not(torch.cuda.is_available()):
        print("GPU not available, Training stopped.")
        return None
    
    params = model.fc.parameters() if args.arch == 'resnet34' else model.classifier.parameters()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params, lr = args.learning_rate)
    
    model.to(device);
    
    
    # Training model classifier
    epochs = args.epochs
    
    model.train()
    
    
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            validation_loss = 0
            validation_accuracy = 0
            model.eval()

            for inputs, label in validationloader:
                inputs, label = inputs.to(device), label.to(device)

                logps = model.forward(inputs)
                validation_loss += criterion(logps, label).item()

                ps = torch.exp(logps)
                top_k, top_class = ps.topk(1, dim = 1)

                equals = top_class == label.view(*top_class.shape)
                validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()



            print(f"Epoch {e+1}/{epochs} ",
                  "Training loss: {:.3f}".format(running_loss/len(trainloader)),
                  "Validation loss: {:.3f}".format(validation_loss/len(validationloader)),
                  "Validation Accuracy: {:.3f}".format(validation_accuracy/len(validationloader)))
       
            model.train()
        
        
    # Save checkpoint 
    model.idx_to_class = { v: k for k, v in train_data.class_to_idx.items()}
    
    model_arch = models.resnet34 if args.arch == 'resnet34' else models.densenet121
    classifier = model.fc if args.arch == 'resnet34' else model.classifier
    
    checkpoint = {'model': model_arch,
              'classifier': classifier ,
              'state_dict': model.state_dict(),
              'epochs': epochs,
              'optimizer_state': optimizer.state_dict,
              'idx_to_class': model.idx_to_class}
    
    torch.save(checkpoint, args.save_dir)
    
if __name__ == "__main__":
    main()