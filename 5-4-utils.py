from torch import nn,optim
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
import numpy as np
import json

def loader_data(data_dir=''):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir , transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_data = datasets.ImageFolder(test_dir , transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    return train_data,trainloader,validloader,testloader

def setupNN(dropout,hiddenlayers,arch):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False



    # number of filters in the bottleneck layer
    if arch=='vgg16':
        num_filters = 25088
    else:
        num_filters=1024



    from collections import OrderedDict
    from torch import nn
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_filters, hiddenlayers)),
        ('relu', nn.ReLU()),
        ('dropout',nn.Dropout(dropout)),
        ('fc2', nn.Linear(hiddenlayers, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return model

def RunSaveNN(model,pu,learningrate,epochs,trainloader,validloader,hiddenlayer,checkpoint_pth,arch,train_data):
    if pu=='cpu':
        device='cpu'
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learningrate)
    model.to(device);
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()


    #     model.class_to_idx = trainloader.class_to_idx

    model.class_to_idx = train_data.class_to_idx
    state = {
        'arch': arch,
        'learning_rate': learningrate,
        'hidden_layers':hiddenlayer ,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'class_to_idx':model.class_to_idx,
        'classifier':model.classifier,
    }

    torch.save(state, checkpoint_pth)


def process_image(image):
    from PIL import Image
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img=Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_processed = img_transforms(img)

    return img_processed.numpy()

def predict_(image_path, model, top_k,jsonfile):
    ''' Predict the class (or classes) of an image using a trained deep learning model.

    image_path: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    top_k: integer. The top K classes to be calculated

    returns top_probabilities(k), top_labels
    '''

    # No need for GPU on this part (just causes problems)
    model.to("cpu")

    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path),
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)

    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] # This is not the correct way to do it but the correct way isnt working thanks to cpu/gpu issues so I don't care.
    top_labels = np.array(top_labels.detach())[0]

    with open(jsonfile, 'r') as f:
        cat_to_name = json.load(f)
    # Convert to classes
    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    print(top_labels)
    print(top_flowers)
    print(top_probs)


    return top_probs, top_labels,top_flowers

def predict(chkpnt_file,target_dir,jsonfile,top_k):
    checkpoint = torch.load(chkpnt_file)

    # Download pretrained model
    if checkpoint['arch'] == 'vgg16':
        model_loaded = models.vgg16(pretrained=True)
        num_features=25088
    elif checkpoint['arch'] == 'densenet121':
        model_loaded = models.densenet121(pretrained=True)
        num_features=1024


    for param in model_loaded.parameters(): param.requires_grad = False

    # Load stuff from checkpoint


    model_loaded.class_to_idx = checkpoint['class_to_idx']
    model_loaded.classifier = checkpoint['classifier']
    model_loaded.load_state_dict(checkpoint['state_dict'])
    predict_(target_dir,model_loaded,top_k,jsonfile)
