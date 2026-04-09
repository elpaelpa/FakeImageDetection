import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data import load_data
from model import Model
from PIL import Image
import time
import argparse

#load data
train_loader,test_loader = load_data()

#Device to run train on 
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

#Model CNN 
model = Model(num_classes=2).to(device)

#Use Cross Entropy Loss for classification
criterion = nn.CrossEntropyLoss()

#Stochastic Gradient Descent optimizer
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

#Train the models for epochs
def train(epochs=10,model_chosen='model.pth'):
    total_start = time.time()

    #Running Epochs, defaults to 10 epochs
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")
    
    total_time = time.time() - total_start
    print(f"Total time: {total_time:.2f}s")



#Evaluate the model using test dataset
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

#Test using provided image
def predict(image_path:str):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    classes = ['ai', 'real'] 
    print(f"Prediction: {classes[predicted.item()]}")


#Arguments for function calls
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['train', 'test', 'predict'])
parser.add_argument('--image', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--model',type=str)
args = parser.parse_args()

if args.mode == 'train':
    print(f"using device:{device}")
    train(args.epochs)
    torch.save(model.state_dict(),args.model)
    print(f"Model saved to {args.model}")

elif args.mode == 'test':
    model.load_state_dict(torch.load(args.model,weights_only=True))
    evaluate()
elif args.mode == 'predict':
    model.load_state_dict(torch.load(args.model,weights_only=True))
    predict(args.image)