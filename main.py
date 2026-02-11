import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 
import matplotlib.pyplot as plot

epochs = 10

class HandNumDetectorLinear(nn.Module):
    def __init__(self):
        super().__init__()

        # previour linear layer
        self.layer1 = nn.Linear(784, 128) # first layer param: input_size(28 x 28 = 784), hidden_size(random, 128 is standard)
        self.layer2 = nn.Linear(128, 10)  # second layer param: hidden_size(same), desired_output_size(0-9 means 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

class HandNumDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.classifier(x)
        return x

HandNumDetector = HandNumDetectorCNN()
# transform raw image data into tensors
transform_pipeline = transforms.Compose([
    transforms.ToTensor()
])

def main():
    # store gpu info
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate the class
    HNmodel = HandNumDetector.to(gpu_device)

    # download training data
    train_set = torchvision.datasets.MNIST(
        root= "./data",
        train= True,
        download= True,
        transform= transform_pipeline
    )

    # download test data
    test_set = torchvision.datasets.MNIST(
        root= "./data",
        train= False,
        download= True,
        transform= transform_pipeline
    )

    # load the training dataset
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size= 64,
        shuffle= True
    )

    # load the test dataset
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle= True,
        batch_size= 64
    )

    # create the loss function & optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(HNmodel.parameters(), lr = 0.001)

    # the training loop
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # move data to gpu
            images = images.to(gpu_device)
            labels = labels.to(gpu_device)

            # forward pass
            outputs = HNmodel(images)
            loss = loss_function(outputs, labels)

            # backward & optimize
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            # print progress after each 64 batches
            if (i + 1) % 64 == 0:
                print(f"Epoch: {epoch + 1}/{epochs}, Step: {i + 1}, Loss: {loss.item():.4f}")

    # switch to evaluation mode
    HNmodel.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(gpu_device)
            labels = labels.to(gpu_device)

            outputs = HNmodel(images)

            # torch.max returns (max_value, index_of_max_value)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy: {100 * correct / total}%")

    torch.save(HNmodel.state_dict(), "HNmodel2.pth")

if __name__ == "__main__":
    main()