import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import Net
from tqdm import tqdm
import os

# CUDA setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(1)

# Training hyperparameters
batch_size = 128
epochs = 10
lr = 0.01
momentum = 0.9

# Data transformations
train_transforms = transforms.Compose([
    transforms.RandomRotation((-15, 15)),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.2),
        scale=(0.9, 1.1),
        shear=(-10, 10)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Data loaders
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=train_transforms),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=test_transforms),
    batch_size=batch_size, shuffle=True, **kwargs)

def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(desc=f'Epoch {epoch+1} Loss={loss.item():.4f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return accuracy

def save_model(model, epoch, optimizer, accuracy, path='checkpoints'):
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }, f'{path}/model_epoch_{epoch}_acc_{accuracy:.2f}.pth')

def count_parameters(model):
    """
    Count the total number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Model initialization
    model = Net().to(device)
    total_params = count_parameters(model)
    print(f'Total trainable parameters: {total_params}')
    
    # For detailed layer-wise parameter count
    # print("\nLayer-wise parameter details:")
    # for name, parameter in model.named_parameters():
    #     if parameter.requires_grad:
    #         print(f"{name}: {parameter.numel():,}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        div_factor=10,
        final_div_factor=100,
    )
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        train(model, device, train_loader, optimizer, scheduler, epoch)
        accuracy = test(model, device, test_loader)
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, epoch, optimizer, accuracy)
            print(f'New best model saved with accuracy: {accuracy:.2f}%')
    
    # Save final model
    save_model(model, epochs-1, optimizer, accuracy, path='checkpoints/final')
    print(f'Training completed. Best accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    main()