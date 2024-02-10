import torch
import torch.nn as nn
from model import UNet3D
from dataloader import get_train_val_test_Dataloaders


train_transforms = None
val_transforms = None
test_transforms = None

# Get the dataloaders
train_dataloader, val_dataloader, test_dataloader = get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(n_channels=1, n_classes=2).to(device)  # n_channels and n_classes should be adjusted based on your dataset
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 25  # Set the number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_dataloader:  # Assuming train_dataloader is your DataLoader instance
        images = batch['image'].to(device, dtype=torch.float)
        true_masks = batch['label'].to(device, dtype=torch.long)

        image = images.unsqueeze(0)
        true_masks = true_masks.unsqueeze(0)

        optimizer.zero_grad()

        output_masks = model(images)
        loss = criterion(output_masks, true_masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}")