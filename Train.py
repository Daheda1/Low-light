import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Custom Dataset for the LOL dataset
class LOLDataset(Dataset):
    def __init__(self, low_light_dir, normal_light_dir, transform=None):
        self.low_light_dir = low_light_dir
        self.normal_light_dir = normal_light_dir
        self.transform = transform
        self.image_names = os.listdir(self.low_light_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        low_light_image_path = os.path.join(self.low_light_dir, self.image_names[idx])
        normal_light_image_path = os.path.join(self.normal_light_dir, self.image_names[idx])

        low_light_image = Image.open(low_light_image_path).convert('RGB')
        normal_light_image = Image.open(normal_light_image_path).convert('RGB')

        if self.transform:
            low_light_image = self.transform(low_light_image)
            normal_light_image = self.transform(normal_light_image)

        return low_light_image, normal_light_image

# Example transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Directories where the images are stored
low_light_dir = 'data/low'       # Replace with your path
normal_light_dir = 'data/high' # Replace with your path

# Create dataset and dataloader
dataset = LOLDataset(low_light_dir, normal_light_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = CBR(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec6 = CBR(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec7 = CBR(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec8 = CBR(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec9 = CBR(128, 64)

        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)

        enc5 = self.enc5(pool4)

        # Decoder
        up6 = self.up6(enc5)
        merge6 = torch.cat([up6, enc4], dim=1)
        dec6 = self.dec6(merge6)

        up7 = self.up7(dec6)
        merge7 = torch.cat([up7, enc3], dim=1)
        dec7 = self.dec7(merge7)

        up8 = self.up8(dec7)
        merge8 = torch.cat([up8, enc2], dim=1)
        dec8 = self.dec8(merge8)

        up9 = self.up9(dec8)
        merge9 = torch.cat([up9, enc1], dim=1)
        dec9 = self.dec9(merge9)

        out = self.conv_last(dec9)
        return out

# Initialize model, loss function, optimizer
model = UNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Check if CUDA is available and use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        low_light_images, normal_light_images = data
        low_light_images = low_light_images.to(device)
        normal_light_images = normal_light_images.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(low_light_images)
        loss = criterion(outputs, normal_light_images)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print statistics every 10 batches
        if i % 10 == 9:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.5f}')
            running_loss = 0.0

    # Save the model checkpoint after each epoch
    torch.save(model.state_dict(), f'unet_model_epoch_{epoch+1}.pth')

print('Finished Training')

# Save the final trained model
torch.save(model.state_dict(), 'unet_low_light_enhancement_model.pth')