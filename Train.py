import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torchvision import models

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
            low_light_image, normal_light_image = self.transform(low_light_image, normal_light_image)

        return low_light_image, normal_light_image

# Define the custom joint transforms
class JointTransform:
    def __init__(self):
        self.resize = transforms.Resize((256, 256))
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __call__(self, img1, img2):
        # Resize
        img1 = self.resize(img1)
        img2 = self.resize(img2)

        # Random horizontal flip
        if random.random() < 0.5:
            img1 = transforms.functional.hflip(img1)
            img2 = transforms.functional.hflip(img2)

        # Random vertical flip
        if random.random() < 0.5:
            img1 = transforms.functional.vflip(img1)
            img2 = transforms.functional.vflip(img2)

        # Random rotate 90 degrees with probability 0.5
        if random.random() < 0.5:
            k = random.randint(1, 3)  # Rotate by 90, 180, or 270 degrees
            img1 = transforms.functional.rotate(img1, 90 * k)
            img2 = transforms.functional.rotate(img2, 90 * k)

        # Random color jitter with probability 0.5
        if random.random() < 0.5:
            img1 = self.color_jitter(img1)
            img2 = self.color_jitter(img2)

        # Convert to tensor
        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)

        # Normalize
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        return img1, img2

# Define the validation transforms
class ValTransform:
    def __init__(self):
        self.resize = transforms.Resize((256, 256))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __call__(self, img1, img2):
        img1 = self.resize(img1)
        img2 = self.resize(img2)

        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)

        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        return img1, img2

# Direktorier hvor billederne er gemt
low_light_dir = '/ceph/home/student.aau.dk/ov73uw/lol_dataset/our485/low'       # Lavt lys sti
normal_light_dir = '/ceph/home/student.aau.dk/ov73uw/lol_dataset/our485/high'   # Normalt lys sti

# Opret dataset med træningstransformationer
train_transform = JointTransform()
val_transform = ValTransform()

dataset = LOLDataset(low_light_dir, normal_light_dir, transform=train_transform)

# Opdel i trænings- og valideringsdatasæt
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# For valideringsdatasættet, overskriv transformeringen
val_dataset.dataset.transform = val_transform

# Opret data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Definer U-Net arkitekturen
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

# Perceptual Loss ved brug af VGG19
class PerceptualLoss(nn.Module):
    def __init__(self, layers=[0, 5, 10, 19, 28], weights=None):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:max(layers)+1].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layers = layers
        if weights is None:
            self.weights = [1.0 / len(layers)] * len(layers)
        else:
            self.weights = weights
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, input, target):
        input = (input + 1) / 2  # Skaler til [0,1]
        target = (target + 1) / 2

        input = (input - self.mean.to(input.device)) / self.std.to(input.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        input_features = []
        target_features = []

        x = input
        y = target
        for i in range(max(self.layers) + 1):
            x = self.vgg[i](x)
            y = self.vgg[i](y)
            if i in self.layers:
                input_features.append(x)
                target_features.append(y)
        loss = 0
        for inp_f, tgt_f, w in zip(input_features, target_features, self.weights):
            loss += w * nn.functional.l1_loss(inp_f, tgt_f)
        return loss

# Initialiser model, loss funktioner, optimizer, scheduler
model = UNet()
criterion = nn.MSELoss()
perceptual_criterion = PerceptualLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Tjek om CUDA er tilgængelig og brug GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
perceptual_criterion.to(device)

# Brug DataParallel hvis flere GPU'er er tilgængelige
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

scaler = GradScaler()
writer = SummaryWriter(log_dir='logs')

num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        low_light_images, normal_light_images = data
        low_light_images = low_light_images.to(device)
        normal_light_images = normal_light_images.to(device)

        # Nulstil gradienter
        optimizer.zero_grad()

        with autocast():
            # Fremadpass
            outputs = model(low_light_images)
            mse_loss = criterion(outputs, normal_light_images)
            perceptual_loss = perceptual_criterion(outputs, normal_light_images)
            loss = mse_loss + 0.01 * perceptual_loss  # Vægtsæt perceptual loss

        # Tilbagepas og optimer
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Print statistik hver 10. batch
        if i % 10 == 9:
            avg_loss = running_loss / 10
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {avg_loss:.5f}')
            writer.add_scalar('Training Loss', avg_loss, epoch * len(train_loader) + i)
            running_loss = 0.0

    # Valideringsloop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            low_light_images, normal_light_images = data
            low_light_images = low_light_images.to(device)
            normal_light_images = normal_light_images.to(device)

            with autocast():
                outputs = model(low_light_images)
                mse_loss = criterion(outputs, normal_light_images)
                perceptual_loss = perceptual_criterion(outputs, normal_light_images)
                loss = mse_loss + 0.01 * perceptual_loss

            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Validation Loss after epoch {epoch + 1}: {val_loss:.5f}')
    writer.add_scalar('Validation Loss', val_loss, epoch + 1)

    # Gem den bedste model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_unet_model.pth')
        print(f'Best model saved at epoch {epoch + 1}')

    # Opdater scheduler
    scheduler.step()

writer.close()
print('Finished Training')

# Gem den endelige trænede model
torch.save(model.state_dict(), 'unet_low_light_enhancement_model.pth')