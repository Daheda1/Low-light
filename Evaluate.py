import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error,
)
from tqdm import tqdm

# Custom Dataset for the LOL dataset (Test Set)
class LOLDatasetTest(Dataset):
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

        return low_light_image, normal_light_image, self.image_names[idx]

# Example transformations (ensure consistency with training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Directories where the test images are stored
test_low_light_dir = 'lol_dataset/eval15/low'       # Replace with your test low-light images path
test_normal_light_dir = 'lol_dataset/eval15/high' # Replace with your test normal-light images path

# Create test dataset and dataloader
test_dataset = LOLDatasetTest(test_low_light_dir, test_normal_light_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

# Initialize model and load trained weights
model = UNet()
model.load_state_dict(torch.load('unet_low_light_enhancement_model.pth'))

# Check if CUDA is available and use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Metrics initialization
psnr_values = []
ssim_values = []
mse_values = []
rmse_values = []
mae_values = []
image_names_list = []

# Evaluation loop
with torch.no_grad():
    for data in tqdm(test_dataloader, desc='Evaluating'):
        low_light_images, normal_light_images, image_names = data
        low_light_images = low_light_images.to(device)
        normal_light_images = normal_light_images.to(device)

        # Forward pass
        outputs = model(low_light_images)

        # Move tensors to CPU and convert to numpy arrays
        outputs_np = outputs.cpu().numpy().squeeze().transpose(1, 2, 0)
        normal_images_np = normal_light_images.cpu().numpy().squeeze().transpose(1, 2, 0)

        # Ensure the pixel values are in [0,1]
        outputs_np = np.clip(outputs_np, 0, 1)
        normal_images_np = np.clip(normal_images_np, 0, 1)

        # Compute MSE
        mse = mean_squared_error(normal_images_np, outputs_np)
        mse_values.append(mse)

        # Compute RMSE
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)

        # Compute MAE
        mae = np.mean(np.abs(outputs_np - normal_images_np))
        mae_values.append(mae)

        # Compute PSNR
        psnr = peak_signal_noise_ratio(normal_images_np, outputs_np, data_range=1)
        psnr_values.append(psnr)

        # Compute SSIM
        ssim = structural_similarity(normal_images_np, outputs_np, multichannel=True, data_range=1)
        ssim_values.append(ssim)

        image_names_list.append(image_names[0])

# Compute average metrics
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)
avg_mse = np.mean(mse_values)
avg_rmse = np.mean(rmse_values)
avg_mae = np.mean(mae_values)

# Save metrics to a txt file
with open('evaluation_metrics.txt', 'w') as f:
    f.write('Evaluation Metrics on Test Set\n')
    f.write('-------------------------------\n')
    f.write(f'Average PSNR: {avg_psnr:.4f} dB\n')
    f.write(f'Average SSIM: {avg_ssim:.4f}\n')
    f.write(f'Average MSE: {avg_mse:.6f}\n')
    f.write(f'Average RMSE: {avg_rmse:.6f}\n')
    f.write(f'Average MAE: {avg_mae:.6f}\n')

    f.write('\nIndividual Image Metrics:\n')
    f.write('Image Name\tPSNR(dB)\tSSIM\t\tMSE\t\tRMSE\t\tMAE\n')
    for idx, image_name in enumerate(image_names_list):
        f.write(f'{image_name}\t{psnr_values[idx]:.4f}\t\t{ssim_values[idx]:.4f}\t{mse_values[idx]:.6f}\t{rmse_values[idx]:.6f}\t{mae_values[idx]:.6f}\n')

print('Evaluation complete. Metrics saved to evaluation_metrics.txt.')