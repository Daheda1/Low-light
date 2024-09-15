import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the U-Net architecture (same as in training)
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

# Load the trained model
model = UNet()
model.load_state_dict(torch.load('unet_low_light_enhancement_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load and preprocess the image
input_image_path = 'lol_dataset/eval15/low/547.png'  # Replace with your image path
image = Image.open(input_image_path).convert('RGB')
input_image = transform(image)
input_image = input_image.unsqueeze(0)  # Add batch dimension

# Run the model
with torch.no_grad():
    output = model(input_image)

# Post-process and save the output image
output_image = output.squeeze(0)  # Remove batch dimension
output_image = output_image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
output_image = output_image.numpy()
output_image = np.clip(output_image, 0, 1)
output_image = (output_image * 255).astype(np.uint8)

output_image = Image.fromarray(output_image)
output_image.save('enhanced_image.jpg')

print('Enhanced image saved as enhanced_image.jpg')