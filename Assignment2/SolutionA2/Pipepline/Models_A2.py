import torch
import torchaudio
import torchvision
import torch.nn as nn
from Pipeline import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchaudio.datasets import SPEECHCOMMANDS
from torchvision import transforms
import os
import librosa
import numpy as np 
import torch.nn.functional as F
import math

# Function to check if data folder exists, if not, create it
def create_data_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Check and create data folders if needed
current_directory = os.getcwd()
image_data_folder = "Image_train_Data"
audio_data_folder = "Audio_train_Data"
create_data_folder(os.path.join(current_directory, image_data_folder))
create_data_folder(os.path.join(current_directory, audio_data_folder))

# for maintaing best model
CHECKPOINT_DIR = 'checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

##    Download the data
image_dataset_downloader = torchvision.datasets.CIFAR10(
    root=image_data_folder,
    download=True,
)

audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(
    root=audio_data_folder,
    download=True,
)

# Define transforms for image data
image_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Use dictionary comprehension to map labels to integers audio data
label_map = {'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9,
                 'zero': 10, 'one': 11, 'two': 12, 'three': 13, 'four': 14, 'five': 15, 'six': 16, 'seven': 17,
                 'eight': 18, 'nine': 19, 'backward': 20, 'bed': 21, 'bird': 22, 'cat': 23, 'dog': 24, 'follow': 25,
                 'forward': 26, 'happy': 27, 'house': 28, 'learn': 29, 'marvin': 30, 'sheila': 31, 'tree': 32,
                 'visual': 33, 'wow': 34}


# imageDataset 
class ImageDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")

        self.datasplit = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),

        ])
        datasetTrain = torchvision.datasets.CIFAR10(image_data_folder,train=True,download=False)
        train_size = int(0.8 * len(datasetTrain))
        val_size = len(datasetTrain) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(datasetTrain, [train_size, val_size])

        # Now we will split the datasetTrain into train, val
        if(self.datasplit=='train'):
            self.dataset = train_dataset

        elif(self.datasplit=='val'):
            self.dataset = val_dataset

        elif(self.datasplit=='test'):
            self.dataset = torchvision.datasets.CIFAR10(image_data_folder,train=False,download=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
      image, label = self.dataset[idx]
      image = image_transform(image)
      return image,label


#AudioDataset
class AudioDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")

        self.datasplit = split
        self.max_length = 64
        self.sample_rate = 16000  # sample rate for simplicity
        if self.datasplit == 'train':
            self.dataset = SPEECHCOMMANDS(audio_data_folder, download=False, subset='training')

        elif self.datasplit == 'val':
            self.dataset = SPEECHCOMMANDS(audio_data_folder, download=False, subset='validation')

        elif self.datasplit == 'test':
            self.dataset = SPEECHCOMMANDS(audio_data_folder, download=False, subset='testing')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]

        # Convert waveform to NumPy array
        waveform = waveform.numpy()[0]

        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=self.sample_rate, n_fft=1024)

        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale

        # Normalize Mel spectrogram
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())

        # Resize or pad Mel spectrogram to a fixed size
        mel_spec_resized = np.zeros((128, 64))
        mel_spec_resized[:, :mel_spec.shape[1]] = mel_spec[:, :self.max_length]

        # Convert to PyTorch tensor
        mel_spec_tensor = torch.from_numpy(mel_spec_resized).float()

        label = label_map[label]
        return mel_spec_tensor, label
    

class Resnet_Q1(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_image = self._make_layer(64, 4, is_image=True)
        self.layer2_image = self._make_layer(128, 5, stride=2, is_image=True)
        self.layer3_image = self._make_layer( 256, 5, stride=2, is_image=True)
        self.layer4_image = self._make_layer( 512, 4, stride=2, is_image=True)
        self.avgpool_image = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_image = nn.Linear(512 * self.ResidualBlock.expansion, 10)

        self.in_channels = 64
        self.conv1d = nn.Conv1d(128, 64, kernel_size=7, stride=2, padding=3)
        self.bn1d = nn.BatchNorm1d(64)
        self.maxpool1d = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1_audio = self._make_layer( 64, 4, stride=1, is_image=False)
        self.layer2_audio = self._make_layer( 128, 5, stride=2, is_image=False)
        self.layer3_audio = self._make_layer( 256, 5, stride=2, is_image=False)
        self.layer4_audio = self._make_layer( 512, 4, stride=2, is_image=False)
        self.avgpool_audio = nn.AdaptiveAvgPool1d(1)
        self.fc_audio = nn.Linear(512* self.ResidualBlock.expansion, 35)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% probability

    def _make_layer(self, out_channels, blocks, stride=1, is_image=False):
        layers = []
        in_channels = self.in_channels
        layers.append(self.ResidualBlock(in_channels, out_channels, stride=stride, is_image=is_image))
        in_channels = out_channels * self.ResidualBlock.expansion
        for _ in range(1, blocks):
            layers.append(self.ResidualBlock(in_channels, out_channels, is_image=is_image))
        self.in_channels = out_channels * self.ResidualBlock.expansion  # Update in_channels for next layer
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 4:  # Image data
            x = self.conv1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1_image(x)
            x = self.layer2_image(x)
            x = self.layer3_image(x)
            x = self.layer4_image(x)

            x = self.avgpool_image(x)
            x = x.view(x.size(0), -1)
            x = self.fc_image(x)

        elif x.dim() == 3:  # Audio data
            x = self.conv1d(x)
            x = self.bn1d(x)
            x = self.relu(x)
            x = self.maxpool1d(x)

            x = self.layer1_audio(x)
            x = self.dropout(x)  # Applying dropout after the first audio layer
            x = self.layer2_audio(x)
            x = self.dropout(x)  # Applying dropout after the second audio layer
            x = self.layer3_audio(x)
            x = self.dropout(x)  # Applying dropout after the third audio layer
            x = self.layer4_audio(x)
            x = self.avgpool_audio(x)
            x = x.view(x.size(0), -1)
            x = self.fc_audio(x)
        else:
            raise ValueError("Input must be 3D (audio) or 4D (image)")

        return x
    class ResidualBlock(nn.Module):
      expansion = 1
      def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_image=True):
          super(Resnet_Q1.ResidualBlock, self).__init__()
          #print("is_image",is_image)
          conv = nn.Conv2d if is_image else nn.Conv1d
          BatchNorm = nn.BatchNorm2d if is_image else nn.BatchNorm1d
          self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
          self.bn1 = BatchNorm(out_channels)
          self.relu = nn.ReLU(inplace=True)
          self.conv2 = conv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
          self.bn2 =BatchNorm(out_channels)
          self.downsample = nn.Sequential(
              conv(in_channels, out_channels, kernel_size=1, stride=stride),
              BatchNorm(out_channels),
          ) if stride != 1 or in_channels != out_channels else None
          # self.is_image = is_image
      def forward(self, x):
          identity = x
          # print('is_image',self.is_image)

          out = self.conv1(x)
          out = self.bn1(out)
          out = self.relu(out)

          out = self.conv2(out)
          out = self.bn2(out)

          if self.downsample is not None:
              identity = self.downsample(x)

          # Adjust dimensions of identity to match those of out if needed
          if identity.shape != out.shape:
              identity = F.pad(identity, (0, out.shape[-1] - identity.shape[-1], 0, out.shape[-2] - identity.shape[-2]))

          out += identity
          out = self.relu(out)
          return out

      

### VGG      
class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # block1 image
        self.channel1 = 256
        self.kernel1 = 3
        self.conv11 = nn.Conv2d(3, self.channel1, kernel_size=self.kernel1, padding=(self.kernel1 - 1) // 2)
        self.bn11 = nn.BatchNorm2d(self.channel1)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(self.channel1, self.channel1, kernel_size=self.kernel1, padding=(self.kernel1 - 1) // 2)
        self.bn12 = nn.BatchNorm2d(self.channel1)
        self.relu12 = nn.ReLU(inplace=True)
        self.poolingb1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # block2 image
        self.channel2 = math.ceil(self.channel1 - (self.channel1 * 0.35))
        self.kernel2 = self.nearest_odd(self.kernel1 + (self.kernel1 * 0.25))
        self.conv21 = nn.Conv2d(self.channel1, self.channel2, kernel_size=self.kernel2, padding=(self.kernel2 - 1) // 2)
        self.bn21 = nn.BatchNorm2d(self.channel2)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(self.channel2, self.channel2, kernel_size=self.kernel2, padding=(self.kernel2 - 1) // 2)
        self.bn22 = nn.BatchNorm2d(self.channel2)
        self.relu22 = nn.ReLU(inplace=True)
        self.poolingb2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # block3 image 
        self.channel3 = math.ceil(self.channel2 - (self.channel2 * 0.35))
        self.kernel3 = self.nearest_odd(self.kernel2 + (self.kernel2 * 0.25))
        self.conv31 = nn.Conv2d(self.channel2, self.channel3, kernel_size=self.kernel3, padding=(self.kernel3 - 1) // 2)
        self.bn31 = nn.BatchNorm2d(self.channel3)
        self.relu31 = nn.ReLU(inplace=True)
        self.conv32 = nn.Conv2d(self.channel3, self.channel3, kernel_size=self.kernel3, padding=(self.kernel3 - 1) // 2)
        self.bn32 = nn.BatchNorm2d(self.channel3)
        self.relu32 = nn.ReLU(inplace=True)
        self.conv33 = nn.Conv2d(self.channel3, self.channel3, kernel_size=self.kernel3, padding=(self.kernel3 - 1) // 2)
        self.bn33 = nn.BatchNorm2d(self.channel3)
        self.relu33 = nn.ReLU(inplace=True)
        self.poolingb3 = nn.MaxPool2d(kernel_size=3, stride=1)

        # block4 image
        self.channel4 = math.ceil(self.channel3 - (self.channel3 * 0.35))
        self.kernel4 = self.nearest_odd(self.kernel3 + (self.kernel3 * 0.25))
        self.conv41 = nn.Conv2d(self.channel3, self.channel4, kernel_size=self.kernel4, padding=(self.kernel4 - 1) // 2)
        self.bn41 = nn.BatchNorm2d(self.channel4)
        self.relu41 = nn.ReLU(inplace=True)
        self.conv42 = nn.Conv2d(self.channel4, self.channel4, kernel_size=self.kernel4, padding=(self.kernel4 - 1) // 2)
        self.bn42 = nn.BatchNorm2d(self.channel4)
        self.relu42 = nn.ReLU(inplace=True)
        self.conv43 = nn.Conv2d(self.channel4, self.channel4, kernel_size=self.kernel4, padding=(self.kernel4 - 1) // 2)
        self.bn43 = nn.BatchNorm2d(self.channel4)
        self.relu43 = nn.ReLU(inplace=True)
        self.poolingb4 = nn.MaxPool2d(kernel_size=3, stride=1)

        # block5 image
        self.channel5 = math.ceil(self.channel4 - (self.channel4 * 0.35))
        self.kernel5 = self.nearest_odd(self.kernel4 + (self.kernel4 * 0.25))
        self.conv51 = nn.Conv2d(self.channel4, self.channel5, kernel_size=self.kernel5, padding=(self.kernel5 - 1) // 2)
        self.bn51 = nn.BatchNorm2d(self.channel5)
        self.relu51 = nn.ReLU(inplace=True)
        self.conv52 = nn.Conv2d(self.channel5, self.channel5, kernel_size=self.kernel5, padding=(self.kernel5 - 1) // 2)
        self.bn52 = nn.BatchNorm2d(self.channel5)
        self.relu52 = nn.ReLU(inplace=True)
        self.conv53 = nn.Conv2d(self.channel5, self.channel5, kernel_size=self.kernel5, padding=(self.kernel5 - 1) // 2)
        self.bn53 = nn.BatchNorm2d(self.channel5)
        self.relu53 = nn.ReLU(inplace=True)
        self.poolingb5 = nn.MaxPool2d(kernel_size=3, stride=1)

        ## Dense layer image
        self.denselayer1 = nn.Linear(1 * 1 * self.channel5, 512)

        self.denselayer2 = nn.Linear(512, 256)

        self.denselayer3 = nn.Linear(256, 10)

        # block1 audio
        self.channel1_aud = 256
        self.kernel1_aud = 3
        self.conv11_aud = nn.Conv1d(128, self.channel1, kernel_size=self.kernel1, padding=(self.kernel1 - 1) // 2)
        self.bn11_aud = nn.BatchNorm1d(self.channel1)
        self.relu11_aud = nn.ReLU(inplace=True)
        self.conv12_aud = nn.Conv1d(self.channel1, self.channel1, kernel_size=self.kernel1, padding=(self.kernel1 - 1) // 2)
        self.bn12_aud = nn.BatchNorm1d(self.channel1)
        self.relu12_aud = nn.ReLU(inplace=True)
        self.poolingb1_aud = nn.MaxPool1d(kernel_size=3, stride=2)

        # block2 audio
        self.channel2_aud = math.ceil(self.channel1 - (self.channel1 * 0.35))
        self.kernel2_aud = self.nearest_odd(self.kernel1 + (self.kernel1 * 0.25))
        self.conv21_aud = nn.Conv1d(self.channel1, self.channel2, kernel_size=self.kernel2, padding=(self.kernel2 - 1) // 2)
        self.bn21_aud = nn.BatchNorm1d(self.channel2)
        self.relu21_aud = nn.ReLU(inplace=True)
        self.conv22_aud = nn.Conv1d(self.channel2, self.channel2, kernel_size=self.kernel2, padding=(self.kernel2 - 1) // 2)
        self.bn22_aud = nn.BatchNorm1d(self.channel2)
        self.relu22_aud = nn.ReLU(inplace=True)
        self.poolingb2_aud = nn.MaxPool1d(kernel_size=3, stride=2)

        # block3 aduio
        self.channel3_aud = math.ceil(self.channel2 - (self.channel2 * 0.35))
        self.kernel3_aud = self.nearest_odd(self.kernel2 + (self.kernel2 * 0.25))
        self.conv31_aud = nn.Conv1d(self.channel2, self.channel3, kernel_size=self.kernel3, padding=(self.kernel3 - 1) // 2)
        self.bn31_aud = nn.BatchNorm1d(self.channel3)
        self.relu31_aud = nn.ReLU(inplace=True)
        self.conv32_aud = nn.Conv1d(self.channel3, self.channel3, kernel_size=self.kernel3, padding=(self.kernel3 - 1) // 2)
        self.bn32_aud = nn.BatchNorm1d(self.channel3)
        self.relu32_aud = nn.ReLU(inplace=True)
        self.conv33_aud = nn.Conv1d(self.channel3, self.channel3, kernel_size=self.kernel3, padding=(self.kernel3 - 1) // 2)
        self.bn33_aud = nn.BatchNorm1d(self.channel3)
        self.relu33_aud = nn.ReLU(inplace=True)
        self.poolingb3_aud = nn.MaxPool1d(kernel_size=3, stride=1)

        # block4 audio
        self.channel4_aud = math.ceil(self.channel3 - (self.channel3 * 0.35))
        self.kernel4_aud = self.nearest_odd(self.kernel3 + (self.kernel3 * 0.25))
        self.conv41_aud = nn.Conv1d(self.channel3, self.channel4, kernel_size=self.kernel4, padding=(self.kernel4 - 1) // 2)
        self.bn41_aud = nn.BatchNorm1d(self.channel4)
        self.relu41_aud = nn.ReLU(inplace=True)
        self.conv42_aud = nn.Conv1d(self.channel4, self.channel4, kernel_size=self.kernel4, padding=(self.kernel4 - 1) // 2)
        self.bn42_aud = nn.BatchNorm1d(self.channel4)
        self.relu42_aud = nn.ReLU(inplace=True)
        self.conv43_aud = nn.Conv1d(self.channel4, self.channel4, kernel_size=self.kernel4, padding=(self.kernel4 - 1) // 2)
        self.bn43_aud = nn.BatchNorm1d(self.channel4)
        self.relu43_aud = nn.ReLU(inplace=True)
        self.poolingb4_aud = nn.MaxPool1d(kernel_size=3, stride=1)

        # block5 audio
        self.channel5_aud = math.ceil(self.channel4 - (self.channel4 * 0.35))
        self.kernel5_aud = self.nearest_odd(self.kernel4 + (self.kernel4 * 0.25))
        self.conv51_aud = nn.Conv1d(self.channel4, self.channel5, kernel_size=self.kernel5, padding=(self.kernel5 - 1) // 2)
        self.bn51_aud = nn.BatchNorm1d(self.channel5)
        self.relu51_aud = nn.ReLU(inplace=True)
        self.conv52_aud = nn.Conv1d(self.channel5, self.channel5, kernel_size=self.kernel5, padding=(self.kernel5 - 1) // 2)
        self.bn52_aud = nn.BatchNorm1d(self.channel5)
        self.relu52_aud = nn.ReLU(inplace=True)
        self.conv53_aud = nn.Conv1d(self.channel5, self.channel5, kernel_size=self.kernel5, padding=(self.kernel5 - 1) // 2)
        self.bn53_aud = nn.BatchNorm1d(self.channel5)
        self.relu53_aud = nn.ReLU(inplace=True)
        self.poolingb5_aud = nn.MaxPool1d(kernel_size=3, stride=1)

        ## Dense layer audio
        self.denselayer1_aud = nn.Linear(423, 512)
        self.denselayer2_aud = nn.Linear(512, 256)
        self.denselayer3_aud = nn.Linear(256, 35)



    def nearest_odd(self, number):
        return math.ceil(number) if math.ceil(number) % 2 != 0 else math.ceil(number) + 1

    def forward(self, x):
        if(x.dim()==4):
          x = self.relu11(self.bn11(self.conv11(x)))
          x = self.relu12(self.bn12(self.conv12(x)))
          x = self.poolingb1(x)

          x = self.relu21(self.bn21(self.conv21(x)))
          x = self.relu22(self.bn22(self.conv22(x)))
          x = self.poolingb2(x)

          x = self.relu31(self.bn31(self.conv31(x)))
          x = self.relu32(self.bn32(self.conv32(x)))
          x = self.relu33(self.bn33(self.conv33(x)))
          x = self.poolingb3(x)

          x = self.relu41(self.bn41(self.conv41(x)))
          x = self.relu42(self.bn42(self.conv42(x)))
          x = self.relu43(self.bn43(self.conv43(x)))
          x = self.poolingb4(x)

          x = self.relu51(self.bn51(self.conv51(x)))
          x = self.relu52(self.bn52(self.conv52(x)))
          x = self.relu53(self.bn53(self.conv53(x)))
          x = self.poolingb5(x)
          x = x.view(x.size(0), -1)  # Flatten

          x = F.relu(self.denselayer1(x))
          x = F.relu(self.denselayer2(x))
          x = self.denselayer3(x)

          return x

        elif(x.dim()==3):
            x = self.relu11_aud(self.bn11_aud(self.conv11_aud(x)))
            x = self.relu12_aud(self.bn12_aud(self.conv12_aud(x)))
            x = self.poolingb1_aud(x)

            x = self.relu21_aud(self.bn21_aud(self.conv21_aud(x)))
            x = self.relu22_aud(self.bn22_aud(self.conv22_aud(x)))
            x = self.poolingb2_aud(x)

            x = self.relu31_aud(self.bn31_aud(self.conv31_aud(x)))
            x = self.relu32_aud(self.bn32_aud(self.conv32_aud(x)))
            x = self.relu33_aud(self.bn33_aud(self.conv33_aud(x)))
            x = self.poolingb3_aud(x)

            x = self.relu41_aud(self.bn41_aud(self.conv41_aud(x)))
            x = self.relu42_aud(self.bn42_aud(self.conv42_aud(x)))
            x = self.relu43_aud(self.bn43_aud(self.conv43_aud(x)))
            x = self.poolingb4_aud(x)

            x = self.relu51_aud(self.bn51_aud(self.conv51_aud(x)))
            x = self.relu52_aud(self.bn52_aud(self.conv52_aud(x)))
            x = self.relu53_aud(self.bn53_aud(self.conv53_aud(x)))
            x = self.poolingb5_aud(x)
     
            x = x.view(x.size(0), -1)  # Flatten

            x = F.relu(self.denselayer1_aud(x))
            x = F.relu(self.denselayer2_aud(x))
            x = self.denselayer3_aud(x)

            return x
        
class Inception_Q3(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # block 1
        self.block1 = self.modifiedInceptionBlock(3,64,64,128,64,61) # total channel after this 64+64+128 = 256

        # block 2
        self.block2 = self.modifiedInceptionBlock(256,64,64,128,64,64) # 512
        # block 3
        self.block3 = self.modifiedInceptionBlock(512,64,64,128,64,64)# 768
        # block 4
        self.block4 = self.modifiedInceptionBlock(768,64,64,128,64,64) # 1024
        # dense layer
        self.d1 = nn.Linear(2*2*1024,512)
        self.bn_d1 = nn.BatchNorm1d(512)  # BatchNorm added
        self.relud1 = nn.ReLU(inplace=True)
        self.d2 = nn.Linear(512,10)
        self.bn_d2 = nn.BatchNorm1d(10)  # BatchNorm added
        self.relud2 = nn.ReLU(inplace=True)


        # block 1 audio
        self.block1_aud = self.modifiedInceptionBlock_aud(128, 64, 64, 128, 64, 61)  # total channel after this 64+64+128 = 256

        # block 2 audio 
        self.block2_aud = self.modifiedInceptionBlock_aud(381, 64, 64, 128, 64, 64)  # 512
        # block 3 audio
        self.block3_aud = self.modifiedInceptionBlock_aud(637, 64, 64, 128, 64, 64)  # 768
        # block 4 audio
        self.block4_aud = self.modifiedInceptionBlock_aud(893, 64, 64, 128, 64, 64)  # 1024
        # dense layer
        self.d1_aud = nn.Linear(4596, 512)
        self.bn_d1_aud = nn.BatchNorm1d(512)  # BatchNorm added
        self.relud1_aud = nn.ReLU(inplace=True)
        self.d2_aud = nn.Linear(512, 35)
        self.bn_d2_aud = nn.BatchNorm1d(35)  # BatchNorm added
        self.relud2_aud = nn.ReLU(inplace=True)

    def forward(self, x):
      if(x.dim()==4):
        x = self.block2(self.block1(x))
        x = self.block4(self.block3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relud2(self.bn_d2(self.d2(self.relud1(self.bn_d1(self.d1(x))))))
        return x

      elif(x.dim()==3):
        x = self.block2_aud(self.block1_aud(x))
        x = self.block4_aud(self.block3_aud(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relud2_aud(self.bn_d2_aud(self.d2_aud(self.relud1_aud(self.bn_d1_aud(self.d1_aud(x))))))

        return x


    class modifiedInceptionBlock_aud(nn.Module):
      def __init__(self, in_channels, ch3x3_1, ch3x3_2, ch5x5_1, ch5x5_2, ch1x1):
          super().__init__()
          # level1
          self.conv3x3_1 = nn.Conv1d(in_channels, ch3x3_1, kernel_size=3, stride=2, padding=1)
          self.bn3x3_1 = nn.BatchNorm1d(ch3x3_1)
          self.relu3x3_1 = nn.ReLU(inplace=True)

          self.conv3x3_2 = nn.Conv1d(in_channels, ch3x3_2, kernel_size=3, stride=2, padding=1)
          self.bn3x3_2 = nn.BatchNorm1d(ch3x3_2)
          self.relu3x3_2 = nn.ReLU(inplace=True)

          # level2
          self.conv1x1 = nn.Conv1d(in_channels, ch1x1, stride=2, kernel_size=1)
          self.bn1x1 = nn.BatchNorm1d(ch1x1)
          self.relu1x1 = nn.ReLU(inplace=True)

          self.conv5x5_1 = nn.Conv1d(ch3x3_1, ch5x5_1, kernel_size=5, padding=2)
          self.bn5x5_1 = nn.BatchNorm1d(ch5x5_1)
          self.relu5x5_1 = nn.ReLU(inplace=True)

          self.conv5x5_2 = nn.Conv1d(ch3x3_2, ch5x5_2, kernel_size=5, padding=2)
          self.bn5x5_2 = nn.BatchNorm1d(ch5x5_2)
          self.relu5x5_2 = nn.ReLU(inplace=True)

          self.pooling = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

      def forward(self, x):
          x1 = self.relu1x1(self.bn1x1(self.conv1x1(x)))

          z1 = self.relu3x3_1(self.bn3x3_1(self.conv3x3_1(x)))
          x2 = self.relu5x5_1(self.bn5x5_1(self.conv5x5_1(z1)))

          z2 = self.relu3x3_2(self.bn3x3_2(self.conv3x3_2(x)))
          x3 = self.relu5x5_2(self.bn5x5_2(self.conv5x5_2(z2)))

          x4 = self.pooling(x)

          # concatenate x1, x2, x3, x4
          output = [x1, x2, x3, x4]
          return torch.cat(output, 1)

    class modifiedInceptionBlock(nn.Module):
        def __init__(self,in_channels,ch3x3_1,ch3x3_2,ch5x5_1,ch5x5_2,ch1x1):
            super().__init__()
            #level1
            self.conv3x3_1 = nn.Conv2d(in_channels,ch3x3_1,kernel_size=3,stride =2,padding = 1)
            self.bn3x3_1  = nn.BatchNorm2d(ch3x3_1)
            self.relu3x3_1  = nn.ReLU(inplace= True)

            self.conv3x3_2 = nn.Conv2d(in_channels,ch3x3_2,kernel_size=3,stride =2,padding = 1)
            self.bn3x3_2  = nn.BatchNorm2d(ch3x3_2)
            self.relu3x3_2  = nn.ReLU(inplace= True)

            # level2
            self.conv1x1 = nn.Conv2d(in_channels,ch1x1,stride =2,kernel_size=1)
            self.bn1x1  = nn.BatchNorm2d(ch1x1)
            self.relu1x1  = nn.ReLU(inplace= True)


            self.conv5x5_1 = nn.Conv2d(ch3x3_1,ch5x5_1,kernel_size=5,padding = 2)
            self.bn5x5_1  = nn.BatchNorm2d(ch5x5_1)
            self.relu5x5_1  = nn.ReLU(inplace= True)

            self.conv5x5_2 = nn.Conv2d(ch3x3_2,ch5x5_2,kernel_size=5,padding = 2)
            self.bn5x5_2  = nn.BatchNorm2d(ch5x5_2)
            self.relu5x5_2  = nn.ReLU(inplace= True)

            self.pooling = nn.MaxPool2d(kernel_size=3,stride =2,padding = 1)

        def forward(self,x):
            x1 = self.relu1x1(self.bn1x1(self.conv1x1(x)))

            z1 = self.relu3x3_1(self.bn3x3_1(self.conv3x3_1(x)))
            x2 = self.relu5x5_1(self.bn5x5_1(self.conv5x5_1(z1)))

            z2 = self.relu3x3_2(self.bn3x3_2(self.conv3x3_2(x)))
            x3 = self.relu5x5_2(self.bn5x5_2(self.conv5x5_2(z2)))

            x4 = self.pooling(x)

            # concatenate x1, x2, x3, x4
            output = [x1,x2,x3,x4]
            return torch.cat(output, 1)
        
class CustomNetwork_Q4(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
       # Input Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual Blocks
        self.residual_block1 = self.ResidualBlock(64, 64)
        self.residual_block2 = self.ResidualBlock(64, 64)
        self.residual_block3 = self.ResidualBlock(320, 64)
        self.residual_block4 = self.ResidualBlock(320, 64)
        self.residual_block5 = self.ResidualBlock(320, 64)

        # Inception Blocks
        self.inception_block1 = self.modifiedInceptionBlock(64, 64, 64, 128, 64, 64) # 320
        self.inception_block2 = self.modifiedInceptionBlock(64, 64, 64, 128, 64, 64)
        self.inception_block3 = self.modifiedInceptionBlock(64, 64, 64, 128, 64, 64)
        self.inception_block4 = self.modifiedInceptionBlock(64, 64, 64, 128, 64, 64)

        # Linear Layer for Classification
        self.linear = nn.Linear(320, 10)
        self.relud = nn.ReLU()

        # Input Layer audio
        self.conv1_aud = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_aud = nn.BatchNorm1d(64)
        self.relu_aud = nn.ReLU(inplace=True)

        # Residual Blocks audio
        self.residual_block1_aud = self.ResidualBlock(64, 64,is_image=False)
        self.residual_block2_aud = self.ResidualBlock(64, 64,is_image=False)
        self.residual_block3_aud = self.ResidualBlock(320, 64,is_image=False)
        self.residual_block4_aud = self.ResidualBlock(320, 64,is_image=False)
        self.residual_block5_aud= self.ResidualBlock(320, 64,is_image=False)

        # Inception Blocks audio
        self.inception_block1_aud = self.modifiedInceptionBlock(64, 64, 64, 128, 64, 64,is_image=False) # 320
        self.inception_block2_aud = self.modifiedInceptionBlock(64, 64, 64, 128, 64, 64,is_image=False)
        self.inception_block3_aud = self.modifiedInceptionBlock(64, 64, 64, 128, 64, 64,is_image=False)
        self.inception_block4_aud = self.modifiedInceptionBlock(64, 64, 64, 128, 64, 64,is_image=False)

        # Linear Layer for Classification
        self.linear_aud1 = nn.Linear(80, 512)
        self.relud_aud1 = nn.ReLU()
        self.linear_aud2 = nn.Linear(512, 35)
        self.relud_aud2 = nn.ReLU()

    def forward(self, x):
      if(x.dim()==4):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.inception_block1(x)
        x = self.residual_block3(x)
        x = self.inception_block2(x)
        x = self.residual_block4(x)
        x = self.inception_block3(x)
        x = self.residual_block5(x)
        x = self.inception_block4(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.relud(self.linear(x))
        return x


      elif(x.dim()==3):
        x = self.relu_aud(self.bn1_aud(self.conv1_aud(x)))
        x = self.residual_block1_aud(x)
        x = self.residual_block2_aud(x)
        x = self.inception_block1_aud(x)
        x = self.residual_block3_aud(x)
        x = self.inception_block2_aud(x)
        x = self.residual_block4_aud(x)
        x = self.inception_block3_aud(x)
        x = self.residual_block5_aud(x)
        x = self.inception_block4_aud(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.relud_aud2(self.linear_aud2(self.relud_aud1(self.linear_aud1(x))))
        return x

    class ResidualBlock(nn.Module):
      expansion = 1

      def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_image=True):
          super(CustomNetwork_Q4.ResidualBlock, self).__init__()
          conv = nn.Conv2d if is_image else nn.Conv1d
          BatchNorm = nn.BatchNorm2d if is_image else nn.BatchNorm1d
          self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
          self.bn1 = BatchNorm(out_channels)
          self.relu = nn.ReLU(inplace=True)
          self.conv2 = conv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
          self.bn2 = BatchNorm(out_channels)
          self.downsample = nn.Sequential(
              conv(in_channels, out_channels, kernel_size=1, stride=stride),
              BatchNorm(out_channels),
          ) if stride != 1 or in_channels != out_channels else None

      def forward(self, x):
          identity = x
          out = self.conv1(x)
          out = self.bn1(out)
          out = self.relu(out)
          out = self.conv2(out)
          out = self.bn2(out)
          if self.downsample is not None:
              identity = self.downsample(x)
          if identity.shape != out.shape:
              identity = F.pad(identity, (0, out.shape[-1] - identity.shape[-1], 0, out.shape[-2] - identity.shape[-2]))
          out += identity
          out = self.relu(out)
          return out

    class modifiedInceptionBlock(nn.Module):
      def __init__(self, in_channels, ch3x3_1, ch3x3_2, ch5x5_1, ch5x5_2, ch1x1,is_image=True):
          super(CustomNetwork_Q4.modifiedInceptionBlock, self).__init__()
          Conv2d = nn.Conv2d if is_image else nn.Conv1d
          BatchNorm2d = nn.BatchNorm2d if is_image else nn.BatchNorm1d 
          MaxPool2d = nn.MaxPool2d if is_image else nn.MaxPool1d
          self.conv3x3_1 = Conv2d(in_channels, ch3x3_1, kernel_size=3, stride=2, padding=1)
          self.bn3x3_1 = BatchNorm2d(ch3x3_1)
          self.relu3x3_1 = nn.ReLU(inplace=True)

          self.conv3x3_2 = Conv2d(in_channels, ch3x3_2, kernel_size=3, stride=2, padding=1)
          self.bn3x3_2 = BatchNorm2d(ch3x3_2)
          self.relu3x3_2 = nn.ReLU(inplace=True)

          self.conv1x1 = Conv2d(in_channels, ch1x1, stride=2, kernel_size=1)
          self.bn1x1 = BatchNorm2d(ch1x1)
          self.relu1x1 = nn.ReLU(inplace=True)

          self.conv5x5_1 = Conv2d(ch3x3_1, ch5x5_1, kernel_size=5, padding=2)
          self.bn5x5_1 = BatchNorm2d(ch5x5_1)
          self.relu5x5_1 = nn.ReLU(inplace=True)

          self.conv5x5_2 = Conv2d(ch3x3_2, ch5x5_2, kernel_size=5, padding=2)
          self.bn5x5_2 = BatchNorm2d(ch5x5_2)
          self.relu5x5_2 = nn.ReLU(inplace=True)

          self.pooling = MaxPool2d(kernel_size=3, stride=2, padding=1)

      def forward(self, x):
          x1 = self.relu1x1(self.bn1x1(self.conv1x1(x)))

          z1 = self.relu3x3_1(self.bn3x3_1(self.conv3x3_1(x)))
          x2 = self.relu5x5_1(self.bn5x5_1(self.conv5x5_1(z1)))

          z2 = self.relu3x3_2(self.bn3x3_2(self.conv3x3_2(x)))
          x3 = self.relu5x5_2(self.bn5x5_2(self.conv5x5_2(z2)))

          x4 = self.pooling(x)

          output = [x1, x2, x3, x4]
          return torch.cat(output, 1)

def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)
    
    # Write your code here
    best_accuracy = 0.0
    for epoch in range(EPOCH):
        running_loss = 0.0
        correct = 0
        total = 0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # loss
        loss = running_loss / len(dataloader)
        # Calculate training accuracy
        accuracy = correct / total
        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            loss,
            accuracy
        ))
        # Save checkpoint if accuracy improved
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save(network.state_dict(), checkpoint_path)


def validator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)
      # Write your code here
    for epoch in range(EPOCH):
        running_loss = 0.0
        correct = 0
        total = 0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # loss
        loss =running_loss / len(dataloader)
        # Calculate training accuracy
        accuracy = correct / total
    print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy))


def evaluator(gpu="F",
                dataloader=None,
                network=None,
                criterion=None,
                optimizer=None):

    # Write your code here
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    network.load_state_dict(torch.load(checkpoint_path))
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)
    total_loss = 0.0
    correct = 0
    total = 0
    total_mse=0
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = network(data)
            # print("output.shape",outputs.shape)
            # Calculate loss
            labels_one_hot = F.one_hot(labels, num_classes=outputs.shape[1]).float()
            #print("labels_one_hot.shape",labels_one_hot.shape)
            mse = mse_loss(outputs,labels_one_hot)
            total_mse += mse.item()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_mse / len(dataloader)
    accuracy = correct / total

    print("[Loss: {}, Accuracy: {}]".format(
        avg_loss,
        accuracy
    ))
    
    
    