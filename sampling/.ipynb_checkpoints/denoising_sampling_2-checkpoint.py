from masked_conv import *
from utils import *
from evaluation import *

class PairDataset(data.Dataset):
    def __init__(self, clean_dataset, noisy_dataset):
        self.clean_dataset = clean_dataset
        self.noisy_dataset = noisy_dataset

    def __getitem__(self, index):
        # image, lable = dataset[index]
        clean_x, clean_y = self.clean_dataset[index]
        noisy_x, noisy_y = self.noisy_dataset[index]

        return clean_x, clean_y, noisy_x, noisy_y

    def __len__(self):
       return min(len(self.clean_dataset),len(self.noisy_dataset))

    def show_pairs(self, indices):
        for index in indices:
            clean_x, clean_y, noisy_x, noisy_y = self.__getitem__(index)

            # Convert PyTorch tensors to NumPy arrays for visualization
            clean_np = clean_x.numpy().squeeze()
            noisy_np = noisy_x.numpy().squeeze()

            # Display clean and noisy images side by side
            plt.figure(figsize=(4, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(clean_np, cmap='gray')

            plt.title(f"Clean images: {clean_y}")
            plt.subplot(1, 2, 2)
            plt.imshow(noisy_np, cmap='gray')
            plt.title(f"Noisy images : {noisy_y}")
            plt.show()

# A class combines both PixelCNN and NoisyCNN
class CombinedCNN(nn.Module):
    def __init__(self, pixel_cnn_channels, noisy_channels,combined_channels,*args, **kwargs):
      super(CombinedCNN, self).__init__(*args, **kwargs)

      self.pixel_cnn = nn.Sequential(
          MaskedConv2d('A', 1,  pixel_cnn_channels, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels, pixel_cnn_channels, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels, pixel_cnn_channels, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels, pixel_cnn_channels, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels, pixel_cnn_channels *2, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels*2), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels*2, pixel_cnn_channels *2, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels*2), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels*2, pixel_cnn_channels *2, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels*2), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels*2, pixel_cnn_channels *2, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels*2), nn.ReLU(True),
          nn.Conv2d(pixel_cnn_channels*2, 256, 1)
      )

      self.noisy_cnn = nn.Sequential(
          nn.Conv2d(1, noisy_channels, 3, padding=1),
          nn.ReLU(True),
          nn.Conv2d(noisy_channels, noisy_channels*2, 3,padding=1),
          nn.ReLU(True),
          nn.Conv2d(noisy_channels*2, noisy_channels*4, 3, padding=1),
          nn.ReLU(True),
          nn.Conv2d(noisy_channels*4, noisy_channels*4, 3, padding=1),
          nn.ReLU(True),
          nn.Conv2d(noisy_channels*4, 256, 1)
          )

      self.combined_cnn = nn.Sequential(
          nn.Conv2d(512,combined_channels, 1),
          nn.ReLU(True),

          nn.Conv2d(combined_channels, combined_channels, 1),
          nn.ReLU(True),

          nn.Conv2d(combined_channels, combined_channels*2, 1),
          nn.ReLU(True),

          nn.Conv2d(combined_channels*2, combined_channels*2, 1),
          nn.ReLU(True),

          nn.Conv2d(combined_channels*2, combined_channels*2, 1),
          nn.ReLU(True),

          nn.Conv2d(combined_channels*2, 256, 1)
          )

    def forward(self, x, y):
        x = self.pixel_cnn(x)
        y = self.noisy_cnn(y)
        c = torch.cat((x,y), dim=1)

        c = self.combined_cnn(c)

        return c

combined_cnn = CombinedCNN(64,32,64).to(device)



clean_test_datasets  = datasets.FashionMNIST('datasets',
                   train=False,
                   download=True,
                   transform= transforms.ToTensor()
                   )


noisy_test_datasets  = datasets.FashionMNIST('datasets',
                   train=False,
                   download=True,
                   transform= noisy_transform_more
                   )

combined_cnn.load_state_dict(torch.load("models/denosing_more_level.pth"))
combined_cnn.eval()
pair_dataset = PairDataset(clean_test_datasets, noisy_test_datasets )
images_pairs = data.DataLoader(pair_dataset, batch_size=1, shuffle=True)






def sample(noise):
  clean_sample = torch.zeros(1,1,28,28)
  for i in range(28):
    for j in range(28):
      denoising = combined_cnn(clean_sample.to(device), noise.to(device))
      probs = F.softmax(denoising[:,:,i,j], dim=1).detach()
      clean_sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
  return clean_sample


loop = 0
pil = transforms.ToPILImage()
for clean, l1, noise, l2 in images_pairs:
  sample_1 = sample(noise)
  sample_2 = sample(noise)
  sample_3 = sample(noise)


  #("Noise with 0.1 standard deviation")
  plt.subplot(1,5,1)
  plt.title("Original/Clean")
  plt.imshow(pil(clean[0]), cmap='gray')
  plt.text(15, 30, f'MSE: {psnr(clean[0], clean[0])}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")

  plt.subplot(1,5,2)
  plt.title("noisy")
  plt.imshow(pil(noise[0]), cmap='gray')
  plt.axis("off")

  plt.subplot(1,5,3)
  plt.title("Sample 1")
  plt.imshow(pil(sample_1[0]), cmap='gray')
  plt.text(15, 30, f'PSNR: {np.round(psnr(clean[0], sample_1[0]))}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")

  plt.subplot(1,5,4)
  plt.title("Sample 2")
  plt.imshow(pil(sample_2[0]), cmap='gray')
  plt.text(15, 30, f'PSNR: {np.round(psnr(clean[0], sample_2[0]))}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")

  plt.subplot(1,5,5)
  plt.title("Sample 3")
  plt.imshow(pil(sample_3[0]), cmap='gray')
  plt.text(15, 30, f'PSNR: {np.round(psnr(clean[0], sample_3[0]))}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")

  plt.show()

  if loop == 10:
    break
    
  loop += 1