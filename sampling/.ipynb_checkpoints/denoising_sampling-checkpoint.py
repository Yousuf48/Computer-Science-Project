from masked_conv import *
from utils import *
from evaluation import *

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

clean_training_datasets_mnist  = datasets.MNIST('datasets',
                   train=False,
                   download=True,
                   transform= transforms.ToTensor()
                   )


noisy_training_datasets_mnist  = datasets.MNIST('datasets',
                   train=False,
                   download=True,
                   transform= noisy_transform
                   )

clean_training_datasets_fashion  = datasets.FashionMNIST('datasets',
                   train=True,
                   download=True,
                   transform= transforms.ToTensor()
                   )


noisy_training_datasets_fashion  = datasets.FashionMNIST('datasets',
                   train=True,
                   download=True,
                   transform= noisy_transform
                   )


combined_cnn = CombinedCNN(64,32,64).to(device)
combined_cnn.load_state_dict(torch.load("models/denosing_mnist.pth"))    

pair_datasets_mnist = PairDataset(clean_training_datasets_mnist, noisy_training_datasets_mnist)
pair_datasets_fashion = PairDataset(clean_training_datasets_fashion, noisy_training_datasets_fashion)


mnist = data.DataLoader(pair_datasets_mnist, batch_size=1,num_workers=4 , pin_memory=True,shuffle=True)
fashion = data.DataLoader(pair_datasets_fashion, batch_size=1,num_workers=4 , pin_memory=True,shuffle=True)

def sampling(noise):
  clean_sample = torch.zeros(1,1,28,28)
  for i in range(28):
    for j in range(28):
      denoising = combined_cnn(clean_sample.to(device), noise.to(device))
      probs = F.softmax(denoising[:,:,i,j], dim=1).detach()
      clean_sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
  
  return clean_sample


def shows(images):
    pil = transforms.ToPILImage()
    for j, (clean, _, noisy, _) in enumerate(images):
      sample1 = sampling(noisy)
      sample2 = sampling(noisy)
      sample3 = sampling(noisy)

    
      #("Noise with 0.1 standard deviation")
      #("Noise with 0.1 standard deviation")
      plt.subplot(1,5,1)
      plt.title("Original/Clean")
      plt.imshow(pil(clean[0]), cmap='gray')
      plt.text(15, 30, f'MSE: {psnr(clean[0], clean[0])}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
      plt.axis("off")

      plt.subplot(1,5,2)
      plt.title("noisy")
      plt.imshow(pil(noisy[0]), cmap='gray')
      plt.axis("off")

      plt.subplot(1,5,3)
      plt.title("Sample 1")
      plt.imshow(pil(sample1[0]), cmap='gray')
      plt.text(15, 30, f'PSNR: {np.round(psnr(clean[0], sample1[0]))}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
      plt.axis("off")

      plt.subplot(1,5,4)
      plt.title("Sample 2")
      plt.imshow(pil(sample2[0]), cmap='gray')
      plt.text(15, 30, f'PSNR: {np.round(psnr(clean[0], sample2[0]))}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
      plt.axis("off")

      plt.subplot(1,5,5)
      plt.title("Sample 3")
      plt.imshow(pil(sample3[0]), cmap='gray')
      plt.text(15, 30, f'PSNR: {np.round(psnr(clean[0], sample3[0]))}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
      plt.axis("off")
      plt.show()

      if j ==10:
        break

        
shows(mnist)
combined_cnn.load_state_dict(torch.load("models/denosing_fashion.pth")) 
combined_cnn.eval()
shows(fashion)