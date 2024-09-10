from masked_conv import *
from evaluation import *


def custom_gray(img):
  _, height, width = img.size()
  new_img = torch.empty(1, height, width*3)

  new_img[0,:,0::3] = img
  new_img[0,:,1::3] = img
  new_img[0,:,2::3] = img

  return new_img


def channel_concatination(img):
  _, height, width = img.size()
  new_img = torch.empty(1,height, width*3)
  new_img[:,:,0::3] = img[0]
  new_img[:,:,1::3] = img[1]
  new_img[:,:,2::3] = img[2]

  return new_img


def reverse_to_original(img):
  _, height, _ = img.size()

  new_img = torch.empty(3,height,height)
  new_img[0] = img[:,:,0::3]
  new_img[1] = img[:,:,1::3]
  new_img[2] = img[:,:,2::3]

  return new_img
        

            
class Colourisation(nn.Module):
    def __init__(self,channels_num, combined_channels, *args, **kwargs):
        super(Colourisation ,self).__init__(*args, **kwargs)

        self.rgb = nn.Sequential(
            MaskedConv2d('A', 1, channels_num, 3,1,1 ), nn.BatchNorm2d(channels_num),nn.ReLU(True),
            MaskedConv2d('B', channels_num  , channels_num  , 3 ,1,1 ), nn.BatchNorm2d(channels_num)  , nn.ReLU(True),
            MaskedConv2d('B', channels_num  , channels_num*2, 3 ,1,1 ), nn.BatchNorm2d(channels_num*2), nn.ReLU(True),
            MaskedConv2d('B', channels_num*2, channels_num*2, 3 ,1,1 ), nn.BatchNorm2d(channels_num*2), nn.ReLU(True),
            MaskedConv2d('B', channels_num*2, channels_num*4, 3 ,1,1 ), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3 ,1,1 ), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3 ,1,1 ), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3 ,1,1 ), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3 ,1,1 ), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3 ,1,1 ), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            nn.Conv2d(channels_num*4, channels_num*8, 1)
        )

        self.gray = nn.Sequential(
          nn.Conv2d(1             , channels_num  , 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num),
          nn.Conv2d(channels_num  , channels_num  , 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num),
          nn.Conv2d(channels_num  , channels_num*2, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*2),
          nn.Conv2d(channels_num*2, channels_num*2, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*2),
          nn.Conv2d(channels_num*2, channels_num*4, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
          nn.Conv2d(channels_num*4, channels_num*4, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
          nn.Conv2d(channels_num*4, channels_num*4, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
          nn.Conv2d(channels_num*4, channels_num*4, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
          nn.Conv2d(channels_num*4, channels_num*4, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
          nn.Conv2d(channels_num*4, channels_num*4, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
          nn.Conv2d(channels_num*4, 256, 1)
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
          nn.Conv2d(combined_channels*2, combined_channels*2, 1),
          nn.ReLU(True),
          nn.Conv2d(combined_channels*2, combined_channels*2, 1),
          nn.ReLU(True),
          nn.Conv2d(combined_channels*2, 256, 1)
          )


    def forward(self,x, y):
        x = self.rgb(x)
        y = self.gray(y)
        
        concat = torch.cat((x,y), dim=1)
        
        concat = self.combined_cnn(concat)
        return concat
class DatasetCustom(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        
    def __getitem__(self, index):
        rgb, _ = self.dataset[index]
        gray_transform = transforms.Grayscale(1)
        
        gray = gray_transform(rgb)

        return channel_concatination(rgb), custom_gray(gray)

    def __len__(self):
        return len(self.dataset)  
    
colourisation = Colourisation(32, 64).to(device)


colourisation.load_state_dict(torch.load("models/colourisation_rgb.pth"))
colourisation.eval()
rg_data = DatasetCustom(datasets.LFWPeople("datasets",
                                           split='test',
                                           download=True,
                                           transform=transforms.Compose(
                                               [transforms.ToTensor(), transforms.Resize((64, 64), antialias=True)])))
testing_data = data.DataLoader(rg_data, batch_size=1, num_workers = 2,shuffle=True)

def sampling(gray):
  clean_sample = torch.zeros(1,1,64,192)
  for i in range(64):
    for j in range(192):
      color = colourisation(clean_sample.to(device), gray.to(device))
      probs = F.softmax(color[:,:,i,j], dim=1).detach()
      clean_sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
  return clean_sample

pil = transforms.ToPILImage()

for j, (rgb, g ) in enumerate(testing_data):

  sample1 = sampling(g)
  sample2 = sampling(g)
  sample3 = sampling(g)

  col = np.array( reverse_to_original(rgb[0])).transpose(1,2,0)
  s1 = np.array(reverse_to_original(sample1[0])).transpose(1,2,0)
  s2 = np.array(reverse_to_original(sample2[0])).transpose(1,2,0)
  s3 = np.array(reverse_to_original(sample3[0])).transpose(1,2,0)

  plt.subplot(1, 5, 1)
  plt.title("original")
  plt.imshow( col)
  plt.text(25, 70, f'SSIM:{(ssim_(col, col))}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")


  plt.subplot(1,5, 2)
  plt.title("grey")
  plt.imshow(pil(g[0,0,:,0::3]),  cmap="gray")
  plt.axis("off")


  plt.subplot(1, 5, 3)
  plt.title("Smaple 1")
  plt.imshow(s1)

  plt.text(25, 70, f'SSIM:{np.round(float(ssim_(col, s1)), 2)}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")

  plt.subplot(1, 5, 4)
  plt.title("Sample 2")
  plt.imshow(s2)
  plt.text(25, 70, f'SSIM:{np.round(float(ssim_(col, s2)), 2)}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")

  plt.subplot(1, 5, 5)
  plt.title("Sample 3")
  plt.imshow(s3)
  plt.text(25, 70, f'SSIM:{np.round(float(ssim_(col, s3)),  2)}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")

  plt.show()


  if j == 10:
    break