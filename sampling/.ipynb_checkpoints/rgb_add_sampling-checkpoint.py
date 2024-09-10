from masked_conv import *
from evaluation import *

def custom_gray(img):
  _, height, width = img.size()
  new_img = torch.empty(2, height, width*3)

  for i in range(height):
    for j in range(width):
      current_value = img[:,i,j]
      new_img[0,i,(j*3)] = current_value
      new_img[1,i,(j*3)] = 0
      new_img[0,i,(j*3 +1)] = current_value
      new_img[1,i,(j*3 +1)] = 1
      new_img[0,i,(j*3+ 2)] = current_value
      new_img[1,i,(j*3 +2)] = 2

  return new_img



def channel_concatination(img):
  _, height, width = img.size()
  new_img = torch.empty(1,height, width*3)

  for i in range(height):
    for j in range(width):
      red_channel = img[0,i,j]
      green_channel = img[1,i,j]
      blue_channel = img[2,i,j]


      new_img[:,i,(j*3)] = red_channel
      new_img[:,i,(j*3 + 1)] = green_channel
      new_img[:,i,(j*3 + 2)] = blue_channel

  return new_img





def reverse_to_original(img):
  _, _, height, _ = img.size()
  new_img = torch.empty(3,height,height)

  for i in range(height):
    for j in range(height):

      new_img[0,i,j] = img[:,:, i,(j*3)]
      new_img[1,i,j] = img[:,:, i,(j*3+1)]
      new_img[2,i,j] = img[:,:, i,(j*3+2)]

  return new_img

def reverse_to_original_gray(img):
  _, _, height, _ = img.size()
  new_img = torch.empty(1,height,height)

  for i in range(height):
    for j in range(height):

      new_img[0,i,j] = img[:,0, i,(j*3)]


  return new_img
        
class PairDataset(data.Dataset):
    def __init__(self, colour_dataset, gray_dataset):
        self.colour_dataset = colour_dataset
        self.gray_dataset = gray_dataset

    def __getitem__(self, index):
        # image, lable = dataset[index]
        colour_x, colour_y = self.colour_dataset[index]
        gray_x, gray_y = self.gray_dataset[index]

        return channel_concatination(colour_x), colour_y, custom_gray(gray_x), gray_y
    
    def __len__(self):
       return min(len(self.colour_dataset),len(self.gray_dataset))

   
            
            
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
            nn.Conv2d(channels_num*4, channels_num*8, 1)
        )

        self.gray = nn.Sequential(
          nn.Conv2d(2             , channels_num  , 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num),
          nn.Conv2d(channels_num  , channels_num  , 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num),
          nn.Conv2d(channels_num  , channels_num*2, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*2),
          nn.Conv2d(channels_num*2, channels_num*2, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*2),
          nn.Conv2d(channels_num*2, channels_num*4, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
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

          nn.Conv2d(combined_channels*2, 256, 1)
          )


    def forward(self,x, y):
        x = self.rgb(x)
        y = self.gray(y)
        
        concat = torch.cat((x,y), dim=1)
        
        concat = self.combined_cnn(concat)
        return concat
    
training_datasets_coloured  = datasets.CIFAR10('datasets',
                   train = False,
                   download=True,
                   transform= transforms.ToTensor())


training_datasets_gray  = datasets.CIFAR10('datasets',
                   train = False,
                   download=True,
                   transform= transforms.Compose([transforms.Grayscale(num_output_channels=1),
                   transforms.ToTensor() ])
                   )

pair_dataset = PairDataset(training_datasets_coloured,training_datasets_gray)



   
colourisation = Colourisation(32, 64).to(device)
print(colourisation)
colourisation.load_state_dict(torch.load("models/colourisation_cifar10.pth"))
colourisation.eval()

evaluation_images = data.DataLoader(pair_dataset, batch_size=1, shuffle=True) 

def sampling(gray_img):
  sample = torch.zeros(1,1,32,96)
  for i in range(32):
    for j in range(96):
        colorisation = colourisation(sample.to(device), gray_img.to(device))

        probs = F.softmax(colorisation[:,:,i,j], dim=1).detach()
        sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0


  return sample

loop = 0

for colour, _, gray,_ in evaluation_images:
  sample1 = sampling(gray)
  sample2 = sampling(gray)
  sample3 = sampling(gray)

  col = np.array( reverse_to_original(colour)).transpose(1,2,0)
  s1 = np.array(reverse_to_original(sample1)).transpose(1,2,0)
  s2 = np.array(reverse_to_original(sample2)).transpose(1,2,0)
  s3 = np.array(reverse_to_original(sample3)).transpose(1,2,0)
  plt.subplot(1, 5, 1)
  plt.title("original")
  plt.imshow( col)

  plt.text(15, 35, f'SSIM:{(ssim_(col, col))}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")


  plt.subplot(1, 5, 2)
  plt.title("gray")
  plt.imshow(reverse_to_original_gray(gray).numpy().squeeze(), cmap="gray")
  plt.axis("off")




  plt.subplot(1, 5, 3)
  plt.title("Smaple 1")
  plt.imshow(s1)

  plt.text(15, 35, f'SSIM:{np.round(float(ssim_(col, s1)), 2)}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")

  plt.subplot(1, 5, 4)
  plt.title("Sample 2")
  plt.imshow(s2)
  plt.text(15, 35, f'SSIM:{np.round(float(ssim_(col, s2)), 2)}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")

  plt.subplot(1, 5, 5)
  plt.title("Sample 3")
  plt.imshow(s3)
  plt.text(15, 35, f'SSIM:{np.round(float(ssim_(col, s3)),  2)}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
  plt.axis("off")

  plt.show()


  if loop == 10:
        break
  loop +=1
