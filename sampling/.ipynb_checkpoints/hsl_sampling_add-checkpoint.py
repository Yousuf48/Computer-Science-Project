from masked_conv import *
from evaluation import *

def rgb_to_hsl(img):
  min_value, _ = torch.min(img, dim=0,  keepdim=True)
  max_value, _ = torch.max(img, dim=0, keepdim=True)

  # Calculate Lightness
  l = (min_value  +  max_value)/ 2
  # calculate Saturation

  sat =  torch.where(l <= 0.5, (max_value-min_value)/(max_value + min_value), ( max_value-min_value)/(2.0-max_value-min_value))
  # if max == min
  sat =  torch.where(min_value == max_value, torch.zeros_like(max_value - min_value), sat)

  # Hue calculation
  hue = torch.zeros_like(max_value)
  hue = torch.where(max_value == img[0], (img[1]-img[2])/(max_value), hue)
  hue = torch.where(max_value == img[1], 2.0 + (img[2]-img[0])/(max_value), hue)
  hue = torch.where(max_value == img[2], 4.0 + (img[0]-img[1])/(max_value), hue)
  hue /= 6.0
  hue = hue % 1


  return torch.cat((hue,sat,l),dim=0)


def hsl_to_rgb(img):

  hue, sat, l = img[:,0],img[:,1],img[:,2]

  temp_1 = torch.where(l<0.5, l*(1.0+sat), l + sat - l*sat)
  temp_2 = 2 * l - temp_1


  temp_r = hue + 1/3
  temp_g = hue
  temp_b = hue - 1/3

  def hue_convert(temp_1, temp_2, temp_c):

    temp_c = torch.where(temp_c < 0, temp_c + 1,
                         torch.where(temp_c > 1, temp_c - 1, temp_c))

    color = torch.where( temp_c < 1/6, temp_2 + (temp_1 - temp_2) * 6 * temp_c,
                        torch.where( temp_c < 1/2, temp_1,
                                    torch.where((temp_c < 2/3), temp_2 + (temp_1 - temp_2) * (2/3 - temp_c) * 6, temp_2)))

    return color

  r = hue_convert(temp_1,temp_2,temp_r)
  g = hue_convert(temp_1,temp_2,temp_g)
  b = hue_convert(temp_1,temp_2,temp_b)

  return torch.stack((r,g,b), dim=1)


def img_extend(img):
  _,h,w = img.size()
  l , h_s = img[2:], img[:2]
  hue_saturation = torch.empty(1,h,w*2)
  lightness = torch.empty(2, h, w * 2)

  lightness[0,:,0::2] = l
  lightness[0,:,1::2] = l
  lightness[1,:,0::2] = torch.zeros((h,w))
  lightness[1,:,1::2] = torch.ones((h,w))

  hue_saturation[0,:,0::2] = h_s[0]
  hue_saturation[0,:,1::2] = h_s[1]

  return hue_saturation, lightness

class RBG2HSL(data.Dataset):
    def __init__(self, rgb_dataset):
        self.rgb_dataset = rgb_dataset

    def __getitem__(self, index):
        rgb_x, label = self.rgb_dataset[index]

        hsl = rgb_to_hsl(rgb_x)
        hue_sat,l = img_extend(hsl)
        l = torch.clamp(torch.nan_to_num(l, nan=0.0), 0, 1)
        hue_sat = torch.clamp(torch.nan_to_num(hue_sat, nan=0.0), 0, 1)
        return hue_sat, l , label

    def __len__(self):
        return len(self.rgb_dataset)

def reverse(l, h_s):
  batch,_,H,_ = l.size()

  hsl = torch.empty(batch,3,H,H)

  for i in range(H):
    for j in range(H):
      hsl[:,0,i,j] = h_s[:,0,i,j*2]
      hsl[:,1,i,j] = h_s[:,0,i,j*2+1]
      hsl[:,2,i,j] = l[:,0,i,j*2]

  return hsl

class ColouriastionHsl(nn.Module):
  def __init__(self,channels_num, combined_channels, *args, **kwargs):
        super(ColouriastionHsl ,self).__init__(*args, **kwargs)

        self.hs = nn.Sequential(
            MaskedConv2d('A', 1, channels_num, 3,1,1 ), nn.BatchNorm2d(channels_num),nn.ReLU(True),
            MaskedConv2d('B', channels_num  , channels_num  , 3 ,1,1 ), nn.BatchNorm2d(channels_num)  , nn.ReLU(True),
            MaskedConv2d('B', channels_num  , channels_num*2, 3 ,1,1 ), nn.BatchNorm2d(channels_num*2), nn.ReLU(True),
            MaskedConv2d('B', channels_num*2, channels_num*2, 3 ,1,1 ), nn.BatchNorm2d(channels_num*2), nn.ReLU(True),
            MaskedConv2d('B', channels_num*2, channels_num*2, 3 ,1,1 ), nn.BatchNorm2d(channels_num*2), nn.ReLU(True),
            MaskedConv2d('B', channels_num*2, channels_num*4, 3 ,1,1 ), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3 ,1,1 ), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3 ,1,1 ), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            nn.Conv2d(channels_num*4, channels_num*8, 1)
        )

        self.l = nn.Sequential(
          nn.Conv2d(2             , channels_num  , 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num),
          nn.Conv2d(channels_num  , channels_num  , 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num),
          nn.Conv2d(channels_num  , channels_num*2, 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num*2),
          nn.Conv2d(channels_num*2, channels_num*2, 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num*2),
          nn.Conv2d(channels_num*2, channels_num*4, 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
          nn.Conv2d(channels_num*4, channels_num*4, 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
          nn.Conv2d(channels_num*4, channels_num*4, 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
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

          nn.Conv2d(combined_channels*2, 256, 1)
          )


  def forward(self,x, y):
      x = self.hs(x)
      y = self.l(y)

      concat = torch.cat((x,y), dim=1)
      concat = self.combined_cnn(concat)

      return concat
    

    
colouriastion_faces = ColouriastionHsl(32,64).to(device)


hsl_data =   RBG2HSL(datasets.LFWPeople("datasets",
                       split ='test',
                       download=True,
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((64,64),antialias=True)])))

testing_data = data.DataLoader(hsl_data, batch_size=1,  shuffle=True, num_workers=4)
colouriastion_faces.load_state_dict(torch.load("models/hsl_add.pth"))

colouriastion_faces.eval()


def samples(gray):
  sample = torch.zeros(1,1,64,128)
  for i in range(64):
    for j in range(128):
      color = colouriastion_faces(sample.to(device), gray.to(device))
      probs = F.softmax(color[:,:,i,j], dim=1).detach()
      sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
  return sample




pil = transforms.ToPILImage()
gray = transforms.Grayscale(1)
def shows():
    for j, (h, l, _) in enumerate(testing_data):

      sample_1 = reverse(l, samples(l))
      sample_2 = reverse(l, samples((l)))
      sample_3 = reverse(l, samples(l))

      col = np.array(hsl_to_rgb(reverse(l, h)[0])).transpose(1,2,0)
      s1 = np.array(hsl_to_rgb(sample_1[0])).transpose(1,2,0)
      s2 = np.array(hsl_to_rgb(sample_2[0])).transpose(1,2,0)
      s3 = np.array(hsl_to_rgb(sample_3[0])).transpose(1,2,0)
        
      plt.subplot(1, 5, 1)
      plt.title("original")
      plt.imshow(col)
      plt.text(25, 70, f'SSIM:{(ssim_(col, col))}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
      plt.axis("off")


 
      plt.subplot(1,5, 2)
      plt.title("gray")
      plt.imshow(pil(l[0,0,:,0::2]), cmap="gray")
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

print("HSL with the additional channel")
shows()
