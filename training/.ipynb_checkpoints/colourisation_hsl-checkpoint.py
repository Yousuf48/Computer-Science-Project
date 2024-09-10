from masked_conv import *


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
  lightness = torch.empty(1, h, w * 2)

  lightness[:,:,0::2] = l
  lightness[:,:,1::2] = l

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
          nn.Conv2d(1             , channels_num  , 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num),
          nn.Conv2d(channels_num  , channels_num  , 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num),
          nn.Conv2d(channels_num  , channels_num*2, 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num*2),
          nn.Conv2d(channels_num*2, channels_num*2, 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num*2),
          nn.Conv2d(channels_num*2, channels_num*4, 3, 1,1), nn.ReLU(True), nn.BatchNorm2d(channels_num*4),
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
                       split ='train',
                       download=True,
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((64,64),antialias=True)])))

training_data = data.DataLoader(hsl_data, batch_size=32,  shuffle=True, num_workers=4)

optimizer = optim.Adam(colouriastion_faces.parameters(), lr=0.002)

#Training
epochs = 100

total_loss = []

total_time = time.time()
for i in range(epochs):
    print(i)
    time_ = time.time()
    losses = []
    colouriastion_faces.train()
    for j, (h_s,l, _) in enumerate(training_data):
        hue_saturation, lightness = h_s.to(device), l.to(device)
        input = colouriastion_faces(hue_saturation, lightness)
        target = (hue_saturation[:,0] * 255).long()
        
        loss = F.cross_entropy(input, target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.detach().item())


    print(f"Epoch {i+1}, loss : {np.mean(losses)}, time: {(time.time() - time_)/60} ")
    total_loss.append(np.mean(losses))

    if i == 50:
        optimizer = optim.Adam(colouriastion_faces.parameters(), lr=0.001)
        
torch.save(colouriastion_faces.state_dict(), "models/colourisation_hsl.pth")
print(f"total time = {(time.time() - total_time)/3600}")
np.save("loss_hsl.npy", total_loss)

