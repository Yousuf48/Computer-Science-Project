from masked_conv import *
from evaluation import *
class ColouriastionRG(nn.Module):
    def __init__(self, channels_num, combined_channels, *args, **kwargs):
        super(ColouriastionRG, self).__init__(*args, **kwargs)

        self.rg = nn.Sequential(
            MaskedConv2d('A', 1, channels_num, 3, 1, 1), nn.BatchNorm2d(channels_num), nn.ReLU(True),
            MaskedConv2d('B', channels_num, channels_num, 3, 1, 1), nn.BatchNorm2d(channels_num), nn.ReLU(True),
            MaskedConv2d('B', channels_num, channels_num*2, 3, 1, 1), nn.BatchNorm2d(channels_num*2), nn.ReLU(True),
            MaskedConv2d('B', channels_num*2, channels_num*2, 3, 1, 1), nn.BatchNorm2d(channels_num*2), nn.ReLU(True),
            MaskedConv2d('B', channels_num*2, channels_num*2, 3, 1, 1), nn.BatchNorm2d(channels_num*2), nn.ReLU(True),
            MaskedConv2d('B', channels_num*2, channels_num*4, 3, 1, 1), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3, 1, 1), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3, 1, 1), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3, 1, 1), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            MaskedConv2d('B', channels_num*4, channels_num*4, 3, 1, 1), nn.BatchNorm2d(channels_num*4), nn.ReLU(True),
            nn.Conv2d(channels_num*4, 256, 1)
        )

        self.gray = nn.Sequential(
            nn.Conv2d(2, channels_num, 3, 1, 1), nn.BatchNorm2d(channels_num), nn.ReLU(True),
            nn.Conv2d(channels_num, channels_num, 3, 1, 1),   nn.BatchNorm2d(channels_num),nn.ReLU(True),
            nn.Conv2d(channels_num, channels_num * 2, 3, 1, 1),  nn.BatchNorm2d(channels_num*2),nn.ReLU(True),
            nn.Conv2d(channels_num * 2, channels_num * 2, 3, 1, 1),  nn.BatchNorm2d(channels_num*2),nn.ReLU(True),
            nn.Conv2d(channels_num * 2, channels_num * 4, 3, 1, 1),  nn.BatchNorm2d(channels_num*4),nn.ReLU(True),
            nn.Conv2d(channels_num * 4, channels_num * 4, 3, 1, 1),  nn.BatchNorm2d(channels_num*4),nn.ReLU(True),
            nn.Conv2d(channels_num * 4, channels_num * 4, 3, 1, 1),  nn.BatchNorm2d(channels_num*4),nn.ReLU(True),
            nn.Conv2d(channels_num * 4, channels_num * 4, 3, 1, 1),  nn.BatchNorm2d(channels_num*4),nn.ReLU(True),
            nn.Conv2d(channels_num * 4, 256, 1)
        )

        self.combined_cnn = nn.Sequential(
            nn.Conv2d(512, combined_channels, 1),
            nn.ReLU(True),

            nn.Conv2d(combined_channels, combined_channels, 1),
            nn.ReLU(True),

            nn.Conv2d(combined_channels, combined_channels * 2, 1),
            nn.ReLU(True),

            nn.Conv2d(combined_channels * 2, combined_channels * 2, 1),
            nn.ReLU(True),

            nn.Conv2d(combined_channels * 2, combined_channels * 2, 1),
            nn.ReLU(True),

            nn.Conv2d(combined_channels * 2, combined_channels * 2, 1),
            nn.ReLU(True),

            nn.Conv2d(combined_channels * 2, 256, 1)
        )

    def forward(self, x, y):
        x = self.rg(x)
        y = self.gray(y)

        concat = torch.cat((x, y), dim=1)
        concat = self.combined_cnn(concat)

        return concat
    

    
def img_extended(rgb, gray):
    _, h, w = rgb.size()

    gray_extended = torch.empty(2, h, w * 2)
    rg_img = torch.empty(1, h, w * 2)

    rg_img[:, :, 0::2] = rgb[0]
    rg_img[:, :, 1::2] = rgb[1]

    gray_extended[0, :, 0::2] = gray
    gray_extended[0, :, 1::2] = gray
    gray_extended[1,:,0::2] = torch.zeros((h,w))
    gray_extended[1,:,1::2] = torch.ones((h,w))
    return rg_img, gray_extended




def reverse_original(gray, rg_img):
    b,_, h, w = rg_img.size()
    rgb = torch.empty(b,3, h, w // 2)

    rgb[:,0] = rg_img[:,:, :, 0::2]
    rgb[:,1] = rg_img[:,:, :, 1::2]
    rgb[:,2] = 3 * gray[:,0, :, 0::2] - (rgb[:,0] + rgb[:,1])


    return rgb

class DatasetCustom(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        rgb_x, _ = self.dataset[index]

        gray_x = torch.mean(rgb_x, dim=0, keepdim=True)

        rg, gray = img_extended(rgb_x, gray_x)

        return rg, gray

    def __len__(self):
        return len(self.dataset)


rg_data = DatasetCustom(datasets.LFWPeople("datasets",
                                           split='test',
                                           download=True,
                                           transform=transforms.Compose(
                                               [transforms.ToTensor(), transforms.Resize((64, 64), antialias=True)])))

testing_data = data.DataLoader(rg_data, batch_size=1, shuffle=True, num_workers=2)

colouriastion_faces = ColouriastionRG(32,64).to(device)
colouriastion_faces.load_state_dict(torch.load("models/rg_add.pth"))
colouriastion_faces.eval()

def sample(gray):
  generated_sample = torch.zeros(1,1,64,128)
  for i in range(64):
    for j in range(128):
      color = colouriastion_faces(generated_sample.to(device), gray.to(device))
      probs = F.softmax(color[:,:,i,j], dim=1).detach()
      generated_sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
  return generated_sample



pil = transforms.ToPILImage()
gray = transforms.Grayscale(1)

def shows():
    for j, (rg, g ) in enumerate(testing_data):


      sample_1 = reverse_original(g, sample(g))
      sample_2 = reverse_original(g, sample(g))
      sample_3 = reverse_original(g, sample(g))
    
      col = np.array(reverse_original(g,rg)[0]).transpose(1,2,0)
      s1 = np.array(sample_1[0]).transpose(1,2,0)
      s2 = np.array(sample_2[0]).transpose(1,2,0)
      s3 = np.array(sample_3[0]).transpose(1,2,0)
      plt.subplot(1, 5, 1)
      plt.title("original")
      plt.imshow( col)

      plt.text(25, 70, f'SSIM:{(ssim_(col, col))}',  verticalalignment='center', horizontalalignment='center', fontsize=10)
      plt.axis("off")


      plt.subplot(1,5, 2)
      plt.title("gray")
      plt.imshow(pil(g[0,0,:,0::2]),  cmap="gray")
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

      if j >= 10:
        break



print("RG with the additional channel")
shows()