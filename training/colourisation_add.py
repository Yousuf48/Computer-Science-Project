from masked_conv import *


import datetime



def custom_gray(img):
  _, height, width = img.size()
  new_img = torch.empty(2, height, width*3)
  channel_rec = torch.tensor([0,1,2])
  channel_rec = channel_rec.repeat(height * width) 
  channel_rec = channel_rec.view(1, height, width*3)

  new_img[0,:,0::3] = img
  new_img[0,:,1::3] = img
  new_img[0,:,2::3] = img
  new_img[1] = channel_rec


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
    
    
colourisation = Colourisation(32, 64).to(device)
print(colourisation)



training_datasets_coloured  = datasets.CIFAR10('datasets',
                   train = True,
                   download=True,
                   transform= transforms.ToTensor())


training_datasets_gray  = datasets.CIFAR10('datasets',
                   train = True,
                   download=True,
                   transform= transforms.Compose([transforms.Grayscale(num_output_channels=1),
                   transforms.ToTensor() ])
                   )



optimizer = optim.Adam(colourisation.parameters(), lr=0.003)


pair_dataset = PairDataset(training_datasets_coloured,training_datasets_gray)


training_data = data.DataLoader(pair_dataset, batch_size=256, num_workers = 4, pin_memory=True,shuffle=True)

epoches = 100


time_tr = time.time()
total_loss = []

for i in range(epoches):
    losses = []
    print(f"epoch {i+1} at the time: {datetime.datetime.now()}")
    time_r = time.time()
    colourisation.train()

    for colored_img, _, gray_img, _ in training_data:
        colored_img, gray_img = colored_img.to(device),  gray_img.to(device)

        inputs = colourisation(colored_img, gray_img)
        target = (colored_img[:,0,:,:] * 255).long()

        loss = F.cross_entropy(inputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  
        
        epoch_time = time.time() - time_r
        
    print(f"time for each epoch_{i}:{epoch_time}s -> {epoch_time/60} min, loss{i+11}= {np.mean(losses)}")
    if i == 20:
        optimizer = optim.Adam(colourisation.parameters(), lr=0.002)
    if i == 60:
        optimizer = optim.Adam(colourisation.parameters(), lr=0.001)
 
torch.save(colourisation.state_dict(), "models/colourisation_cifar10.pth")
print(f"total time = {(time.time() - time_tr)/3600}")


np.save("add_rgb_loss.npy", total_loss)