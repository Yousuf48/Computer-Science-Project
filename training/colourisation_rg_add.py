import time
from masked_conv import *


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
    _, h, w = rg_img.size()
    rgb = torch.empty(3, h, w // 2)

    rgb[0] = rg_img[:, :, 0::2]
    rgb[1] = rg_img[:, :, 1::2]
    rgb[2] = 3 * gray[:, :, 0::2] - (rgb[0] + rgb[1])

    return rgb


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


class DatasetCustom(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        #self.gray = transforms.Grayscale(1)
        
    def __getitem__(self, index):
        rgb_x, _ = self.dataset[index]
        gray_x = torch.mean(rgb_x, dim=0, keepdim=True)
 
        #gray_x = self.gray(rgb_x)

        rg, gray = img_extended(rgb_x, gray_x)

        return rg, gray

    def __len__(self):
        return len(self.dataset)


rg_data = DatasetCustom(datasets.LFWPeople("datasets",
                                           split='train',
                                           download=True,
                                           transform=transforms.Compose(
                                               [transforms.ToTensor(), transforms.Resize((64, 64), antialias=True)])))

training_data = data.DataLoader(rg_data, batch_size=64, shuffle=True, num_workers=4)

colouriastion_faces_rg = ColouriastionRG(32,64).to(device)

optimizer = optim.Adam(colouriastion_faces_rg.parameters(), lr=0.002)
epochs = 130


losses = []

total_time = time.time()
for i in range(epochs):
    print(i)
    time_ = time.time()
    losses1 = []
    colouriastion_faces_rg.train()
    for rg, gray in training_data:
        red_green, gray = rg.to(device), gray.to(device)
        inputs = colouriastion_faces_rg(red_green, gray)
        target = (red_green[:, 0] * 255).long()

        loss = F.cross_entropy(inputs, target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses1.append(loss.detach().item())
        
    losses.append(np.mean(losses1))
    print(f"loss epoch {i+1}: {np.mean(losses1)}, time: {(time.time()-time_)/60}")
    if i == 50:
        optimizer = optim.Adam(colouriastion_faces_rg.parameters(), lr=0.001)
        
torch.save(colouriastion_faces_rg.state_dict(), "models/rg_add.pth")
print(f"total time: {(time.time()-total_time)/3600}")
loss = np.array(losses)
np.save('loss_rg_add.npy', loss)