import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torchviz import make_dot

class Temporal3DCNN(nn.Module):
    def __init__(self):
        super(Temporal3DCNN,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(1,8,kernel_size=(3,3,3), padding=(1,1,1)), # (B,1,5,H,W) --> (B,8,5,H,W)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,1,1),stride=(1,1,1)) # (B,8,5,H,W) --> (B,8,4,H,W)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(8,16,kernel_size=(3,3,3), padding=(1,1,1)), # (B,8,4,H,W) --> (B,16,4,H,W)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,1,1)), # (B,16,4,H,W) --> (B,16,2,H,W)

        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(16,32,kernel_size=(3,3,3), padding=(1,1,1)), # (B,16,2,H,W) --> (B,32,2,H,W)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,1,1)), # (B,32,2,H,W) --> (B,32,1,H,W)
        )

        self.projection = nn.Conv3d(32,1,kernel_size=1) # (B,32,1,H,W) --> (B,1,1,H,W)
    
    def forward(self,x):
        #print("Input shape:", x.shape)  # (B, T,C, H, W)
        x = x.permute(0,2,1,3,4)  # (B,T,H,W,C) --> (B,C,T,H,W) where C = 1 and T = 5
        #print("After permute to (B,C,T,H,W):", x.shape)
        x = self.layer1(x)
        #print("After layer1 (T=5→4):", x.shape)

        x = self.layer2(x)
        #print("After layer2 (T=4→2):", x.shape)

        x = self.layer3(x)
        #print("After layer3 (T=2→1):", x.shape)

        x = self.projection(x)
        #print("After final projection Conv3d (channels 32→1):", x.shape)

        x = x.squeeze(2)  ## eliminate temporal dimension --> (B,1,1,H,W) --> (B,1,H,W)
        #print("After squeeze (remove T=1):", x.shape)
        return x

'''
input_tensor = torch.randn(1,5,1080,1920,1)
model = Temporal3DCNN()
output = model(input_tensor)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")
'''

model = Temporal3DCNN()
model.to("cpu")
#input_model=torch.randn(1, 5, 1, 1080, 1020)
# Entrada amb forma (B, T, C, H, W) → en aquest model es permuta a (B, C, T, H, W)
summary(model,input_size=(1, 5, 1, 1080, 1920),device="cpu")

'''
output = model(input_model)
dot = make_dot(output, params=dict(model.named_parameters()))

# Desa'l com a imatge
dot.format = "png"
dot.render("temporal3dcnn_architecture")
'''