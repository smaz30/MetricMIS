
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """
    Applies LayerNorm over the channel dimension of 4D image tensors (B, C, H, W)
    by permuting to (B, H, W, C), normalizing, then permuting back.
    """
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)       # -> (B, H, W, C)
        x = self.norm(x)               # LayerNorm over C
        x = x.permute(0, 3, 1, 2)      # -> (B, C, H, W)
        return x
    

class AdapterWSpatial(nn.Module):
    def __init__(
            self, 
            embed_dim:int = 768,  
            in_channels: int = 384,
            ch1:int=192,
            ch3:int = 96,
            ch5:int = 96,
            skip = False,
            zero_init = True  
            ):
        super().__init__()

        self.skip  = skip
        self.D_fc1 = nn.Linear(embed_dim, in_channels)
        self.D_fc2 = nn.Linear(in_channels, embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=True)

        self.oxo = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=1, bias=True),
            LayerNorm2d(ch1),
            nn.ReLU(inplace=True)
        )

        self.txt = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=1, bias=True),
            nn.BatchNorm2d(24, eps=0.001),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, ch3, kernel_size=3,padding=1, bias=True),
            LayerNorm2d(ch3),
            nn.ReLU(inplace=True),
        )

        self.fxf = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=1, bias=True),
            LayerNorm2d(24),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, ch5, kernel_size=5,padding=2, bias=True),
            LayerNorm2d(ch5),
            nn.ReLU(inplace=True),
        )

        if zero_init:
            self.init_weights()


    def init_weights(self):
        for n, m in self.named_modules():

            if 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0.)
                    nn.init.constant_(m.bias, 0.)
            else:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)


    def forward(self,x,y,patch_h,patch_w):
        x = F.dropout(x, p=0.2, training=self.training)
        y = F.dropout(y, p=0.2, training=self.training)
        x_in = self.D_fc1(x)
        B,P,D = x_in.shape

        
        y_in = self.conv1(y)
        #reshape into image
        x_img = x_in[:,1:,:]
        x_img = x_img.reshape(B,patch_h,patch_w,D).permute(0,3,1,2)

        fused = x_img + y_in
        # perform convs
        one = self.oxo(fused)
        three = self.txt(fused)
        five = self.fxf(fused)
        # fuse
        outputs = [one, three, five]
        outputs = torch.cat(outputs,dim=1)
        outputs = y_in + x_img + outputs
        outputs = outputs.reshape(B,D,patch_h * patch_w).permute(0,2,1)

        clstoken =  x_in[:,0:1,:]
        outputs = torch.cat([clstoken,outputs],dim=1)

        outputs = self.D_fc2(outputs)
        if self.skip:
            outputs+=x
        return outputs
    
