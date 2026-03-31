
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
    
class AdapterWSpatial2(nn.Module):
    def __init__(
            self, 
            embed_dim:int = 768,  
            in_channels: int = 384,
            ch1:int=192,
            ch3:int = 96,
            ch5:int = 96,
            skip = False  
            ):
        super().__init__()

        self.skip  = skip
        self.conv1_2 = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=True)
        self.D_fc2 = nn.Linear(in_channels, embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=True)

        self.oxo = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=1, bias=True),
            # nn.BatchNorm2d(ch1, eps=0.001),
            LayerNorm2d(ch1),
            nn.ReLU(inplace=True)
        )

        self.txt = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=1, bias=True),
            nn.BatchNorm2d(24, eps=0.001),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, ch3, kernel_size=3,padding=1, bias=True),
            # nn.BatchNorm2d(ch3, eps=0.001),
            LayerNorm2d(ch3),
            nn.ReLU(inplace=True),
        )

        self.fxf = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=1, bias=True),
            # nn.BatchNorm2d(24, eps=0.001),
            LayerNorm2d(24),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, ch5, kernel_size=5,padding=2, bias=True),
            # nn.BatchNorm2d(ch5, eps=0.001),
            LayerNorm2d(ch5),
            nn.ReLU(inplace=True),
        )
        self.norm = LayerNorm2d(embed_dim)
        # Gate network: learns where to use which features
        # self.gate_conv = nn.Sequential(
        #     nn.Conv2d(in_channels * 2, in_channels // 4, 1),
        #     LayerNorm2d(in_channels //4),  # H, W = feature map size
        #     nn.GELU(),
        #     nn.Conv2d(in_channels // 4, in_channels, 1),
        #     nn.Sigmoid()  # Gate values in [0, 1]
        # )
        self.init_weights()
        # Residual weight
        #self.alpha = nn.Parameter(torch.tensor(0.1))

    def init_weights(self):
        for n, m in self.named_modules():

            for n2, m2 in m.named_modules():
                if 'D_fc2' in n2:
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0.)
                        nn.init.constant_(m2.bias, 0.)
            # for n2, m2 in m.named_modules():
            #     if 'oxo' in n2:
            #         if isinstance(m2, nn.Conv2d):
            #             nn.init.constant_(m2.weight, 0.00001)
            #             nn.init.constant_(m2.bias, 0.00001)
            #     if 'txt' in n2:
            #         if isinstance(m2, nn.Conv2d):
            #             nn.init.constant_(m2.weight, 0.00001)
            #             nn.init.constant_(m2.bias, 0.00001)
            #     if 'fxf' in n2:
            #         if isinstance(m2, nn.Conv2d):
            #             nn.init.constant_(m2.weight, 0.00001)
            #             nn.init.constant_(m2.bias, 0.00001)
            #     if 'conv1' in n2:
            #         if isinstance(m2, nn.Conv2d):
            #             nn.init.constant_(m2.weight, 0.00001)
            #             nn.init.constant_(m2.bias, 0.00001)

    def forward(self,x,y,patch_h,patch_w):
        
        # x_in = self.D_fc1(x)
        B,P,D1 = x.shape
        # x_in = F.relu(x_in, inplace=True)
        
        y_in = self.conv1(y)
        # y_in = F.relu(y_in, inplace=True)
        #reshape into image
        x = x[:,1:,:]
        x_img = x#[:,1:,:]
        x_img = x_img.reshape(B,patch_h,patch_w,D1).permute(0,3,1,2)
        x_img = self.conv1_2(self.norm(x_img))
        B, D2, _, _ = x_img.shape
        # # Concatenate for gate computation
        # combined = torch.cat([x_img, y_in], dim=1)
        
        # # Compute spatial gate: where to use ViT vs ConvNeXt
        # gate = self.gate_conv(combined)  # (B, C, H, W) in [0, 1]
        
        # # Gated fusion
        # fused = gate * x_img + (1 - gate) * y_in
        fused = x_img + y_in
        # perform convs
        one = self.oxo(fused)
        three = self.txt(fused)
        five = self.fxf(fused)
        # fuse
        outputs = [one, three, five]
        outputs = torch.cat(outputs,dim=1)
        # print(outputs.shape)
        outputs = y_in + x_img + outputs
        outputs = outputs.reshape(B,D2,patch_h * patch_w).permute(0,2,1)

        
        


        outputs += x_img.reshape(B,D2,patch_h * patch_w).permute(0,2,1)
        outputs = self.D_fc2(outputs)
        if self.skip:
            outputs+=x
        # NOTE: remember to normalize the out afterwards
        return outputs


class AdapterWSpatialHR(nn.Module):
    def __init__(
            self, 
            embed_dim:int = 768,  
            in_channels: int = 384,
            ch1:int=192,
            ch3:int = 96,
            ch5:int = 96,
            skip = False  
            ):
        super().__init__()

        self.skip  = skip

        self.conv1_1 = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=True)
        self.conv1_2 = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=True)
        self.conv_out = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=True)
        # Zero init convout
        nn.init.constant_(self.conv_out.weight, 0.)
        nn.init.constant_(self.conv_out.bias, 0.)
        self.oxo = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=1, bias=True),
            nn.BatchNorm2d(ch1, eps=0.001),
            nn.ReLU(inplace=True)
        )

        self.txt = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=1, bias=True),
            nn.BatchNorm2d(24, eps=0.001),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, ch3, kernel_size=3,padding=1, bias=True),
            nn.BatchNorm2d(ch3, eps=0.001),
            nn.ReLU(inplace=True),
        )

        self.fxf = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=1, bias=True),
            nn.BatchNorm2d(24, eps=0.001),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, ch5, kernel_size=5,padding=2, bias=True),
            nn.BatchNorm2d(ch5, eps=0.001),
            nn.ReLU(inplace=True),
        )

        # Gate network: learns where to use which features

        #self.init_weights()
        # Residual weight
        #self.alpha = nn.Parameter(torch.tensor(0.1))

    def init_weights(self):
        for n, m in self.named_modules():

            for n2, m2 in m.named_modules():
                if 'D_fc2' in n2:
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0.)
                        nn.init.constant_(m2.bias, 0.)
            for n2, m2 in m.named_modules():
                if 'oxo' in n2:
                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)
                if 'txt' in n2:
                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)
                if 'fxf' in n2:
                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)
                if 'conv1' in n2:
                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)

    def forward(self,x,y):
        # print(x.shape, y.shape, 'adapter')
        y = F.interpolate(y,(x.shape[-2:]), mode='bilinear', align_corners=False) 
        x_in = self.conv1_1(x)
  
        # x_in = F.relu(x_in, inplace=True)
        
        y_in = self.conv1_2(y)
        # y_in = F.relu(y_in, inplace=True

        # # Concatenate for gate computation
        # combined = torch.cat([x_img, y_in], dim=1)
        
        # # Compute spatial gate: where to use ViT vs ConvNeXt
        # gate = self.gate_conv(combined)  # (B, C, H, W) in [0, 1]
        
        # # Gated fusion
        # fused = gate * x_img + (1 - gate) * y_in
        fused = x_in + y_in
        # perform convs
        one = self.oxo(fused)
        three = self.txt(fused)
        five = self.fxf(fused)
        # fuse
        outputs = [one, three, five]
        outputs = torch.cat(outputs,dim=1)


        outputs += x_in
        outputs = self.conv_out(outputs)

        # NOTE: remember to normalize the out afterwards
        return outputs