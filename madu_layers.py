import torch
import torch.nn as nn
import utils
# torch.autograd.set_detect_anomaly(True)

class Conv3dC(nn.Module):
    '''
    input: [B, T, H, W] -> output: [B, T, H, W]
    '''

    def __init__(self, kernel, bias=True, gaussian_init=False, sigma=.5, in_channels=1, out_channels=1,
                 return_squeezed=False, groups=1):
        super(Conv3dC, self).__init__()
        pad0 = int((kernel[0] - 1) / 2)
        pad1 = int((kernel[1] - 1) / 2)
        self.conv = nn.Conv3d(in_channels, out_channels, (kernel[0], kernel[1], kernel[1]), (1, 1, 1),
                              (pad0, pad1, pad1), bias=bias, padding_mode='replicate', groups=groups)
        self.return_squeezed = return_squeezed
        if gaussian_init is True:
            g3d = utils.gaussian_3D(kernel, sigma=sigma, normalized=True)
            self.conv.weight.data[0, 0, ...] = torch.tensor(g3d)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        x = self.conv(x)
        if self.return_squeezed:
            x = x.squeeze(1)
        return x

class Madu_Layer(nn.Module):
    def __init__(self, kernel, coef_L_initializer,
                 coef_S_initializer, coef_S_side_initializer,
                 l1_l1, reweightedl1_l1, hidden_dim=32, l1_l2=False):
        super(Madu_Layer, self).__init__()

        self.convM = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel, bias=False)

        # For the L branch
        self.conv0 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, return_squeezed=True)

        # For the M branch
        self.conv1 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, bias=False)
        self.conv2 = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel, bias=False)
        self.conv3 = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel, bias=False)
        self.conv4 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, return_squeezed=True, bias=False)

        self.tau_l = nn.Parameter(torch.tensor(1.0))
        self.tau_l2 = nn.Parameter(torch.tensor(1.0))
        self.tau_l3 = nn.Parameter(torch.tensor(1.0))

        self.fft_th1 = nn.Parameter(torch.tensor(1.0))
        self.fft_th2 = nn.Parameter(torch.tensor(1.0))
        self.fft_th3 = nn.Parameter(torch.tensor(1.0))

        self.tau_s = nn.Parameter(torch.tensor(1.0))

        self.rho = nn.Parameter(torch.tensor(1.0))
        self.mask_scaling = nn.Parameter(torch.tensor(5.0))

        self.G_conv = Conv3dC((1,7,7), in_channels=hidden_dim, out_channels=hidden_dim, bias=False, gaussian_init=True, sigma=.5, return_squeezed=False, groups=hidden_dim)

        self.g = nn.Parameter(torch.tensor(torch.ones(hidden_dim)))
        self.coef_L = nn.Parameter(torch.tensor(coef_L_initializer))
        self.coef_S = nn.Parameter(torch.tensor(coef_S_initializer))
        self.coef_S_side = nn.Parameter(torch.tensor(coef_S_side_initializer))
        self.l1_l1 = l1_l1
        self.reweightedl1_l1 = reweightedl1_l1
        self.l1_l2 = l1_l2
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()


        w = 3
        self.weights_L4 = nn.Parameter(torch.ones(w) / w)
      
    def generate_mask_codes(self, M):
        return self.conv_M(M)

    def forward(self, data, mode):
        tau_l = nn.functional.softplus(self.tau_l)
        tau_s = nn.functional.softplus(self.tau_s)

        # data = (input, low-rank, mask, multiplier, U vector, V vector)
        X = data[0]
        L = data[1]
        M = data[2]


        B, T, H, W = X.shape
       
        # Conversion to feature map
        M = self.convM(M)
     
        # L branch
        background_mask = 1 - self.conv0(M)
        background_mask_squared = torch.square(background_mask)
        _L = L - tau_l * background_mask_squared * L
        _X = tau_l * background_mask_squared * X
        Ltmp = _L + _X

        if mode == 'mean':
            L = self.pixelwise_mean(Ltmp)
        elif mode == 'median':
            L = self.pixelwise_median(Ltmp)
        elif mode == 'dft':
            Ltmp = self.fft_threshold(Ltmp.view(B, T, H * W), self.fft_th1)
            L = Ltmp.view(B, T, H, W)
        elif mode == 'svd':
            Ltmp = self.svtC(Ltmp.view(B, T, H * W), self.coef_L)
            L = Ltmp.view(B, T, H, W)
        elif mode == 'Madu1':
            L_mean = self.pixelwise_mean(Ltmp)
            Ltmp_dft = self.fft_threshold(Ltmp.view(B, T, H * W), self.fft_th1)
            L_dft = Ltmp_dft.view(B, T, H, W)
            Ltmp_svd = self.svtC(Ltmp.view(B, T, H * W), self.coef_L)
            L_svd = Ltmp_svd.view(B, T, H, W)
            L = self.weights_L4[0] * L_mean + self.weights_L4[1] * L_dft + self.weights_L4[2] * L_svd
        elif mode == 'Madu2':
            L_median = self.pixelwise_median(Ltmp)
            Ltmp_dft = self.fft_threshold(Ltmp.view(B, T, H * W), self.fft_th1)
            L_dft = Ltmp_dft.view(B, T, H, W)
            Ltmp_svd = self.svtC(Ltmp.view(B, T, H * W), self.coef_L)
            L_svd = Ltmp_svd.view(B, T, H, W)   
            L = self.weights_L4[0] * L_median + self.weights_L4[1] * L_dft + self.weights_L4[2] * L_svd
        elif mode == 'Madu3':
            L_mean = self.pixelwise_mean(Ltmp)
            Ltmp_dft = self.fft_threshold(Ltmp.view(B, T, H * W), self.fft_th1)
            L_dft = Ltmp_dft.view(B, T, H, W)
            L_median = self.pixelwise_median(Ltmp)
            L = self.weights_L4[0] * L_mean + self.weights_L4[1] * L_median + self.weights_L4[2] * L_dft
        
        # M branch
        foreground = X - L
        foreground_squared = torch.square(foreground)
        _M = M
        _X = tau_s * self.conv2((1 - self.conv1(M)) * foreground_squared.unsqueeze(1))
        Mtmp = _M + _X

        Mtmp = self.soft_l1_reweighted(Mtmp, self.coef_S, self.g)

        Mtmp = self.conv4(Mtmp)
        
        M = nn.functional.sigmoid((Mtmp-.5)*self.mask_scaling)

        return (X, L, M), foreground_squared
    
  
    def svtC(self, x, th):
        U, S, V = torch.svd(x)
        S = torch.sign(S) * self.relu(torch.abs(S) - nn.functional.relu(th) * torch.abs(S[:, 0]).unsqueeze(1))
        S_diag = torch.diag_embed(S, dim1=-2, dim2=-1)
        return torch.matmul(torch.matmul(U, S_diag), V.transpose(2, 1))
    
    def pixelwise_median(self, X):
        return torch.median(X, dim=1).values
    
    def pixelwise_mean(self, X):
        return torch.mean(X, dim=1)

    def fft_threshold(self, x, fft_th):
        fft_x = torch.fft.fft2(x)
        shifted_fft = torch.fft.fftshift(fft_x)
        magnitude = torch.abs(shifted_fft)  
        mask = (magnitude > fft_th).float()  
        masked_fft = shifted_fft * mask
        x_thresholded = torch.fft.ifft2(torch.fft.ifftshift(masked_fft))
        return x_thresholded.real 

    def soft_l1_reweighted(self, z, th, g):
        B, C, T, H, W = z.shape
        z = z.view(B, C, T * H * W)
        th = th.unsqueeze(-1).unsqueeze(-1)
        out = torch.sign(z) * nn.functional.relu(torch.abs(z) - th * g.unsqueeze(0).unsqueeze(-1))
        return out.view(B, C, T, H, W)

    def freeze_weights(self):
        # Freeze the weights of the pretrained layers
        self.pretrained_weights_frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        # Unfreeze the weights of the pretrained layers
        self.pretrained_weights_frozen = False
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_combine_weights(self):
        self.weights_L4.requires_grad = True
