import torch
import torch.nn as nn
from torch.autograd import Variable

import utils


class Conv3dC(nn.Module):
    '''
    input: [B, T, H, W] -> output: [B, T, H, W]
    '''

    def __init__(self, kernel, bias=True, gaussian_init=False, sigma=.5, in_channels=1, out_channels=1, return_squeezed=False, groups=1):
        super(Conv3dC, self).__init__()
        pad0 = int((kernel[0] - 1) / 2)
        pad1 = int((kernel[1] - 1) / 2)
        self.conv = nn.Conv3d(in_channels, out_channels, (kernel[0], kernel[1], kernel[1]), (1, 1, 1), (pad0, pad1, pad1), bias=bias, padding_mode='replicate', groups=groups)
        self.return_squeezed = return_squeezed
        if gaussian_init is True:
            g3d = utils.gaussian_3D(kernel, sigma=sigma, normalized=True)
            self.conv.weight.data[0,0,...] = torch.tensor(g3d)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        x = self.conv(x)
        if self.return_squeezed:
            x = x.squeeze(1)
        return x


class ROMAN_R_Layer(nn.Module):
    def __init__(self, kernel, coef_L_initializer,
                 coef_S_initializer, coef_S_side_initializer,
                 l1_l1, reweightedl1_l1, hidden_dim=32, l1_l2=False):
        super(ROMAN_R_Layer, self).__init__()

        self.convM = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel, bias=False)

        # For the L branch
        self.conv0 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, return_squeezed=True)

        # For the M branch
        self.conv1 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, bias=False)
        self.conv2 = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel, bias=False)
        self.conv3 = Conv3dC(in_channels=1, out_channels=hidden_dim, kernel=kernel, bias=False)
        self.conv4 = Conv3dC(in_channels=hidden_dim, out_channels=1, kernel=kernel, return_squeezed=True, bias=False)

        self.tau_l = nn.Parameter(torch.tensor(1.0))
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

    def generate_mask_codes(self, M):
        return self.conv_M(M)

    def forward(self, data):

        tau_l = nn.functional.softplus(self.tau_l)
        tau_s = nn.functional.softplus(self.tau_s)
        rho = nn.functional.softplus(self.rho)


        # data = (input, low-rank, mask, multiplier)
        X = data[0]
        L = data[1]
        M = data[2]
        U = data[3]

        B, T, H, W = X.shape

        # Conversion to feature map
        M = self.convM(M)

        # L branch

        background_mask = 1 - self.conv0(M)
        background_mask_squared = torch.square(background_mask)
        _L = L - tau_l * background_mask_squared * L
        _X = tau_l * background_mask_squared * X
        _U = -tau_l/(rho+1e-4) * background_mask * U
        Ltmp = _L + _X + _U

        Ltmp = self.svtC(Ltmp.view(B, T, H * W), self.coef_L)
        L = Ltmp.view(B, T, H, W)

        # M branch
        foreground = X - L
        foreground_squared = torch.square(foreground)
        _M = M
        _X = tau_s * self.conv2((1 - self.conv1(M)) * foreground_squared.unsqueeze(1))
        _U = - tau_s/(rho + 1e-4) * self.conv3(foreground * U)
        Mtmp = _M + _X + _U


        if self.l1_l2:
            M_zero = Mtmp[:, :, 0, ...].unsqueeze(2)
            M_side = torch.cat((M_zero, Mtmp[:, :, :-1, ...]), dim=2)
            M_side_motion = self.G_conv(M_side)
            if len(M_side_motion.shape) == 3: M_side_motion = M_side_motion.unsqueeze(0)

            Mtmp = Mtmp + self.coef_S_side * (M_side_motion - _M)
            if self.reweightedl1_l1:
                Mtmp = self.soft_l1_reweighted(Mtmp, self.coef_S, self.g)
            else:
                Mtmp = self.soft_l1(Mtmp, self.coef_S)

        elif self.l1_l1:
            M_zero = Mtmp[:, :, 0, ...].unsqueeze(2)
            M_side = torch.cat((M_zero, Mtmp[:, :, :-1, ...]), dim=2)
            M_side_motion = self.G_conv(M_side)
            if len(M_side_motion.shape) == 3: M_side_motion = M_side_motion.unsqueeze(0)
            Mtmp = self.soft_l1_l1(Mtmp, self.coef_S,
                                              self.coef_S_side, M_side_motion)

        elif self.reweightedl1_l1:
            M_zero = Mtmp[:,:,0,...].unsqueeze(2)
            M_side = torch.cat((M_zero, Mtmp[:, :, :-1, ...]), dim=2)
            M_side_motion = self.G_conv(M_side)
            if len(M_side_motion.shape) == 3: M_side_motion = M_side_motion.unsqueeze(0)
            Mtmp = self.soft_l1_l1_reweighted(Mtmp, self.coef_S,
                                           self.coef_S_side, M_side_motion, None, self.g)

        else:
            Mtmp = self.soft_l1_reweighted(Mtmp, self.coef_S, self.g)

        Mtmp = self.conv4(Mtmp)
        M = nn.functional.sigmoid((Mtmp-.5)*self.mask_scaling)

        # U branch
        U = U + rho*(1 - M) * (L - X)

        return (X, L, M, U)

    def svtC(self, x, th):
        U, S, V = torch.svd(x)
        S = torch.sign(S) * self.relu(torch.abs(S) - nn.functional.relu(th) * torch.abs(S[:, 0]).unsqueeze(1))
        S_diag = torch.diag_embed(S, dim1=-2, dim2=-1)
        return torch.matmul(torch.matmul(U, S_diag), V.transpose(2, 1))

    def mixthre(self, x, th):
        x_norm = torch.norm(x, p=2, dim=1)
        return self.relu(1 - th / x_norm)[:, None] * x

    def soft_l1(self, z, th):
        B, C, T, H, W = z.shape
        z = z.view(B, C, T * H * W)
        th = th.unsqueeze(-1).unsqueeze(-1)
        out = torch.sign(z) * nn.functional.relu(torch.abs(z) - th)
        return out.view(B, C, T, H, W)

    def soft_l1_reweighted_referenced(self, z, th, ref, g):
        B, C, T, H, W = z.shape
        z = z.view(B, C, T * H * W)
        ref = ref.view(B, C, T*H*W)
        x = z - ref
        th = th.unsqueeze(-1).unsqueeze(-1)
        out = torch.sign(x) * nn.functional.relu(torch.abs(x) - th*g.unsqueeze(0).unsqueeze(-1)) + ref
        return out.view(B, C, T, H, W)

    def soft_l1_reweighted(self, z, th, g):
        B, C, T, H, W = z.shape
        z = z.view(B, C, T * H * W)
        th = th.unsqueeze(-1).unsqueeze(-1)
        out = torch.sign(z) * nn.functional.relu(torch.abs(z) - th * g.unsqueeze(0).unsqueeze(-1))
        return out.view(B, C, T, H, W)

    def soft_l1_l1(self, z, w0, w1, alpha1):

        B, C, T, H, W = z.shape
        z = z.view(B, C, T * H * W)
        alpha1 = alpha1.view(B, C, T * H * W)

        alpha0 = torch.zeros(alpha1.size(), device=z.device, dtype=z.dtype)
        condition = alpha0 <= alpha1
        alpha0_sorted = torch.where(condition, alpha0, alpha1)
        alpha1_sorted = torch.where(condition, alpha1, alpha0)

        w0 = w0.unsqueeze(-1).unsqueeze(-1).repeat(1, C, T * H * W)
        w1 = w1.unsqueeze(-1).unsqueeze(-1).repeat(1, C, T * H * W)

        w0_sorted = torch.where(condition, w0, w1)
        w1_sorted = torch.where(condition, w1, w0)

        cond1 = z >= alpha1_sorted + w0_sorted + w1_sorted
        cond2 = z >= alpha1_sorted + w0_sorted - w1_sorted
        cond3 = z >= alpha0_sorted + w0_sorted - w1_sorted
        cond4 = z >= alpha0_sorted - w0_sorted - w1_sorted

        res1 = z - w0_sorted - w1_sorted
        res2 = alpha1_sorted
        res3 = z - w0_sorted + w1_sorted
        res4 = alpha0_sorted
        res5 = z + w0_sorted + w1_sorted
        return torch.where(cond1, res1,
                           torch.where(cond2, res2, torch.where(cond3, res3, torch.where(cond4, res4, res5)))).view(B, C, T, H, W)

    def soft_l1_l1_reweighted(self, x, w0, w1, alpha1, Z=None, g=None):

        B, C, T, H, W = x.shape
        x = x.view(B, C, T * H * W)
        alpha1 = alpha1.view(B, C, T * H * W)

        alpha0 = torch.zeros(alpha1.size(), device=x.device, dtype=x.dtype)
        condition = alpha0 <= alpha1
        alpha0_sorted = torch.where(condition, alpha0, alpha1)
        alpha1_sorted = torch.where(condition, alpha1, alpha0)

        w0 = w0.unsqueeze(-1).unsqueeze(-1).repeat(1, C, T*H*W)
        w1 = w1.unsqueeze(-1).unsqueeze(-1).repeat(1, C, T*H*W)

        w0_sorted = torch.where(condition, w0, w1) * g.unsqueeze(0).unsqueeze(-1)
        w1_sorted = torch.where(condition, w1, w0) * g.unsqueeze(0).unsqueeze(-1)

        cond1 = x >= alpha1_sorted + w0_sorted + w1_sorted
        cond2 = x >= alpha1_sorted + w0_sorted - w1_sorted
        cond3 = x >= alpha0_sorted + w0_sorted - w1_sorted
        cond4 = x >= alpha0_sorted - w0_sorted - w1_sorted

        res1 = x - w0_sorted - w1_sorted
        res2 = alpha1_sorted
        res3 = x - w0_sorted + w1_sorted
        res4 = alpha0_sorted
        res5 = x + w0_sorted + w1_sorted

        return torch.where(cond1, res1,
                           torch.where(cond2, res2, torch.where(cond3, res3, torch.where(cond4, res4, res5)))).view(B, C, T, H, W)



class ROMAN_R(nn.Module):
    def __init__(self, params=None):
        super(ROMAN_R, self).__init__()

        self._earlystop=None
        self.hd = params['hidden_filters']
        self.params = params
        self.filter = self.make_layers()


    def make_layers(self):
        params = self.params
        filt = []
        for i in range(params['layers']):
            filt.append(ROMAN_R_Layer(kernel=params['kernel'][i],
                                      coef_L_initializer=params['coef_L'],
                                      coef_S_initializer=params['coef_S'],
                                      coef_S_side_initializer=params['coef_S_side'],
                                      l1_l1=params['l1_l1'],
                                      reweightedl1_l1=params['reweightedl1_l1'],
                                      hidden_dim=self.hd,
                                      l1_l2=params['l1_l2']))

        return nn.Sequential(*filt)

    def forward(self, x):
        D = x
        L = torch.median(D, dim=1, keepdim=True).values.repeat(1, D.shape[1], 1, 1).cuda()
        M = torch.where(torch.abs(x.cuda() - L) > .05, torch.ones_like(L), torch.zeros_like(L))
        # M = torch.where(torch.abs(x.cuda() - L) > .5, torch.ones_like(L), torch.zeros_like(L))
        # M = torch.abs(x.cuda()-L)

        U = torch.zeros((L.shape[0], L.shape[1], L.shape[2], L.shape[3])).cuda()

        if self._earlystop is None:
            D, L, M, U = self.filter((D, L, M, U))
        else:
            for i in range(self._earlystop):
                D, L, M, U = self.filter[i].forward((D, L, M, U))

        return L, None, M


    def get_threshold_params(self):
        params = []
        for layer in self.filter:
            params.append(layer.coef_S)
            params.append(layer.coef_S_side)
            params.append(layer.coef_L)

        return params

    def set_early(self, es):
        self._earlystop = es
