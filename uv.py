from roman_r import Conv3dC
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)


class UV_Layer(nn.Module):
    def __init__(self, kernel, H, threshold_all, coef_L_initializer,
                 coef_S_initializer, coef_S_side_initializer,
                 l1_l1, reweightedl1_l1, hidden_dim=32, l1_l2=False):
        super(UV_Layer, self).__init__()

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
        self.tau_alpha1 = nn.Parameter(torch.tensor(1.0))
        self.tau_alpha2 = nn.Parameter(torch.tensor(1.0))

        self.rho = nn.Parameter(torch.tensor(1.0))
        self.mask_scaling = nn.Parameter(torch.tensor(5.0))

        self.g = nn.Parameter(torch.ones(hidden_dim))
        self.th = nn.Parameter(torch.full((128,), coef_S_initializer))
        self.g2 = nn.Parameter(torch.tensor(torch.ones(H)))

        self.coef_L = nn.Parameter(torch.tensor(coef_L_initializer))
        self.coef_S = nn.Parameter(torch.tensor(coef_S_initializer))
        self.coef_S_side = nn.Parameter(torch.tensor(coef_S_side_initializer))
        self.l1_l1 = l1_l1
        self.reweightedl1_l1 = reweightedl1_l1
        self.l1_l2 = l1_l2
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.threshold_all = threshold_all

        self.u_learn2 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(5)])
        self.v_learn2 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(5)])

    def forward(self, data):

        tau_s = nn.functional.softplus(self.tau_s)

        X = data[0]
        U = data[1]
        V = data[2]
        M = data[3]

        B, T, H, W = X.shape

        # Conversion to feature map
        M = self.convM(M)

        background_mask = 1 - torch.relu(self.conv0(M))
        background_mask_squared = torch.square(background_mask)
        for i in range(5):
            temp = background_mask_squared * (X - torch.matmul(U, V).view(1, T, H, W))
            temp = temp.squeeze(0).view(T, H * W)

            # Update U
            gradient_U = torch.matmul(temp.permute(1, 0), V.T)

            U = U + 0.00001 * self.u_learn2[i] * gradient_U

            # Update V
            gradient_V = torch.matmul(temp, U).view(1, T)

            V = V + 0.00001 * self.v_learn2[i] * gradient_V

        # L branch
        L = torch.matmul(U, V).permute(1, 0).view(T, H, W).unsqueeze(0)

        # M branch
        foreground = X - L
        foreground_squared = torch.square(foreground)
        _M = M
        _X = tau_s * self.conv2((1 - self.conv1(M)) * foreground_squared.unsqueeze(1))

        Mtmp = _M + _X
        if self.threshold_all:
            Mtmp = self.soft_l1_reweighted2(Mtmp, self.g)
        else:
            Mtmp = self.soft_l1_reweighted(Mtmp, self.coef_S, self.g)

        Mtmp = self.conv4(Mtmp)

        M = nn.functional.sigmoid((Mtmp - 0.5) * self.mask_scaling)

        return (X, U, V, M)

    def soft_l1_reweighted(self, z, th, g):
        B, C, T, H, W = z.shape
        z = z.view(B, C, T * H * W)
        th = th.unsqueeze(-1).unsqueeze(-1)
        out = torch.sign(z) * nn.functional.relu(torch.abs(z) - th * g.unsqueeze(0).unsqueeze(-1))
        return out.view(B, C, T, H, W)

    def soft_l1_reweighted2(self, z, g):
        B, C, T, H, W = z.shape
        z = z.view(B, C, T, H * W)
        th2 = self.th.unsqueeze(0).unsqueeze(0) * self.g2.unsqueeze(0).unsqueeze(-1) * self.g.unsqueeze(-1).unsqueeze(
            -1)
        th2 = th2.view(C, H * W)
        th2 = th2.unsqueeze(0).unsqueeze(2)
        out = torch.sign(z) * nn.functional.relu(torch.abs(z) - th2)
        return out.view(B, C, T, H, W)

