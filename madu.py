import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import torch.nn.functional as F
import math
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)
from mean import Mean_Layer
from median import Median_Layer
from uv import UV_Layer

class ParallelLayer(nn.Module):
    def __init__(self, params):
        super(ParallelLayer, self).__init__()
        self.mean_layer = Mean_Layer(
            kernel=params['kernel'][0],
            H=params['H'],
            threshold_all=True,
            coef_L_initializer=params['coef_L'],
            coef_S_initializer=params['coef_S'],
            coef_S_side_initializer=params['coef_S_side'],
            l1_l1=params['l1_l1'],
            reweightedl1_l1=params['reweightedl1_l1'],
            hidden_dim=params['hidden_filters'],
            l1_l2=params['l1_l2']
        ).cuda()
        self.median_layer = Median_Layer(
            kernel=params['kernel'][0],
            H=params['H'],
            threshold_all=True,
            coef_L_initializer=params['coef_L'],
            coef_S_initializer=params['coef_S'],
            coef_S_side_initializer=params['coef_S_side'],
            l1_l1=params['l1_l1'],
            reweightedl1_l1=params['reweightedl1_l1'],
            hidden_dim=params['hidden_filters'],
            l1_l2=params['l1_l2']
        ).cuda()
        self.uv_layer = UV_Layer(
            kernel=params['kernel'][0],
            H=params['H'],
            threshold_all=True,
            coef_L_initializer=params['coef_L'],
            coef_S_initializer=params['coef_S'],
            coef_S_side_initializer=params['coef_S_side'],
            l1_l1=params['l1_l1'],
            reweightedl1_l1=params['reweightedl1_l1'],
            hidden_dim=params['hidden_filters'],
            l1_l2=params['l1_l2']
        ).cuda()
        
        # Initialize weights
        w = 3
        self.weights_L = nn.Parameter(torch.ones(w) / w)  # Equal weights for L
        self.weights_M = nn.Parameter(torch.ones(w) / w)  # Equal weights for M

        self.weights_L3 = nn.Parameter(torch.ones(w) / w)  # Equal weights for L
        self.weights_M3 = nn.Parameter(torch.ones(w) / w)  # Equal weights for M

        y = 4
        self.weights_L2 = nn.Parameter(torch.ones(y) / y)  # Equal weights for L
        self.weights_M2 = nn.Parameter(torch.ones(y) / y)  # Equal weights for M

    def forward_roman(self, x, l, m):
        # Forward pass for mean layer
        D, l_roman, m_roman = self.roman_layer((x, l, m))
        return l_roman, m_roman
    def forward_mean(self, x, l, m):
        # Forward pass for mean layer
        D, l_mean, m_mean = self.mean_layer((x, l, m))
        return l_mean, m_mean

    def forward_median(self, x, l, m):
        # Forward pass for median layer
        D, l_median, m_median = self.median_layer((x, l, m))
        return l_median, m_median

    def forward_uv(self, x, u, v, m):
        # Forward pass for UV layer
        B, T, H, W = x.shape
        D, u, v, m_uv = self.uv_layer((x, u, v, m))
        l_uv = torch.matmul(u, v).permute(1, 0).view(B, T, H, W)
        return l_uv, m_uv, u, v

    def forward(self, x, l, m, u, v):
        # Run each layer
        B, T, H, W = x.shape
        D, l_mean, m_mean = self.mean_layer((x, l, m))
        D, l_median, m_median = self.median_layer((x, l, m))
        D, u, v, m_uv = self.uv_layer((x, u, v, m))
        l_uv = torch.matmul(u, v).permute(1, 0).view(B, T, H, W)

        # Normalize weights for L and M
        normalized_weights_L = F.softmax(self.weights_L, dim=0)
        normalized_weights_M = F.softmax(self.weights_M, dim=0)

        # Combine outputs
        L_combined = normalized_weights_L[0] * l_uv + normalized_weights_L[1] * l_mean + normalized_weights_L[2] * l_median
        M_combined = normalized_weights_M[0] * m_uv + normalized_weights_M[1] * m_mean + normalized_weights_M[2] * m_median

        return L_combined, M_combined, u, v

    def freeze_weights(self):
        # Freeze the weights of the pretrained layers
        self.pretrained_weights_frozen = True
        for layer in [self.mean_layer, self.median_layer, self.uv_layer]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_weights(self):
        # Unfreeze the weights of the pretrained layers
        self.pretrained_weights_frozen = False
        for layer in [self.mean_layer, self.median_layer, self.uv_layer]:
            for param in layer.parameters():
                param.requires_grad = True


class Madu2(nn.Module):
    def __init__(self, params):
        super(Madu2, self).__init__()
        self.params = params
        self.parallel_layers = self.make_parallel_layers()

        self.h = params['H']


        # Defining learnable weight matrices for each model
        self.weights_L_mean = nn.Parameter(torch.ones((2, 2)) / 4*3)  # For mean model
        self.weights_L_median = nn.Parameter(torch.ones((2, 2)) / 4*3)  # For median model
        self.weights_L_uv = nn.Parameter(torch.ones((2, 2)) / 4*3)  # For UV model

        self.weights_M_mean = nn.Parameter(torch.ones((2, 2)) / 4*3)  # For mean model
        self.weights_M_median = nn.Parameter(torch.ones((2, 2)) / 4*3)  # For median model
        self.weights_M_uv = nn.Parameter(torch.ones((2, 2)) / 4*3)  # For UV model

        # Low rank weights
        self.u_L_mean = nn.Parameter(torch.ones(self.h, 1))
        self.v_L_mean = nn.Parameter(torch.ones(128, 1) / 3.0)

        self.u_L_median = nn.Parameter(torch.ones(self.h, 1))
        self.v_L_median = nn.Parameter(torch.ones(128, 1) / 3.0)

        self.u_L_uv = nn.Parameter(torch.ones(self.h, 1))
        self.v_L_uv = nn.Parameter(torch.ones(128, 1) / 3.0)

        self.u_M_mean = nn.Parameter(torch.ones(self.h, 1))
        self.v_M_mean = nn.Parameter(torch.ones(128, 1) / 3.0)

        self.u_M_median = nn.Parameter(torch.ones(self.h, 1))
        self.v_M_median = nn.Parameter(torch.ones(128, 1) / 3.0)

        self.u_M_uv = nn.Parameter(torch.ones(self.h, 1))
        self.v_M_uv = nn.Parameter(torch.ones(128, 1) / 3.0)

        w = 3
        self.weights_L4 = nn.Parameter(torch.ones(w) / w)  # Equal weights for L
        self.weights_M4 = nn.Parameter(torch.ones(w) / w)  # Equal weights for M

        

    def depthwise_softmax(self,mean,median,uv):
        # Stack matrices along a new dimension (0) to apply softmax depthwise
        stacked = torch.stack([mean, median, uv], dim=0)  # Shape becomes [3, H, W] for 3 matrices
        # Apply softmax across the first dimension (models)
        softmaxed = F.softmax(stacked, dim=0)
        return softmaxed


    def make_parallel_layers(self):
        layers = []
        for _ in range(self.params['layers']):
            layers.append(ParallelLayer(self.params))
        return nn.Sequential(*layers)
    
    def forward_roman(self, x):
        D = x
        B, T, H, W = x.shape
        U = torch.median(x, dim=1).values.view(H * W).cuda()
        V = torch.ones(T).cuda()
        U = U.view(H * W, 1)
        V = V.view(1, T)
        L = torch.matmul(U, V).permute(1, 0).view(T, H, W).unsqueeze(0)
        M = torch.where(torch.abs(x - L) > 0.05, torch.ones_like(x), torch.zeros_like(x))
        for layer in self.parallel_layers:
            L, M = layer.forward_roman(x, L, M)
        return L, None, M

    def forward_mean(self, x):
        D = x
        B, T, H, W = x.shape
        U = torch.median(x, dim=1).values.view(H * W).cuda()
        V = torch.ones(T).cuda()
        U = U.view(H * W, 1)
        V = V.view(1, T)
        L = torch.matmul(U, V).permute(1, 0).view(T, H, W).unsqueeze(0)
        M = torch.where(torch.abs(x - L) > 0.05, torch.ones_like(x), torch.zeros_like(x))
        for layer in self.parallel_layers:
            L, M = layer.forward_mean(x, L, M)
        return L, None, M

    def forward_median(self, x):
        D = x
        B, T, H, W = x.shape
        U = torch.median(x, dim=1).values.view(H * W).cuda()
        V = torch.ones(T).cuda()
        U = U.view(H * W, 1)
        V = V.view(1, T)
        L = torch.matmul(U, V).permute(1, 0).view(T, H, W).unsqueeze(0)
        M = torch.where(torch.abs(x - L) > 0.05, torch.ones_like(x), torch.zeros_like(x))
        for layer in self.parallel_layers:
            L, M = layer.forward_median(x, L, M)
        return L, None, M

    def forward_uv(self, x):
        D = x
        B, T, H, W = x.shape
        U = torch.median(x, dim=1).values.view(H * W).cuda()
        V = torch.ones(T).cuda()
        U = U.view(H * W, 1)
        V = V.view(1, T)
        L = torch.matmul(U, V).permute(1, 0).view(T, H, W).unsqueeze(0)
        M = torch.where(torch.abs(x - L) > 0.05, torch.ones_like(x), torch.zeros_like(x))
        for layer in self.parallel_layers:
            L, M, U, V = layer.forward_uv(x, U, V, M)
        return L, None, M
    

    def forward(self, x):
        L_uv, _, M_uv = self.forward_uv(x)
        L_median, _, M_median = self.forward_median(x)
        L_mean, _, M_mean = self.forward_mean(x)

        # Create rank-1 weight matrices for each model's L output
        weight_matrix_L_uv = torch.matmul(self.u_L_uv, self.v_L_uv.T)
        weight_matrix_L_median = torch.matmul(self.u_L_median, self.v_L_median.T)
        weight_matrix_L_mean = torch.matmul(self.u_L_mean, self.v_L_mean.T)

        weight_matrix_M_uv = torch.matmul(self.u_M_uv, self.v_M_uv.T)
        weight_matrix_M_median = torch.matmul(self.u_M_median, self.v_M_median.T)
        weight_matrix_M_mean = torch.matmul(self.u_M_mean, self.v_M_mean.T)

        # Apply depthwise softmax to normalize the weight matrices
        softmaxed_weights_L = self.depthwise_softmax(weight_matrix_L_uv, weight_matrix_L_median, weight_matrix_L_mean)
        softmaxed_weights_M = self.depthwise_softmax(weight_matrix_M_uv, weight_matrix_M_median, weight_matrix_M_mean)

        # Use the softmaxed weight matrices to weight the L outputs
        L_weighted = softmaxed_weights_L[0].unsqueeze(0).unsqueeze(0) * L_uv + softmaxed_weights_L[1].unsqueeze(0).unsqueeze(0) * L_median + softmaxed_weights_L[2].unsqueeze(0).unsqueeze(0) * L_mean
        M_weighted = softmaxed_weights_M[0].unsqueeze(0).unsqueeze(0) * M_uv + softmaxed_weights_M[1].unsqueeze(0).unsqueeze(0) * M_median + softmaxed_weights_M[2].unsqueeze(0).unsqueeze(0) * M_mean
    
        M_weighted = M_weighted.clamp(min=0, max=1)
        return L_weighted, None, M_weighted

    def freeze_parallel_layers(self):
        # Freeze weights in all ParallelLayer instances
        for layer in self.parallel_layers:
            layer.freeze_weights()

    def unfreeze_parallel_layers(self):
        # Unfreeze weights in all ParallelLayer instances
        for layer in self.parallel_layers:
            layer.unfreeze_weights()





                                                                                                   
