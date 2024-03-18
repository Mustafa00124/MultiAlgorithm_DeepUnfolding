import numpy as np
import cv2
import math
import torch

import torch


def divide_sequence(source, time_steps):
    seq_num = int(len(source) / time_steps)
    return [source[time_steps * i:time_steps * (i + 1), :, :] for i in range(seq_num)]


def resize_sequence(source, maxsize, inter):
    scalefac = maxsize / max(source.shape[1], source.shape[2])
    if len(source.shape) == 3:
        output = np.empty((source.shape[0], round(source.shape[1] * scalefac), round(source.shape[2] * scalefac)),
                          dtype=source.dtype)
    else:
        output = np.empty((source.shape[0], round(source.shape[1] * scalefac), round(source.shape[2] * scalefac), 3),
                          dtype=source.dtype)

    for i in range(source.shape[0]):
        output[i, ...] = cv2.resize(source[i, ...], (0, 0), fx=scalefac, fy=scalefac, interpolation=inter)
    return output

def resize_unique(source, maxsize, inter):
    scalefac = maxsize / max(source.shape[0], source.shape[1])
    output = cv2.resize(source, (0, 0), fx=scalefac, fy=scalefac, interpolation=inter)
    return output


def crop_sequence(source, crop_size):
    if len(crop_size) == 1:
        crop_size_x = crop_size_y = crop_size
    else:
        crop_size_x = crop_size[0]
        crop_size_y = crop_size[1]

    offsetW = round((source.shape[1] - crop_size_x) / 2)
    offsetH = round((source.shape[2] - crop_size_y) / 2)
    output = source[:, offsetW:offsetW + crop_size_x, offsetH:offsetH + crop_size_y, ...]

    return output

def crop_unique(source, crop_size):
    offsetW = round((source.shape[0] - crop_size) / 2)
    offsetH = round((source.shape[1] - crop_size) / 2)
    output = source[offsetW:offsetW + crop_size, offsetH:offsetH + crop_size, ...]

    return output


def create_patches(source, patch_size):
    n = source.shape[0]
    duration = source.shape[1]
    sx, sy = source.shape[2], source.shape[3]
    nx, ny = math.floor(sx / patch_size), math.floor(sy / patch_size)
    outputs = []

    for k in range(n):
        inp = source[k, ...]
        out = []
        for i in range(nx):
            for j in range(ny):
                out.append(inp[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size])

        outputs.append(np.asarray(out))
        # outputs.append(torch.stack(out, dim=0))

    output = np.concatenate(outputs, axis=0)

    return output

def create_patches_torch(source, patch_size):
    n = source.shape[0]
    assert n==1
    duration = source.shape[1]
    sx, sy = source.shape[2], source.shape[3]
    nx, ny = math.floor(sx / patch_size), math.floor(sy / patch_size)
    outputs = []

    for k in range(n):
        inp = source[k, ...]
        out = []
        for i in range(nx):
            for j in range(ny):
                out.append(inp[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size])

        # outputs.append(np.asarray(out))
        outputs.append(torch.stack(out, dim=0))

    output = outputs[0]

    return output



def create_overlapping_patches(source, patch_size, stride):
    duration = source.shape[0]
    sx, sy = source.shape[1], source.shape[2]
    assert(stride <= patch_size)
    #Check if image fits the patches
    assert((sx - patch_size) % stride == 0)
    assert((sy - patch_size) % stride == 0)

    ny = int((sy - patch_size)/stride + 1)
    nx = int((sx - patch_size)/stride + 1)

    #extract patches
    out = np.empty((nx*ny, duration, patch_size, patch_size), dtype=source.dtype)
    for i in range(nx):
        for j in range(ny):
            out[i*ny + j, ...] = source[:, i*stride:i*stride + patch_size, j*stride:j*stride + patch_size]

    return out


def assemble_patches(source, patch_size, sx, sy, overlap=0):
    if overlap==0:
        assert (sx % patch_size == 0)
        assert (sy % patch_size == 0)
        nx, ny = int(sx / patch_size), int(sy / patch_size)
        timesteps = source.shape[0]

        output = np.empty((timesteps, sx, sy), dtype='float32')
        for i in range(nx):
            for j in range(ny):
                for t in range(timesteps):
                    output[t, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = source[t, i*ny + j, ...]

        return output

    else:
        raise Exception('Overlap !=0 not yet implemented')

def assemble_patches_torch(source, patch_size, sx, sy, overlap=0):
    if overlap==0:
        assert (sx % patch_size == 0)
        assert (sy % patch_size == 0)
        nx, ny = int(sx / patch_size), int(sy / patch_size)
        timesteps = source.shape[1]

        output = torch.empty((1, timesteps, sx, sy), device=source.device, dtype=source.dtype)
        for i in range(nx):
            for j in range(ny):
                    output[0, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = source[i*ny + j, ...]

        return output

    else:
        raise Exception('Overlap !=0 not yet implemented')


def assemble_overlapping_patches(source, patch_size, sx, sy, stride):
    # Check if image fits the patches
    assert ((sx - patch_size) % stride == 0)
    assert ((sy - patch_size) % stride == 0)
    ny = int((sy - patch_size)/stride + 1)
    nx = int((sx - patch_size)/stride + 1)
    timesteps = source.shape[0]

    output = np.zeros((timesteps, sx, sy), dtype='float32')
    avmat = np.zeros((sx, sy), dtype='float32')
    for i in range(nx):
        for j in range(ny):
            for t in range(timesteps):
                output[t, i*stride:i*stride + patch_size, j*stride:j*stride+patch_size] += source[t, i*ny + j, ...]
            avmat[i*stride:i*stride + patch_size, j*stride:j*stride+patch_size] += 1
    output = output / avmat
    return output

