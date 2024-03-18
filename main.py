from pickle import TRUE
import cv2
from cv2 import VideoWriter_fourcc

from dataloader.video_loader import Loader as LoaderTransform
from roman_r import ROMAN_R
from madu import Madu2
import numpy as np
import time
import datetime

import csv
import torch
import torch.nn as nn
import os
import yaml
import tqdm
import matplotlib.pyplot as plt
from utils import stats_per_frame, compute_F1, compute_pre_rec, TverskyLoss, MultiTaskLoss
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

mse_loss = nn.MSELoss()
cce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()
mae_loss = nn.L1Loss()
tversky = TverskyLoss(alpha_t=0.5, beta_t=0.5)
mtloss = None
mtloss2 = None


def save_results_to_csv(file_path, model_name, model_params, epochs, F1thresholds, F1_train, F1_eval, losses, category,
                        model_time):
    # Define the mapping from category number to name
    category_name_mapping = {
        0: 'baseline/pedestrians', 1: 'baseline/PETS2006', 2: 'baseline/highway', 3: 'baseline/office',
        4: 'lowFramerate/port_0_17fps', 5: 'lowFramerate/tramCrossroad_1fps', 6: 'lowFramerate/tunnelExit_0_35fps',
        7: 'lowFramerate/turnpike_0_5fps', 8: 'thermal/corridor', 9: 'thermal/diningRoom',
        10: 'thermal/lakeSide', 11: 'thermal/library', 12: 'thermal/park',
        13: 'badWeather/blizzard', 14: 'badWeather/skating', 15: 'badWeather/snowFall',
        16: 'badWeather/wetSnow', 17: 'dynamicBackground/boats', 18: 'dynamicBackground/canoe',
        19: 'dynamicBackground/fall', 20: 'dynamicBackground/fountain01', 21: 'dynamicBackground/fountain02',
        22: 'dynamicBackground/overpass', 23: 'shadow/backdoor', 24: 'shadow/bungalows',
        25: 'shadow/busStation', 26: 'shadow/copyMachine', 27: 'shadow/cubicle',
        28: 'shadow/peopleInShade', 29: 'nightVideos/bridgeEntry', 30: 'nightVideos/busyBoulvard',
        31: 'nightVideos/fluidHighway', 32: 'nightVideos/streetCornerAtNight', 33: 'nightVideos/tramStation',
        34: 'nightVideos/winterStreet', 35: 'turbulence/turbulence0', 36: 'turbulence/turbulence1',
        37: 'turbulence/turbulence2', 38: 'turbulence/turbulence3', 47: 'cameraJitter/badminton',
        48: 'cameraJitter/boulevard', 49: 'cameraJitter/sidewalk', 50: 'cameraJitter/traffic'
    }

    full_category_name = category_name_mapping.get(category,
                                                   "Unknown Category/Unknown Video")  # Default if category number is not found
    category_name, category_video = full_category_name.split('/', 1)  # Split the full category name into name and video

    try:
        with open(file_path, mode='a', newline='') as csv_file:  # Use 'a' for append mode
            fieldnames = ['Model Name', 'Model Parameters', 'Epochs', 'Category', 'Category Name', 'Category Video',
                          'Model Time'] + [f'F1 ({thresh})' for thresh in F1thresholds] + ['Loss']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if csv_file.tell() == 0:
                writer.writeheader()

            if not isinstance(F1_train, list):
                F1_train = [F1_train]  # Convert to a list if it's not already

            if not isinstance(F1_eval, list):
                F1_eval = [F1_eval]  # Convert to a list if it's not already

            if not isinstance(losses, list):
                losses = [losses]  # Convert to a list if it's not already

            for i in range(len(F1_train)):
                row_data = {
                    'Model Name': model_name,
                    'Model Parameters': model_params,
                    'Epochs': epochs,
                    'Category': category,
                    'Category Name': category_name,
                    'Category Video': category_video,
                    'Model Time': model_time,
                }
                row_data.update({f'F1 ({thresh})': F1_eval[i] for i, thresh in enumerate(F1thresholds)})
                row_data['Loss'] = losses[i]

                writer.writerow(row_data)

        print(f"Results appended to {file_path} successfully.")
    except Exception as e:
        print(f"Error appending results to {file_path}: {str(e)}")


def run_one_batch(net, batch, cfgs, l_type,forward_pass_type="default", nof1=False, profile=False, F1thresholds=None, class_weights=None,
                  counter=None):
    D, M, roi_batch = batch

    D = D.float().cuda()
    D.requires_grad = False

    if np.random.randint(2) == 1:
        M_motion = (M == 255)
    else:
        M_motion = torch.logical_or(M == 255, M == 170)

    # M_motion = torch.logical_or(M==255, M==170)
    M_motion = M_motion.long().cuda()
    M_motion.requires_grad = False

    L = None
    S = None

    if profile is False:
        if forward_pass_type == "mean":
            outputs_L, outputs_S, outputs_M = net.forward_mean(D)
        elif forward_pass_type == "median":
            outputs_L, outputs_S, outputs_M = net.forward_median(D)
        elif forward_pass_type == "uv":
            outputs_L, outputs_S, outputs_M = net.forward_uv(D)
        else:
            outputs_L, outputs_S, outputs_M = net(D)
    else:
        with torch.autograd.profiler.profile(use_cuda=True, with_flops=True) as prof:
            if forward_pass_type == "mean":
                outputs_L, outputs_S, outputs_M = net.forward_mean(D)
            elif forward_pass_type == "median":
                outputs_L, outputs_S, outputs_M = net.forward_median(D)
            elif forward_pass_type == "uv":
                outputs_L, outputs_S, outputs_M = net.forward_uv(D)
            else:
                outputs_L, outputs_S, outputs_M = net(D)
        print(prof.key_averages().table(sort_by='cuda_time_total'))

    if l_type == 'L_bce':
        mask = torch.logical_or(M == 0, M == 85).long().cuda().detach()
        if class_weights is not None:
            _w = torch.where(M_motion == 0, class_weights[0], class_weights[1])
            _w.requires_grad = False
            loss_tot = cfgs.alpha * _w * bce_loss(outputs_M, M_motion.float()) + mse_loss(mask * outputs_L, mask * D)
        else:
            loss_tot = cfgs.alpha * bce_loss(outputs_M, M_motion.float()) + mse_loss(mask * outputs_L, mask * D)

    elif l_type == 'L_tversky':
        mask = torch.logical_or(M == 0, M == 85).long().cuda().detach()
        loss_tot = cfgs.alpha * tversky(outputs_M, M_motion.float()) + mse_loss(mask * outputs_L, mask * D)

    elif l_type == 'L_tversky_bce':
        mask = torch.logical_or(M == 0, M == 85).long().cuda().detach()
        loss_tot = 0.5 * tversky(outputs_M, M_motion.float()) + 0.5 * bce_loss(outputs_M, M_motion.float()) + mse_loss(
            mask * outputs_L, mask * D)

    elif l_type == 'tversky_bce':
        mask = torch.logical_or(M == 0, M == 85).long().cuda().detach()
        loss_tot = 0.5 * tversky(outputs_M, M_motion.float()) + 0.5 * bce_loss(outputs_M, M_motion.float())

    elif l_type == 'LM':
        loss_tot = cfgs.alpha * bce_loss(outputs_M, M.float()) + mse_loss(outputs_L, L)

    if nof1 is False:

        if F1thresholds is None:
            stats = stats_per_frame((outputs_M.detach().cpu().numpy() > 0.5).astype(int),
                                    M.detach().cpu().numpy())
        else:
            stats = [stats_per_frame((outputs_M.detach().cpu().numpy() > tr).astype(int),
                                     M.detach().cpu().numpy()) for tr in F1thresholds]

    else:
        stats = (1, 1, 1, 1)

    return (loss_tot, stats), (outputs_L, outputs_S, outputs_M)


def train_submodel(net, data_loader, cfgs, log_file, epochs, forward_pass_type="default"):
    l_type = cfgs.loss_type

    # Initialize the optimizer and scheduler
    optimizer = optim.Adam(net.parameters(), lr=cfgs.initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfgs.lr_decay_intv, gamma=cfgs.lr_decay_rate)

    for epoch in range(epochs):
        print(f'Training epoch {epoch} for {forward_pass_type}')
        train_one_epoch(net, data_loader, cfgs, optimizer, scheduler, log_file, l_type, forward_pass_type)

def train_one_epoch(net, data_loader, cfgs, optimizer, scheduler, log_file, l_type, forward_pass_type="default"):
    net.train()
    train_losses = []
    batch_size = cfgs.batch_size

    data_loader.current_idx_train = 0
    data_loader.current_idx_val = 0
    # Initialize tqdm loop
    iterator = tqdm.tqdm(range(data_loader.train_samples // batch_size))

    for batch_idx in iterator:
        optimizer.zero_grad()

        # Load batch
        batch = data_loader.load_batch_train(batch_size)

        # Rest of the operations remain the same
        losses, outputs = run_one_batch(net, batch, cfgs, l_type, nof1=True, forward_pass_type=forward_pass_type)
        loss_tot = losses[0]
        loss_tot.backward()
        train_losses.append(loss_tot.item())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25)
        optimizer.step()

    avg_loss = np.mean(train_losses)

    print(f'Average training loss: {avg_loss}')

    scheduler.step()
    data_loader.shuffle()

    return avg_loss

def main_training_function(net, log_dir, data_loader, cfgs, log_file, checkpoint_file, pytorch_total_params, writecsv):

    params = 0
    epochs = cfgs.epochs
    print("\n\nStep 1: Training loop for Mean only layers")
    train_submodel(net, data_loader, cfgs, log_file, epochs, forward_pass_type="mean")

    print("\n\nStep 2: Training loop for Median only layers")
    train_submodel(net, data_loader, cfgs, log_file, epochs, forward_pass_type="median")

    print("\n\nStep 3: Training loop for UV layers")
    train_submodel(net, data_loader, cfgs, log_file, epochs, forward_pass_type="uv")

    mode = "default"
   
    print("\n\nStep 4: Freeze The Mean Median UV layers and optimize only the weights for each algorithm at each layer")

    net.freeze_parallel_layers()

    weight_epoch = 1
    train_submodel(net, data_loader, cfgs, log_file, weight_epoch, forward_pass_type= mode)

    print("\n\nStep 5: (Optional) Unfreeze the parameters and fine-tune the whole Network")
    net.unfreeze_parallel_layers()

    train_submodel(net, data_loader, cfgs, log_file, epochs, forward_pass_type= mode)

    print("\n\nStep 6: Evaluate the model")
    f1_test = eval(net, data_loader, cfgs, log_dir, log_file, checkpoint_file, cfgs.loss_type,forward_pass_type= mode)

    # Save the results to the CSV file
    model_name = cfgs.name
    epochs = cfgs.epochs
    F1thresholds = [0.5]  # Update with your F1thresholds
    F1_train = f1_test[0]  # Update with your F1_train values
    F1_eval = f1_test[1][-1]  # Update with your F1_eval values
    all_losses = f1_test[-2]  # Update with your losses
    model_time = f1_test[-1]
    file_write = cfgs.file_write
    if writecsv == True:
        print("Writing the results to excel file")
        save_results_to_csv(
            file_write,
            model_name,
            params,
            epochs,
            F1thresholds,
            F1_train,
            F1_eval,
            all_losses,
            cfgs.category,
            model_time
        )

    return f1_test

def eval(net, data_loader, cfgs, log_dir, log_file, ckpt, l_type, verbose=True, threshVid=None, forward_pass_type = "default"):
    # Evaluating on real videos, with visual output and statistics computed on a video-basis (if the selected dataset has multiple videos)

    log_dir = os.path.join(log_dir, cfgs.name)  # Append cfgs.name to the existing log_dir path

    # Create the directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if verbose: print('Testing on videos separately:')
    net.eval()

    all_losses = []
    all_F1_train = []
    all_F1_eval = []
    all_pr_eval = []

    gt_probs = []
    probs = []

    cat_idx = 0
    model_time = 0
    with torch.no_grad():

        counter = 0
        counter2 = 0
        # F1thresholds = list(np.arange(.1, 1.1, step=.1,))
        F1thresholds = [0.5]
        counter3 = 0

        # Looping over categories
        while True:

            clip_idx = 0
            eval_losses = []
            train_stats = []
            eval_stats = []
            # Looping over clips
            while True:

                batch = data_loader.load_clip_from_category_eval(cat_idx, clip_idx)

                if batch is None:
                    # Meaning we are out of categories
                    break
                if len(batch[0]) == 0:
                    # Meaning we are out of clips in category
                    break

                if (cat_idx, clip_idx) not in data_loader.eval:
                    clip_idx += 1
                    continue

                if counter2 == 20:
                    break
                else:
                    counter2 += 1

                D, M, roi = batch

                # Forward pass

                start_time = time.time()
                losses, outputs = run_one_batch(net, batch, cfgs, l_type, nof1=True, profile=False, counter=counter3, forward_pass_type=forward_pass_type)
                model_time1 = time.time() - start_time
                model_time = model_time + model_time1
                L_assembled = outputs[0]
                M_assembled = outputs[2]

                # Visual output

                inputs = batch[0][0, ...].detach().cpu().numpy().astype(np.float32)
                M_true = batch[1][0, ...].numpy().astype(np.float32)
                outputs_L = L_assembled[0, ...]
                outputs_M = M_assembled[0, ...]

                if log_dir is not None:
                    # for frame_idx in range(inputs.shape[0]):
                    D_image = cv2.cvtColor(inputs[10, ...]*255, cv2.COLOR_GRAY2BGR)
                    M_image = cv2.cvtColor(outputs_M[10, ...].detach().cpu().numpy()*255, cv2.COLOR_GRAY2BGR)
                    M_True_image = cv2.cvtColor(M_true[10, ...]*255, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(os.path.join(log_dir, '{}_{}_D.png'.format(counter, 10)), D_image)
                    cv2.imwrite(os.path.join(log_dir, '{}_{}_M.png'.format(counter, 10)),M_image)
                    cv2.imwrite(os.path.join(log_dir, '{}_{}_MTrue.png'.format(counter, 10)),M_True_image)
            

                counter += 1

                # Statistics
                loss_tot, f1 = losses

                stats = [stats_per_frame((outputs_M.unsqueeze(0).detach().cpu().numpy() >= tr).astype(int),
                                         M.detach().cpu().numpy()) for tr in F1thresholds]

                if (cat_idx, clip_idx) in data_loader.eval:
                    eval_losses.append(loss_tot.item())
                    eval_stats.append(stats)

                    # FOR ROC
                    masks_target = M[0].detach().cpu().numpy()
                    mask_gt = (masks_target == 255).astype(int)  # F1 score computed for WHITE pixels (motion)
                    roi_main = np.logical_and(masks_target != 85, masks_target != 170).astype(
                        int)  # ROI does not accounts for ROI and NON-UNKNOWN pixels

                    gt_probs.append(mask_gt[roi_main == True])
                    probs.append(outputs_M.detach().cpu().numpy()[roi_main == True])
                if (cat_idx, clip_idx) in data_loader.train:
                    train_stats.append(stats)

                clip_idx += 1


                # break

            if batch is None:
                # Meaning we are out of categories
                break

            catF1_train = [compute_F1([batch_stats[f1_index] for batch_stats in train_stats]) for f1_index in
                           range(len(F1thresholds))]
            catF1_eval = [compute_F1([batch_stats[f1_index] for batch_stats in eval_stats]) for f1_index in
                          range(len(F1thresholds))]
            all_F1_train.append(catF1_train)
            all_F1_eval.append(catF1_eval)
            all_pr_eval.append([
                compute_pre_rec([batch_stats[f1_index] for batch_stats in eval_stats]) for f1_index in
                range(len(F1thresholds))
            ])

            catLoss = np.mean(eval_losses)
            all_losses.append(catLoss)

            toprint = 'Video {} -- loss: {:.6f} '.format(cat_idx, catLoss) + ', '.join(
                ['F1 ({}): {:.6f}'.format(F1thresholds[i], catF1_eval[i]) for i in range(len(F1thresholds))])
            if verbose:
                print(toprint)
                with open(log_file, 'a') as myfile:
                    myfile.write(toprint + '\n')

            best_trainF1_idx = np.argmax(catF1_train)
            if verbose: print('Best F1 on training set: {}'.format(catF1_train[best_trainF1_idx]))
            if verbose: print('Best F1 obtained with threshold {}: F1= {}'.format(F1thresholds[best_trainF1_idx],
                                                                                  catF1_eval[best_trainF1_idx]))

            cat_idx += 1

        totLoss = np.mean(all_losses)
        totF1 = np.mean(np.asarray(all_F1_eval), axis=0)
        toprint = 'Overall -- loss: {:.6f} '.format(totLoss) + ', '.join(
            ['F1 ({}): {:.6f}'.format(F1thresholds[i], totF1[i]) for i in range(len(F1thresholds))])
        if verbose:
            print(toprint)
            with open(log_file, 'a') as myfile:
                myfile.write(toprint + '\n')

        return best_trainF1_idx, all_F1_eval, probs, gt_probs, all_pr_eval, all_F1_train, totLoss, model_time / counter


if __name__ == '__main__':

    if torch.cuda.is_available():
        print("CUDA is available. You can use GPU for computations.")
    else:
        print("CUDA is not available. Check your GPU and installation.")

    ##### Main configuration #####
    # dataset = 'mmnist'
    from configs.cdnet import cfgs

    if 'model' not in cfgs:
        cfgs.model = 'roman_s'

    np.random.seed(cfgs.seed)
    torch.manual_seed(cfgs.seed)

    writecsv = cfgs.writecsv

    # Log files
    if cfgs.load is None:
        log_dir = os.path.join(cfgs.log_dir, cfgs.name)  # Use cfgs.name instead of the timestamp
        duplicate_counter = 1

        log_dir_2 = log_dir
        while os.path.exists(log_dir_2) is True:
            duplicate_counter += 1
            log_dir_2 = log_dir + '_' + str(duplicate_counter)  # Add counter to avoid overwriting existing directories
        log_dir = log_dir_2
        os.makedirs(log_dir)

        log_file = os.path.join(log_dir, 'log_{}'.format(cfgs.name))  # Use cfgs.name in log filename
        with open(log_file, 'a') as myfile:
            myfile.write(str(cfgs))

        checkpoint_file = os.path.join(log_dir, 'checkpoint.pth')
        cfgs_file = os.path.join(log_dir, 'config.yaml')
        with open(cfgs_file, 'w') as myfile:
            yaml.dump(cfgs, myfile, default_flow_style=False)

    ##### Data #####

    data_loader = None

    data_loader = LoaderTransform(cfgs.data_path, cfgs.categories, time_steps=cfgs.time_steps,
                                  flatten=False, scale=False, maxsize=cfgs.max_size,
                                  patch_size=cfgs.patch_size, seed=cfgs.seed, crop_size=cfgs.crop_size,
                                  compute_weights=True, include_in_train=cfgs.include_in_train, split=cfgs.split)

    batch = data_loader.load_clip_from_category_eval(0, 0)
    D, M, roi = batch
    H = D.shape[2]
    params_net2 = {'layers': cfgs.layers,
                   'kernel': [(1, 5)] * 10,
                   'hidden_filters': cfgs.hidden_filters,
                   'coef_L': cfgs.coeff_L,
                   'coef_S': cfgs.coeff_S,
                   'coef_S_side': cfgs.coeff_Sside,
                   'l1_l1': False,
                   'reweightedl1_l1': cfgs.reweighted,
                   'l1_l2': cfgs.l1_l2,
                   'img_size': cfgs.crop_size,
                   'H': H,
                   'threshold_all': cfgs.threshold_all
                   }

    net = Madu2(params_net2)

    if torch.cuda.is_available():
        net = net.cuda()

    layer_list = cfgs.layer_list

    #### Counting parameters ####
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print('Model has {} trainable parameters'.format(pytorch_total_params))

    # -------------------------------------------------------------------------------------#
    ##### Train #####
    log_dir2 = f'./Tensorboard/category_{cfgs.category}/3_layers'
    writer = SummaryWriter(log_dir=log_dir2)
    f2 = main_training_function(net, log_dir, data_loader, cfgs, log_file, checkpoint_file, pytorch_total_params,
                           writecsv)
    writer.close()

