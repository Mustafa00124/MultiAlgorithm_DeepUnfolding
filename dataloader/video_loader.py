import numpy as np
import os
import cv2
import torch

from dataloader import utils
import tqdm
import torchvision


class Loader:

    def __init__(self, path, clips, time_steps,
                 flatten=True, scale=False, maxsize=None, patch_size=None, seed=None, crop_size=None, compute_weights=False, include_in_train=None, split=-1, all_in_val=False):

        # This class contains a dataloader which yields either random samples in form of image patches or full images, depending on the model configuration (patch-based training or full-frame evaluation)

        self.time_steps = time_steps
        self.patch_size = patch_size
        self.seed = seed
        self.split = split
        self.include_in_train = include_in_train
        self.flatten = flatten
        np.random.seed(self.seed)

        # Defining the image transforms
        self.transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        self.transform_train_mask = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])


        self.transform_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        self.transform_eval_mask = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        videos = []
        masks = []
        ROIS = []
        include_in_train_idx = []


        # Importing each video separately and clipping the video so that its length is a multiple of time_steps
        clip_idx = 0
        for clip in tqdm.tqdm(clips):
            video, mask, ROI, temporal_offset = self.load_clip(path, clip)

            max_frames = (video.shape[0] // time_steps) * time_steps
            video = video[0:max_frames, ...]
            mask = mask[0:max_frames, ...]

            videos.append(video)
            masks.append(mask)

            if len(ROI.shape) != 3:
                ROI = np.expand_dims(ROI, 0)
            ROIS.append(ROI)

            include_in_train_idx.append([(t-temporal_offset)//time_steps for t in include_in_train[clip_idx]])
        # Debug: Print the shape of the loaded data before any processing
       
        clip_idx += 1
        # Debug: Print the shape of the loaded data after trimming
        

       
        # Resize if necessary
        if maxsize is not None:
            videos = [utils.resize_sequence(v, maxsize, cv2.INTER_AREA) for v in videos]
            masks = [utils.resize_sequence(m, maxsize, cv2.INTER_NEAREST) for m in masks]
            ROIS = [utils.resize_sequence(r, maxsize, cv2.INTER_NEAREST) for r in ROIS]
            # Debug: Print the shape of the loaded data after trimming
            

        # Crop if necessary, which is, if a patch size is specifier
        if patch_size is not None:
            videos = [utils.crop_sequence(v, (round(v.shape[1]//patch_size*patch_size), round(v.shape[2]//patch_size*patch_size))) for v in videos]
            masks = [utils.crop_sequence(m, (round(m.shape[1]//patch_size*patch_size), round(m.shape[2]//patch_size*patch_size))) for m in masks]
            ROIS = [utils.crop_sequence(r, (round(r.shape[1]//patch_size*patch_size), round(r.shape[2]//patch_size*patch_size))) for r in ROIS]
            

        # Split in time_steps frames long patches
        videos = [np.asarray(utils.divide_sequence(v, self.time_steps)) for v in videos]
        masks = [np.asarray(utils.divide_sequence(m, self.time_steps)) for m in masks]

        # Debug: Print the shape of the data after dividing into segments
        

        # Flatten
        if flatten is True:
            print("videos before: ", videos[0].shape)
            videos = [np.reshape(v, (v.shape[0], v.shape[1], -1)) for v in videos]
            print("videos after: ", videos[0].shape)
            masks = [np.reshape(m, (m.shape[0], m.shape[1], -1)) for m in masks]
            #ROIS = [np.reshape(r, (r.shape[0], -1)) for r in ROIS]
            
            

        # Convert mask to binary mask if necessary
        if compute_weights is True:
            _masks = [np.logical_or(m == 50, np.logical_or(m == 170, m == 255)).astype(np.int_) for m in masks]
            bg_weight = [np.count_nonzero(m==0)/m.size for m in _masks]
            fg_weight = [np.count_nonzero(m==1)/m.size for m in _masks]
            self.class_weights = np.asarray([1/np.mean(bg_weight), 1/np.mean(fg_weight)])
            self.class_weights = self.class_weights/np.mean(self.class_weights)/2
            

        # Expand ROIS to samples num
        for i in range(len(ROIS)):
            ROIS[i] = np.repeat(ROIS[i], repeats=videos[i].shape[0], axis=0)

        if scale is True:
            self.videos = [v.astype(np.float)/255.0 for v in videos]
        else:
            self.videos = videos

        self.masks = masks
        self.rois = ROIS

        # Split in train / val / test
        if all_in_val is False:
            if self.split == -1:
                self.train_cutoff = 1.0
                self.val_cutoff = 1.0
            else:

                self.train_cutoff = 0.4
                self.val_cutoff = 1.0
        else:
            self.train_cutoff = 0.0
            self.val_cutoff = 1.0

        self.update_split(self.split)

        self.shuffle()

        self.current_idx_train = 0
        self.current_idx_val = 0
        self.current_idx_test = 0

        self.n_categories = len(videos)




    def update_split(self, split):
        self.train = []
        self.eval = []
        self.test = []

        self.train_patches = []
        self.eval_patches = []
        self.test_patches = []

        # Creating arrays of clips indexes as well as patches indexes

        for i in range(len(self.videos)):
            n_clips = self.videos[i].shape[0]
            idx_clips = np.arange(0, n_clips)
            np.random.shuffle(idx_clips)
            if split == -1:
                pass

                '''idx_clips = list(idx_clips)
                # Make sure that the mandatory training clips are in the training category. Simply put them at the beginning of the list
                for include_idx in self.include_in_train_idx[i]:
                    idx_clips.insert(0, idx_clips.pop(idx_clips.index(include_idx)))
                idx_clips = np.asarray(idx_clips)'''

            else:
                # Roll the train indexes, 10 being a complete rotation
                idx_clips = np.roll(idx_clips, -int(np.round(len(idx_clips) / 10 * split)))

            train_idx = idx_clips[:int(n_clips * self.train_cutoff)]
            eval_idx = idx_clips[int(n_clips * self.train_cutoff):int(n_clips * self.val_cutoff)]
            test_idx = idx_clips[int(n_clips * self.val_cutoff):]
            self.train.extend([(i, idx) for idx in train_idx])
            self.eval.extend([(i, idx) for idx in eval_idx])
            self.test.extend([(i, idx) for idx in test_idx])

        self.train_samples = len(self.train)
        self.eval_samples = len(self.eval)
        self.test_samples = len(self.test)

        print(self.train_samples, self.eval_samples, self.test_samples)

   

    def load_clip(self, path, clip):

        clip_path = os.path.join(path, clip)
        assert os.path.exists(clip_path), 'Error: clip {} does not exist'.format(clip)

        # Getting temporal ROI
        with open(os.path.join(clip_path, 'temporalROI.txt'), 'r') as myfile:
            line = myfile.readlines()[0].strip().split(' ')
            start = int(line[0])
            end = int(line[1])

        # Getting spatial ROI
        ROI = cv2.imread(os.path.join(clip_path, 'ROI.bmp'), cv2.IMREAD_GRAYSCALE)/255

        frames = []
        masks = []
        for counter in range(start, end+1):
            frame_path = os.path.join(clip_path, 'input', 'in{:06d}.jpg'.format(counter))
            mask_path = os.path.join(clip_path, 'groundtruth', 'gt{:06d}.png'.format(counter))
            frames.append(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY))
            masks.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype('uint8'))

        clip = np.asarray(frames)
        masks = np.asarray(masks, dtype='uint8')

        return clip, masks, ROI, start

    def load_image(self, path, clip, index):

        clip_path = os.path.join(path, clip)
        assert os.path.exists(clip_path), 'Error: clip {} does not exist'.format(clip)
        frame_path = os.path.join(clip_path, 'bg', '{:06d}.png'.format(index))
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        return frame

    def shuffle(self):

        if self.seed is not None: np.random.seed(self.seed)
        indices = np.random.permutation(len(self.train))
        self.train = [self.train[i] for i in indices]

        indices = np.random.permutation(len(self.train_patches))
        self.train_patches = [self.train_patches[i] for i in indices]


    # Yields random clips from the training set (full video frames)
    def load_batch_train(self, batch_size):
        if self.current_idx_train + batch_size > self.train_samples:
            self.shuffle()
            self.current_idx_train = 0

        batch_indices = self.train[self.current_idx_train:self.current_idx_train + batch_size]
        rnd_seed = np.random.randint(2147483647)
        torch.manual_seed(rnd_seed)
        if self.flatten == False:
            batch_D = torch.stack([self.transform_train(self.videos[elem[0]][elem[1]].transpose(1,2,0)) for elem in batch_indices], dim=0)
        else:
            batch_D = torch.stack([self.transform_train(self.videos[elem[0]][elem[1]].transpose(1,0)) for elem in batch_indices], dim=0)
        torch.manual_seed(rnd_seed)
        # batch_M = torch.stack([self.transform_train_mask(self.masks[elem[0]][elem[1]].transpose(1,2,0)) for elem in batch_indices], dim=0)
        batch_M = torch.stack([torch.Tensor(self.masks[elem[0]][elem[1]]) for elem in batch_indices], dim=0)
        torch.manual_seed(rnd_seed)
        batch_roi = torch.cat([self.transform_train_mask(self.rois[elem[0]][elem[1]]) for elem in batch_indices], dim=0)
        batch = (batch_D, batch_M, batch_roi)
        self.current_idx_train += batch_size
        return batch



    def load_batch_validation(self, batch_size):
        if self.current_idx_val + batch_size > self.eval_samples:
            self.current_idx_val = 0
            return ([], [], [])
        # batch = (self.val_clips[self.current_idx_val: self.current_idx_val + batch_size, ...],
        #          self.val_masks[self.current_idx_val: self.current_idx_val + batch_size, ...],
        #          self.val_backgrounds[self.current_idx_val: self.current_idx_val + batch_size, ...],
        #          self.val_rois[self.current_idx_val: self.current_idx_val + batch_size, ...])
        batch_indices = self.eval[self.current_idx_val:self.current_idx_val + batch_size]
        rnd_seed = np.random.randint(2147483647)
        torch.manual_seed(rnd_seed)
        if self.flatten == False:
            batch_D = torch.stack([self.transform_train(self.videos[elem[0]][elem[1]].transpose(1,2,0)) for elem in batch_indices], dim=0)
        else:
            batch_D = torch.stack([self.transform_train(self.videos[elem[0]][elem[1]].transpose(1,0)) for elem in batch_indices], dim=0)
        if len(batch_D.shape) == 3: batch_D = batch_D.unsqueeze(0)
        torch.manual_seed(rnd_seed)
        # batch_M = torch.stack(
        #     [self.transform_eval_mask(self.masks[elem[0]][elem[1]].transpose(1, 2, 0)) for elem in batch_indices], dim=0)
        batch_M = torch.stack(
            [torch.Tensor(self.masks[elem[0]][elem[1]]) for elem in batch_indices], dim=0)
        if len(batch_M.shape) == 3: batch_M = batch_M.unsqueeze(0)
        torch.manual_seed(rnd_seed)
        batch_roi = torch.cat(
            [self.transform_eval_mask(self.rois[elem[0]][elem[1]]) for elem in batch_indices], dim=0)
        if len(batch_roi.shape) == 2: batch_roi = batch_roi.unsqueeze(0)
        batch = (batch_D, batch_M, batch_roi)
        self.current_idx_val += batch_size
        return batch


    def load_clip_from_category_eval(self, cat_id, clip_idx):
        # Returns a specific clip from a specific video sequence (useful for full video evaluation)
        rnd_seed = np.random.randint(2147483647)

        if cat_id>=len(self.videos):
            return None
        if clip_idx>=len(self.videos[cat_id]):
            return ([], [], [])

        torch.manual_seed(rnd_seed)

        if self.flatten == False:
            batch_D = torch.stack([self.transform_eval(self.videos[cat_id][clip_idx].transpose(1, 2, 0))], dim=0)
        else:
            batch_D = torch.stack([self.transform_eval(self.videos[cat_id][clip_idx].transpose(1,0))], dim=0)
        torch.manual_seed(rnd_seed)
        batch_M = torch.stack([torch.Tensor(self.masks[cat_id][clip_idx])], dim=0)
        torch.manual_seed(rnd_seed)
        batch_roi = torch.cat([self.transform_eval_mask(self.rois[cat_id][clip_idx])], dim=0)
        batch = (batch_D, batch_M, batch_roi)
        return batch


    def load_batch_test(self, batch_size):
        if self.current_idx_test + batch_size > self.test_samples:
            self.current_idx_test = 0
            return ([], [], [])
        batch_indices = self.test[self.current_idx_test:self.current_idx_test + batch_size]
        rnd_seed = np.random.randint(2147483647)
        torch.manual_seed(rnd_seed)
        batch_D = torch.stack(
            [self.transform_eval(self.videos[elem[0]][elem[1]].transpose(1, 2, 0)) for elem in batch_indices], dim=0)
        torch.manual_seed(rnd_seed)
        # batch_M = torch.stack(
        #     [self.transform_eval_mask(self.masks[elem[0]][elem[1]].transpose(1, 2, 0)) for elem in batch_indices], dim=0)
        batch_M = torch.stack(
            [torch.Tensor(self.masks[elem[0]][elem[1]]) for elem in batch_indices],
            dim=0)
        torch.manual_seed(rnd_seed)
        batch_roi = torch.cat(
            [self.transform_eval_mask(self.rois[elem[0]][elem[1]]) for elem in batch_indices], dim=0)
        batch = (batch_D, batch_M, batch_roi)
        self.current_idx_val += batch_size
        return batch


if __name__ == '__main__':
    # d = Loader('../../../CDnet2014/dataset2014/dataset', ['lowFramerate/port_0_17fps'],
    #            20, flatten=False, scale=True, maxsize=320, patch_size=None, crop_size=128, preprocess_mask=True)

    d = Loader('../../../CDnet2014/dataset2014/dataset', ['lowFramerate/port_0_17fps'], time_steps=20,
                                          flatten=False, scale=False, maxsize=320,
                                          patch_size=32, seed=123, crop_size=None,
                                          preprocess_mask=True, include_in_train=[[]],
                                          split=-1)
