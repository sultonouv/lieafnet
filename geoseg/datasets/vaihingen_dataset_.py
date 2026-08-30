import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu

import matplotlib.patches as mpatches
from PIL import Image
import random
from .transform import *

CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]]

ORIGIN_IMG_SIZE = (512, 512)  # CHANGED from 1024
INPUT_IMG_SIZE = (512, 512)   # CHANGED from 1024
TEST_IMG_SIZE = (512, 512)    # CHANGED from 1024


def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask, dsm):  # MODIFIED - added dsm parameter
    crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75,
                                    ignore_index=len(CLASSES), nopad=False)])
    img, mask, dsm = crop_aug(img, mask, dsm)  # MODIFIED
    img, mask, dsm = np.array(img), np.array(mask), np.array(dsm)  # MODIFIED
    
    # Apply same augmentation to img, mask, and dsm
    aug = get_training_transform()(image=img.copy(), masks=[mask.copy(), dsm.copy()])  # MODIFIED
    img = aug['image']
    mask = aug['masks'][0]
    dsm = aug['masks'][1]  # ADD
    
    return img, mask, dsm  # MODIFIED


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask, dsm):  # MODIFIED - added dsm parameter
    img, mask, dsm = np.array(img), np.array(mask), np.array(dsm)  # MODIFIED
    aug = get_val_transform()(image=img.copy(), masks=[mask.copy(), dsm.copy()])  # MODIFIED
    img = aug['image']
    mask = aug['masks'][0]
    dsm = aug['masks'][1]  # ADD
    return img, mask, dsm  # MODIFIED


class VaihingenDataset(Dataset):
    def __init__(self, data_root='data/vaihingen/test', mode='val', 
                 img_dir='images_1024', mask_dir='masks_1024', dsm_dir='dsm_1024',  # MODIFIED - changed to 512 and added dsm_dir
                 img_suffix='.tif', mask_suffix='.png', dsm_suffix='.png',  # ADD dsm_suffix
                 transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dsm_dir = dsm_dir  # ADD
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.dsm_suffix = dsm_suffix  # ADD
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask, dsm = self.load_img_and_mask(index)  # MODIFIED
            if self.transform:
                img, mask, dsm = self.transform(img, mask, dsm)  # MODIFIED
        else:
            img, mask, dsm = self.load_mosaic_img_and_mask(index)  # MODIFIED
            if self.transform:
                img, mask, dsm = self.transform(img, mask, dsm)  # MODIFIED

        # Convert to tensor and concatenate RGB + DSM
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
        dsm = torch.from_numpy(dsm).unsqueeze(0).float()  # (1, H, W)
        img = torch.cat([img, dsm], dim=0)  # (4, H, W) - RGBD
        
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        dsm_name = osp.join(self.data_root, self.dsm_dir, img_id + self.dsm_suffix)  # ADD
        
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        dsm = Image.open(dsm_name).convert('L')  # ADD - load DSM as grayscale
        
        return img, mask, dsm  # MODIFIED

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a, dsm_a = self.load_img_and_mask(indexes[0])  # MODIFIED
        img_b, mask_b, dsm_b = self.load_img_and_mask(indexes[1])  # MODIFIED
        img_c, mask_c, dsm_c = self.load_img_and_mask(indexes[2])  # MODIFIED
        img_d, mask_d, dsm_d = self.load_img_and_mask(indexes[3])  # MODIFIED

        img_a, mask_a, dsm_a = np.array(img_a), np.array(mask_a), np.array(dsm_a)  # MODIFIED
        img_b, mask_b, dsm_b = np.array(img_b), np.array(mask_b), np.array(dsm_b)  # MODIFIED
        img_c, mask_c, dsm_c = np.array(img_c), np.array(mask_c), np.array(dsm_c)  # MODIFIED
        img_d, mask_d, dsm_d = np.array(img_d), np.array(mask_d), np.array(dsm_d)  # MODIFIED

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        # Apply crops to img, mask, and dsm
        croped_a = random_crop_a(image=img_a.copy(), masks=[mask_a.copy(), dsm_a.copy()])  # MODIFIED
        croped_b = random_crop_b(image=img_b.copy(), masks=[mask_b.copy(), dsm_b.copy()])  # MODIFIED
        croped_c = random_crop_c(image=img_c.copy(), masks=[mask_c.copy(), dsm_c.copy()])  # MODIFIED
        croped_d = random_crop_d(image=img_d.copy(), masks=[mask_d.copy(), dsm_d.copy()])  # MODIFIED

        img_crop_a, mask_crop_a, dsm_crop_a = croped_a['image'], croped_a['masks'][0], croped_a['masks'][1]  # MODIFIED
        img_crop_b, mask_crop_b, dsm_crop_b = croped_b['image'], croped_b['masks'][0], croped_b['masks'][1]  # MODIFIED
        img_crop_c, mask_crop_c, dsm_crop_c = croped_c['image'], croped_c['masks'][0], croped_c['masks'][1]  # MODIFIED
        img_crop_d, mask_crop_d, dsm_crop_d = croped_d['image'], croped_d['masks'][0], croped_d['masks'][1]  # MODIFIED

        # Concatenate images
        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        # Concatenate masks
        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        
        # Concatenate DSMs - ADD THIS
        top_dsm = np.concatenate((dsm_crop_a, dsm_crop_b), axis=1)
        bottom_dsm = np.concatenate((dsm_crop_c, dsm_crop_d), axis=1)
        dsm = np.concatenate((top_dsm, bottom_dsm), axis=0)
        
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        dsm = np.ascontiguousarray(dsm)  # ADD
        
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        dsm = Image.fromarray(dsm)  # ADD

        return img, mask, dsm  # MODIFIED


# Visualization functions remain mostly the same
def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith('.png')]
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        mask = cv2.imread(f'{mask_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert('P')
        mask.putpalette(np.array(PALETTE, dtype=np.uint8))
        mask = np.array(mask.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE ' + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('Mask True ' + seg_id)
        ax[i, 2].set_axis_off()
        ax[i, 2].imshow(img_seg)
        ax[i, 2].set_title('Mask Predict ' + seg_id)
        ax[i, 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_seg(seg_path, img_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith('.png')]
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE '+img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(img_seg)
        ax[i, 1].set_title('Seg IMAGE '+seg_id)
        ax[i, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    ax1.imshow(img)
    ax1.set_title('RS IMAGE ' + str(img_id)+'.tif')
    ax2.imshow(mask)
    ax2.set_title('Mask ' + str(img_id)+'.png')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')