import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0]) # label 1
LowVeg = np.array([255, 255, 0]) # label 2
Tree = np.array([0, 255, 0]) # label 3
Car = np.array([0, 255, 255]) # label 4
Clutter = np.array([0, 0, 255]) # label 5
Boundary = np.array([0, 0, 0]) # label 6
num_classes = 6


# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="data/vaihingen/train_images")
    parser.add_argument("--mask-dir", default="data/vaihingen/train_masks")
    parser.add_argument("--dsm-dir", default="data/vaihingen/train_dsm")  # ADD THIS
    parser.add_argument("--output-img-dir", default="data/vaihingen/train/images_1024")
    parser.add_argument("--output-mask-dir", default="data/vaihingen/train/masks_1024")
    parser.add_argument("--output-dsm-dir", default="data/vaihingen/train/dsm_1024")  # ADD THIS
    parser.add_argument("--eroded", action='store_true')
    parser.add_argument("--gt", action='store_true')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--val-scale", type=float, default=1.0)
    parser.add_argument("--split-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    return parser.parse_args()


def get_img_mask_dsm_padded(image, mask, dsm, patch_size, mode):  # MODIFIED
    img, mask, dsm = np.array(image), np.array(mask), np.array(dsm)  # MODIFIED
    oh, ow = img.shape[0], img.shape[1]
    rh, rw = oh % patch_size, ow % patch_size
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh

    h, w = oh + height_pad, ow + width_pad
    pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT, value=0)(image=img)
    pad_mask = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT, value=6)(image=mask)
    pad_dsm = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT, value=0)(image=dsm)  # ADD THIS
    
    img_pad, mask_pad, dsm_pad = pad_img['image'], pad_mask['image'], pad_dsm['image']  # MODIFIED
    img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)
    mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)
    # DSM is grayscale, no color conversion needed
    return img_pad, mask_pad, dsm_pad  # MODIFIED


def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def car_color_replace(mask):
    mask = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]
    return mask


def rgb_to_2D_label(_label):
    _label = _label.transpose(2, 0, 1)
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)
    label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
    label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 6
    return label_seg


def image_augment(image, mask, dsm, patch_size, mode='train', val_scale=1.0):  # MODIFIED
    image_list = []
    mask_list = []
    dsm_list = []  # ADD THIS
    image_width, image_height = image.size[1], image.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]
    dsm_width, dsm_height = dsm.size[1], dsm.size[0]  # ADD THIS

    assert image_height == mask_height == dsm_height and image_width == mask_width == dsm_width  # MODIFIED
    
    if mode == 'train':
        h_vlip = RandomHorizontalFlip(p=1.0)
        v_vlip = RandomVerticalFlip(p=1.0)
        
        image_h_vlip, mask_h_vlip, dsm_h_vlip = h_vlip(image.copy()), h_vlip(mask.copy()), h_vlip(dsm.copy())  # MODIFIED
        image_v_vlip, mask_v_vlip, dsm_v_vlip = v_vlip(image.copy()), v_vlip(mask.copy()), v_vlip(dsm.copy())  # MODIFIED

        image_list_train = [image, image_h_vlip, image_v_vlip]
        mask_list_train = [mask, mask_h_vlip, mask_v_vlip]
        dsm_list_train = [dsm, dsm_h_vlip, dsm_v_vlip]  # ADD THIS
        
        for i in range(len(image_list_train)):
            image_tmp, mask_tmp, dsm_tmp = get_img_mask_dsm_padded(image_list_train[i], mask_list_train[i], 
                                                                     dsm_list_train[i], patch_size, mode)  # MODIFIED
            mask_tmp = rgb_to_2D_label(mask_tmp.copy())
            image_list.append(image_tmp)
            mask_list.append(mask_tmp)
            dsm_list.append(dsm_tmp)  # ADD THIS
    else:
        rescale = Resize(size=(int(image_width * val_scale), int(image_height * val_scale)))
        image, mask, dsm = rescale(image.copy()), rescale(mask.copy()), rescale(dsm.copy())  # MODIFIED
        image, mask, dsm = get_img_mask_dsm_padded(image.copy(), mask.copy(), dsm.copy(), patch_size, mode)  # MODIFIED
        mask = rgb_to_2D_label(mask.copy())
        image_list.append(image)
        mask_list.append(mask)
        dsm_list.append(dsm)  # ADD THIS
        
    return image_list, mask_list, dsm_list  # MODIFIED


def randomsizedcrop(image, mask, dsm):  # MODIFIED
    h, w = image.shape[0], image.shape[1]
    crop = albu.RandomSizedCrop(min_max_height=(int(3*h//8), int(h//2)), width=h, height=w)(
        image=image.copy(), masks=[mask.copy(), dsm.copy()])  # MODIFIED
    img_crop = crop['image']
    mask_crop = crop['masks'][0]
    dsm_crop = crop['masks'][1]  # ADD THIS
    return img_crop, mask_crop, dsm_crop  # MODIFIED


def car_aug(image, mask, dsm):  # MODIFIED
    assert image.shape[:2] == mask.shape == dsm.shape  # MODIFIED
    v_flip = albu.VerticalFlip(p=1.0)(image=image.copy(), masks=[mask.copy(), dsm.copy()])  # MODIFIED
    h_flip = albu.HorizontalFlip(p=1.0)(image=image.copy(), masks=[mask.copy(), dsm.copy()])  # MODIFIED
    rotate_90 = albu.RandomRotate90(p=1.0)(image=image.copy(), masks=[mask.copy(), dsm.copy()])  # MODIFIED
    
    image_vflip, mask_vflip, dsm_vflip = v_flip['image'], v_flip['masks'][0], v_flip['masks'][1]  # MODIFIED
    image_hflip, mask_hflip, dsm_hflip = h_flip['image'], h_flip['masks'][0], h_flip['masks'][1]  # MODIFIED
    image_rotate, mask_rotate, dsm_rotate = rotate_90['image'], rotate_90['masks'][0], rotate_90['masks'][1]  # MODIFIED
    
    image_list = [image, image_vflip, image_hflip, image_rotate]
    mask_list = [mask, mask_vflip, mask_hflip, mask_rotate]
    dsm_list = [dsm, dsm_vflip, dsm_hflip, dsm_rotate]  # ADD THIS

    return image_list, mask_list, dsm_list  # MODIFIED


def vaihingen_format(inp):
    (img_path, mask_path, dsm_path, imgs_output_dir, masks_output_dir, dsms_output_dir, 
     eroded, gt, mode, val_scale, split_size, stride) = inp  # MODIFIED
    
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    dsm_filename = os.path.splitext(os.path.basename(dsm_path))[0]  # ADD THIS
    
    if eroded:
        mask_path = mask_path[:-4] + '_noBoundary.tif'
    
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    dsm = Image.open(dsm_path).convert('L')  # ADD THIS - Load as grayscale
    
    if gt:
        mask_ = car_color_replace(mask)
        out_origin_mask_path = os.path.join(masks_output_dir + '/origin/', "{}.tif".format(mask_filename))
        cv2.imwrite(out_origin_mask_path, mask_)
    
    image_list, mask_list, dsm_list = image_augment(image=img.copy(), mask=mask.copy(), dsm=dsm.copy(),
                                                      patch_size=split_size, mode=mode, val_scale=val_scale)  # MODIFIED
    
    assert img_filename == mask_filename == dsm_filename and len(image_list) == len(mask_list) == len(dsm_list)  # MODIFIED
    
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]
        dsm = dsm_list[m]  # ADD THIS
        
        assert img.shape[0] == mask.shape[0] == dsm.shape[0] and img.shape[1] == mask.shape[1] == dsm.shape[1]  # MODIFIED
        
        if gt:
            mask = pv2rgb(mask)

        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                img_tile = img[y:y + split_size, x:x + split_size]
                mask_tile = mask[y:y + split_size, x:x + split_size]
                dsm_tile = dsm[y:y + split_size, x:x + split_size]  # ADD THIS

                if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size \
                        and mask_tile.shape[0] == split_size and mask_tile.shape[1] == split_size \
                        and dsm_tile.shape[0] == split_size and dsm_tile.shape[1] == split_size:  # MODIFIED
                    
                    image_crop, mask_crop, dsm_crop = randomsizedcrop(img_tile, mask_tile, dsm_tile)  # MODIFIED
                    bins = np.array(range(num_classes + 1))
                    class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)
                    cf = class_pixel_counts / (mask_crop.shape[0] * mask_crop.shape[1])
                    
                    if cf[4] > 0.1 and mode == 'train':
                        car_imgs, car_masks, car_dsms = car_aug(image_crop, mask_crop, dsm_crop)  # MODIFIED
                        for i in range(len(car_imgs)):
                            out_img_path = os.path.join(imgs_output_dir,
                                                        "{}_{}_{}_{}.tif".format(img_filename, m, k, i))
                            cv2.imwrite(out_img_path, car_imgs[i])

                            out_mask_path = os.path.join(masks_output_dir,
                                                         "{}_{}_{}_{}.png".format(mask_filename, m, k, i))
                            cv2.imwrite(out_mask_path, car_masks[i])
                            
                            out_dsm_path = os.path.join(dsms_output_dir,
                                                        "{}_{}_{}_{}.png".format(dsm_filename, m, k, i))  # ADD THIS
                            cv2.imwrite(out_dsm_path, car_dsms[i])  # ADD THIS
                    else:
                        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.tif".format(img_filename, m, k))
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(mask_filename, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)
                        
                        out_dsm_path = os.path.join(dsms_output_dir, "{}_{}_{}.png".format(dsm_filename, m, k))  # ADD THIS
                        cv2.imwrite(out_dsm_path, dsm_tile)  # ADD THIS

                k += 1


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    dsms_dir = args.dsm_dir  # ADD THIS
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    dsms_output_dir = args.output_dsm_dir  # ADD THIS
    gt = args.gt
    eroded = args.eroded
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride
    
    img_paths = glob.glob(os.path.join(imgs_dir, "*.tif"))
    mask_paths_raw = glob.glob(os.path.join(masks_dir, "*.tif"))
    dsm_paths = glob.glob(os.path.join(dsms_dir, "*.jpg"))  # ADD THIS
    
    if eroded:
        mask_paths = [(p[:-15] + '.tif') for p in mask_paths_raw]
    else:
        mask_paths = mask_paths_raw
    
    img_paths.sort()
    mask_paths.sort()
    dsm_paths.sort()  # ADD THIS

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        if gt:
            os.makedirs(masks_output_dir+'/origin')
    if not os.path.exists(dsms_output_dir):  # ADD THIS
        os.makedirs(dsms_output_dir)  # ADD THIS

    inp = [(img_path, mask_path, dsm_path, imgs_output_dir, masks_output_dir, dsms_output_dir, 
            eroded, gt, mode, val_scale, split_size, stride)
           for img_path, mask_path, dsm_path in zip(img_paths, mask_paths, dsm_paths)]  # MODIFIED

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(vaihingen_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))