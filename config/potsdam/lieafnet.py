"""
Training/eval config for LiEAF-Net on ISPRS Potsdam.
Reproduces the paper's main result: 86.45% mIoU / 92.58% mF1 / 91.39% OA.

Run from the repo root, e.g.:
    python train_supervision.py -c config/potsdam/lieafnet.py
    python potsdam_test.py -c config/potsdam/lieafnet.py -o fig_results/potsdam/lieafnet --rgb

See README.md "Data preparation" for how to populate data/potsdam/.
"""
from pathlib import Path
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.potsdam_dataset_ import *
from geoseg.models.lieafnet import LiEAFNet
from tools.utils import Lookahead, process_model_params

REPO_ROOT = Path(__file__).resolve().parents[2]

max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 2
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

model_name = "lieafnet"
weights_name = f"{model_name}-potsdam-512-e105"
weights_path = str(REPO_ROOT / "model_weights" / "potsdam" / weights_name)
test_weights_name = weights_name
log_name = "potsdam/{}".format(weights_name)
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = [0]  # edit to match your available GPU index/indices
resume_ckpt_path = None

net = LiEAFNet(n_class=num_classes)

loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = False

DROOT = str(REPO_ROOT / "data" / "potsdam")
train_dataset = PotsdamDataset(
    data_root=DROOT + "/train", mode="train",
    img_dir="images_1024", mask_dir="masks_1024", dsm_dir="dsm_1024",
    mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(
    data_root=DROOT + "/valid", mode="val",
    img_dir="images_1024", mask_dir="masks_1024", dsm_dir="dsm_1024",
    transform=val_aug)

test_dataset = PotsdamDataset(
    data_root=DROOT + "/test", mode="test",
    img_dir="images_1024", mask_dir="masks_1024", dsm_dir="dsm_1024",
    transform=val_aug)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size,
                          num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
val_loader   = DataLoader(dataset=val_dataset,   batch_size=val_batch_size,
                          num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=val_batch_size,
                          num_workers=4, shuffle=False, pin_memory=True, drop_last=False)

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
