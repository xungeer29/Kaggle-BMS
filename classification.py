# -*- coding:utf-8 -*-
import warnings 
warnings.filterwarnings('ignore')
import os, argparse, time
import random
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
import timm
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

from dataset import get_train_file_path, get_test_file_path
from utils import seed_everything, init_logger, AverageMeter, timeSince
from configs_cls import config as cfg

def sp_noise(image,prob):
  '''
  添加椒盐噪声
  prob:噪声比例 
  '''
  output = np.zeros(image.shape,np.uint8)
  thres = 1 - prob 
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output

def gasuss_noise(image, mean=0, var=0.001):
  ''' 
    添加高斯噪声
    mean : 均值 
    var : 方差
  '''
  image = np.array(image/255, dtype=float)
  noise = np.random.normal(mean, var ** 0.5, image.shape)
  out = image + noise
  if out.min() < 0:
    low_clip = -1.
  else:
    low_clip = 0.
  out = np.clip(out, low_clip, 1.0)
  out = np.uint8(out*255)
  #cv.imshow("gasuss", out)
  return out

class BMSDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.labels = df['InChI_text'].values
        self.transform = A.Compose([
            # A.Resize(cfg.size, cfg.size),
            # A.RandomResizedCrop(cfg.size, cfg.size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
            ])
        # self.imgs = self.load_to_memory()
    
    def __len__(self):
        return len(self.df)

    def load_to_memory(self):
        print('load to memory ...')
        imgs = []
        for i in range(len(self.df)):
            file_path = self.file_paths[i]
            image = cv2.imread(file_path)
            imgs.append(image)
        return imgs
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        # image = self.imgs[idx]
        h,w,c = image.shape
        h, w = min([h, w]), max([h, w])

        InChI = self.df['InChI'][idx]
        m = Chem.MolFromInchi(InChI)
        if m != None:
            img = Draw.MolToImage(m, size=(w*3, h*3))
            img = np.array(img)
            # kernel = np.ones((3, 3),np.uint8)  
            # img = cv2.dilate(img, kernel, iterations=1) # dilate erode
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
            if np.random.uniform(0,1) > 0.7:
                img = sp_noise(img, prob=0.0001)
            # cv2.imwrite('tmp.jpg', img)
            label = np.random.randint(0, 4)
            img = np.rot90(img, label)

            augmented = self.transform(image=img)
            img = augmented['image']

            return img, label
        else:
            idx_ = np.random.randint(0, len(self.df))
            self.__getitem__(idx_)

class TestDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.transform = A.Compose([
            # A.Resize(cfg.size, cfg.size),
            # A.RandomResizedCrop(cfg.size, cfg.size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        augmented = self.transform(image=img)
        img = augmented['image']

        return img, os.path.basename(file_path)
        
def train_one_epoch(train_loader, model, criterion, optimizer, epoch, lr_scheduler, device, scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        labels = labels.to(device)
        images = images.to(device)
        batch_size = images.size(0)
        # apex
        with autocast():
          logits = model(images)
          loss = criterion(logits, labels)
          pred = torch.argmax(logits, -1)
          acc = (pred==labels).sum() / pred.shape[0]

        # record loss
        losses.update(loss.item(), batch_size)
        accs.update(acc.item(), batch_size)
        if cfg.gradient_accumulation_steps > 1:
          loss = loss / cfg.gradient_accumulation_steps
        scaler.scale(loss).backward()
        # loss.backward()
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
          # encoder_optimizer.step()
          # decoder_optimizer.step()
          scaler.step(optimizer)
          scaler.update()

          optimizer.zero_grad()
          global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader)-1):
            info = f'Epoch: {epoch+1} {step}/{len(train_loader)} lr: {lr_scheduler.get_lr()[0]:.6f} '
            info += f'Loss: {losses.val:.4f}({losses.avg:.4f}) acc: {accs.val:.4f}({accs.avg:.4f}) '
            info += f'Elapsed: {timeSince(start, float(step+1)/len(train_loader))}'
            LOGGER.info(info)

            iters = step + epoch * len(train_loader)
            writer.add_scalar('train/loss.val', losses.val, iters)
            writer.add_scalar('train/acc.val', accs.avg, iters)
            writer.add_scalar('lr/encode', lr_scheduler.get_lr()[0], iters)

    return losses.avg, accs.avg

def evaluate(valid_loader, model, criterion, device):
    print('evaluate...')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    # switch to evaluation mode
    model.eval()
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            logits = model(images)
            loss = criterion(logits, labels)
            pred = torch.argmax(logits, -1)
            acc = (pred==labels).sum() / pred.shape[0]
        losses.update(loss.item(), batch_size)
        accs.update(acc.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

    # return text_preds, losses.avg
    return losses.avg, accs.avg

def inference(test_loader, model, device):
    print('inference...')
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, names) in tqdm(enumerate(test_loader)):
        # measure data loading time
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, -1)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds)

    return preds

def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")
    # loader
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['InChI'].values

    train_dataset = BMSDataset(train_folds)
    valid_dataset = BMSDataset(valid_folds)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size*4, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # model & optimizer
    model = timm.create_model(cfg.model_name, num_classes=4, pretrained=True)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=False)
    # optimizer = SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr, last_epoch=-1)

    criterion = nn.CrossEntropyLoss()

    best_acc = np.inf
    best_loss = np.inf

    # continue train
    # LOGGER.info(f'loading logs/{cfg.version}/fold0_last.pth')
    # states = torch.load(f'logs/{cfg.version}/fold0_last.pth', map_location=torch.device('cpu'))
    # encoder.load_state_dict(states['encoder'])
    # encoder.to(device)
    # decoder.load_state_dict(states['decoder'])
    # decoder.to(device)

    # apex
    scaler = GradScaler()
    for epoch in range(cfg.epochs):
        start_time = time.time()
        # train
        avg_loss, avg_acc = train_one_epoch(train_loader, model, criterion, optimizer, epoch, lr_scheduler, device, scaler)

        # eval
        loss_val, loss_acc = evaluate(valid_loader, model, criterion, device)
        lr_scheduler.step()

        elapsed = time.time() - start_time

        # LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Eval Epoch {epoch+1} Loss: {loss_val:.4f} acc: {loss_acc:.4f}')
        writer.add_scalar('eval/acc', loss_acc, epoch)
        writer.add_scalar('eval/loss', loss_val, epoch)

        torch.save({'state_dict': model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'lr_scheduler': lr_scheduler.state_dict(), 
                   },
                    f'logs/{cfg.version}/fold{fold}_last.pth')
        if loss_acc < best_acc:
            best_acc = loss_acc
            LOGGER.info(f'Epoch {epoch+1} - Save Best Acc: {best_acc:.4f} Model')
            torch.save({'state_dict': model.state_dict(), 
                        'optimizer': optimizer.state_dict(), 
                        'lr_scheduler': lr_scheduler.state_dict(), 
                       },
                        f'logs/{cfg.version}/fold{fold}_best.pth')

   
# ---------------------------------------------------------------------------------
seed_everything(seed=42)
parser = argparse.ArgumentParser(description='bms classification')
parser.add_argument('-v', '--version', default='debug', type=str)
parser.add_argument('--info', default='', type=str)
args = parser.parse_args()

cfg.version = f'cls_{args.version}'
os.makedirs(f'./logs/{cfg.version}/code', exist_ok=True)
os.system(f'cp classification.py ./logs/{cfg.version}/code/')
os.system(f'cp configs_cls.py ./logs/{cfg.version}/code/')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOGGER = init_logger(log_file=f'./logs/{cfg.version}/train.log')
writer = SummaryWriter(f'logs/{cfg.version}')

data_train = pd.read_pickle('data/train2.pkl')
data_train['file_path'] = data_train['image_id'].apply(get_train_file_path)

if 'debug' in cfg.version:
    cfg.epochs = 1
    data_train = data_train.sample(n=1000, random_state=cfg.seed).reset_index(drop=True)

folds = data_train.copy()
Fold = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['InChI_length'])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
print(folds.groupby(['fold']).size())

# train 
if cfg.train:
    # train
    for fold in range(cfg.n_fold):
        if fold in cfg.trn_fold:
            train_loop(folds, fold)
# exit()
# inference
test = pd.read_csv('data/sample_submission.csv')
test['file_path'] = test['image_id'].apply(get_test_file_path)

# print(f'test.shape: {test.shape}')
states = torch.load(f'logs/{cfg.version}/fold0_best.pth', map_location=torch.device('cpu'))
# states = torch.load('logs/v0/resnet34_fold0_best.pth', map_location=torch.device('cpu'))

model = timm.create_model(cfg.model_name, num_classes=4, pretrained=False)
model.load_state_dict(states['state_dict'])
model.to(device)

test_dataset = TestDataset(test)
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size*4, shuffle=False, num_workers=cfg.num_workers)
predictions = inference(test_loader, model, device)
# submission
test['rotate'] = predictions
test[['image_id', 'rotate']].to_csv(f'logs/{cfg.version}/test.csv', index=False)

train = pd.read_csv('data/train_labels.csv')
train['file_path'] = train['image_id'].apply(get_train_file_path)
train_dataset = TestDataset(train)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size*4, shuffle=False, num_workers=cfg.num_workers)
predictions = inference(train_loader, model, device)
train['rotate'] = predictions
train[['image_id', 'rotate']].to_csv(f'logs/{cfg.version}/train.csv', index=False)

