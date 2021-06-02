import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold


import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

import timm

import warnings 
warnings.filterwarnings('ignore')

from lr_schedulers import get_scheduler
from models import Encoder, Attention, DecoderWithAttention
from utils import get_score, init_logger, seed_everything, AverageMeter, timeSince
from dataset import preprocess, Tokenizer, get_train_file_path, BMSDataset, TestDataset, get_transforms, bms_collate, get_test_file_path
from configs import config as cfg


def train_one_epoch(train_loader, encoder, decoder, criterion, 
             encoder_optimizer, decoder_optimizer, epoch,
             encoder_scheduler, decoder_scheduler, start_step,
             best_score, device, scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    encoder.train()
    decoder.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels, label_lengths) in enumerate(train_loader):
        #if start_step+1 == len(train_loader):
        #    return 0
        #if step < start_step:
        #    continue
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = images.size(0)
        # apex
        with autocast():
          features = encoder(images.to(device))
          predictions, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, labels.to(device), label_lengths.to(device))
          targets = caps_sorted[:, 1:]
          predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
          targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
          loss = criterion(predictions, targets)

        # Add doubly stochastic attention regularization
        if cfg.alpha_c > 0:
          loss += cfg.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # record loss
        losses.update(loss.item(), batch_size)
        if cfg.gradient_accumulation_steps > 1:
          loss = loss / cfg.gradient_accumulation_steps
        scaler.scale(loss).backward()
        scaler.unscale_(encoder_optimizer)
        scaler.unscale_(decoder_optimizer)
        # loss.backward()
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
        decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg.max_grad_norm)
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
          # encoder_optimizer.step()
          # decoder_optimizer.step()
          scaler.step(encoder_optimizer)
          scaler.step(decoder_optimizer)
          scaler.update()

          encoder_optimizer.zero_grad()
          decoder_optimizer.zero_grad()
          global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader)-1):
            elr = encoder_optimizer.param_groups[0]['lr']
            dlr = decoder_optimizer.param_groups[0]['lr']
            info = f'{epoch+1}/{cfg.epochs} {step}/{len(train_loader)} encode_lr: {elr:.6f} decode_lr: {dlr:.6f} '
            info += f'Loss: {losses.val:.4f}({losses.avg:.4f}) Encoder Grad: {encoder_grad_norm:.4f} Decoder Grad: {decoder_grad_norm:.4f} '
            info += f'Elapsed: {timeSince(start, float(step+1)/len(train_loader))}'
            LOGGER.info(info)

            iters = step + epoch * len(train_loader)
            writer.add_scalar('train/loss.val', losses.val, iters)
            writer.add_scalar('train/loss.avg', losses.avg, iters)
            writer.add_scalar('train/encode_grad', encoder_grad_norm, iters)
            writer.add_scalar('train/decode_grad', decoder_grad_norm, iters)
            writer.add_scalar('lr/encode', elr, iters)
            writer.add_scalar('lr/decode', dlr, iters)
        encoder_scheduler.step()
        decoder_scheduler.step()

        if (step+1) % 1000 == 0 or (step+1) == train_loader:
            torch.save({'encoder': encoder.state_dict(), 
                        'encoder_optimizer': encoder_optimizer.state_dict(), 
                        'encoder_scheduler': encoder_scheduler.state_dict(), 
                        'decoder': decoder.state_dict(), 
                        'decoder_optimizer': decoder_optimizer.state_dict(), 
                        'decoder_scheduler': decoder_scheduler.state_dict(), 
                        'epoch': epoch,
                        'best_score': best_score,
                        'step': step,
                       },
                    f'logs/{cfg.version}/fold{fold}_last.pth')

    return losses.avg


def valid_fn(valid_loader, encoder, decoder, tokenizer, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluation mode
    encoder.eval()
    decoder.eval()
    text_preds = []
    start = end = time.time()
    for step, (images) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            features = encoder(images)
            predictions = decoder.predict(features, CFG.max_len, tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        _text_preds = tokenizer.predict_captions(predicted_sequence)
        text_preds.append(_text_preds)
        
    text_preds = np.concatenate(text_preds)
    return text_preds

def evaluate(valid_loader, encoder, decoder, tokenizer, criterion, device):
    print('evaluate...')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluation mode
    encoder.eval()
    decoder.eval()
    text_preds = []
    start = end = time.time()
    for step, (images, labels, label_lengths) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            features = encoder(images)
            predictions = decoder.predict(features, cfg.max_len, tokenizer)

            # predictions, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, labels.to(device), label_lengths.to(device))

        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        _text_preds = tokenizer.predict_captions(predicted_sequence)
        text_preds.append(_text_preds)

        # targets = caps_sorted[:, 1:]
        # predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
        # targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        # loss = criterion(predictions, targets)
        # # Add doubly stochastic attention regularization
        # # loss += cfg.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        # losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    text_preds = np.concatenate(text_preds)

    # return text_preds, losses.avg
    return text_preds, 0

def inference(test_loader, encoder, decoder, tokenizer, device):
    encoder.eval()
    decoder.eval()
    text_preds = []
    tk0 = tqdm(test_loader, total=len(test_loader))
    for images in tk0:
        images = images.to(device)
        with torch.no_grad():
            features = encoder(images)
            predictions = decoder.predict(features, cfg.max_len, tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        _text_preds = tokenizer.predict_captions(predicted_sequence)
        text_preds.append(_text_preds)
    text_preds = np.concatenate(text_preds)
    return text_preds

def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")
    # loader
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['InChI'].values

    train_dataset = BMSDataset(train_folds, tokenizer, transform=get_transforms(data='train', cfg=cfg))
    valid_dataset = BMSDataset(valid_folds, tokenizer, transform=get_transforms(data='valid', cfg=cfg))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True, collate_fn=bms_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size*4, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, collate_fn=bms_collate)

    # model & optimizer
    encoder = Encoder(cfg.model_name, pretrained=True)
    encoder.to(device)
    encoder_optimizer = Adam(encoder.parameters(), lr=cfg.encoder_lr/100 if cfg.warmup else cfg.encoder_lr, 
                              weight_decay=cfg.weight_decay, amsgrad=False)
    # encoder_optimizer = SGD(encoder.parameters(), lr=cfg.encoder_lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    cfg.steps = len(train_loader)
    encoder_scheduler = get_scheduler(encoder_optimizer, cfg)
    
    decoder = DecoderWithAttention(attention_dim=cfg.attention_dim,
                                   embed_dim=cfg.embed_dim,
                                   decoder_dim=cfg.decoder_dim,
                                   vocab_size=len(tokenizer),
                                   encoder_dim=encoder.n_features,
                                   dropout=cfg.dropout,
                                   device=device,
                                   rnn=cfg.rnn)
    decoder.to(device)
    decoder_optimizer = Adam(decoder.parameters(), lr=cfg.decoder_lr/100 if cfg.warmup else cfg.decoder_lr, 
                             weight_decay=cfg.weight_decay, amsgrad=False)
    # decoder_optimizer = SGD(decoder.parameters(), lr=cfg.decoder_lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    decoder_scheduler = get_scheduler(decoder_optimizer, cfg)

    # loop
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])

    best_score = np.inf
    best_loss = np.inf
    start_ep, start_step = 0, 0

    # continue train
    if cfg.continue_train:
      if os.path.exists(f'logs/{cfg.version}/fold{fold}_last.pth'):
        LOGGER.info(f'loading logs/{cfg.version}/fold{fold}_last.pth')
        states = torch.load(f'logs/{cfg.version}/fold{fold}_last.pth', map_location=torch.device('cpu'))
        encoder.load_state_dict(states['encoder'])
        encoder.to(device)
        decoder.load_state_dict(states['decoder'])
        decoder.to(device)
        encoder_scheduler.load_state_dict(states['encoder_scheduler'])
        decoder_scheduler.load_state_dict(states['decoder_scheduler'])
        encoder_optimizer.load_state_dict(states['encoder_optimizer'])
        decoder_optimizer.load_state_dict(states['decoder_optimizer'])
        start_ep = states['epoch']
        best_score = states['best_score']
        start_step = states['step'] if 'step' in states.keys() else start_step
        # start_ep = 1
        # best_score = 18.6637
        print(f'train from {start_ep} epoch, {start_step} step')
      else:
        LOGGER.info(f'logs/{cfg.version}/fold{fold}_last.pth is not exist. Train from epoch=0')

    # apex
    scaler = GradScaler()
    for epoch in range(start_ep, cfg.epochs):
        start_time = time.time()
        # train
        avg_loss = train_one_epoch(train_loader, encoder, decoder, criterion, 
                            encoder_optimizer, decoder_optimizer, epoch, 
                            encoder_scheduler, decoder_scheduler, start_step, best_score, device, scaler)

        # eval
        text_preds, loss_val = evaluate(valid_loader, encoder, decoder, tokenizer, criterion, device)
        text_preds = [f"InChI=1S/{text}" for text in text_preds]        
        # scoring
        score = get_score(valid_labels, text_preds)

        elapsed = time.time() - start_time

        # LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Eval Epoch {epoch+1} Loss: {loss_val:.4f} Score: {score:.4f}')
        writer.add_scalar('eval/score', score, epoch)
        writer.add_scalar('eval/loss', loss_val, epoch)

        if score < best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'encoder': encoder.state_dict(), 
                        'encoder_optimizer': encoder_optimizer.state_dict(), 
                        'encoder_scheduler': encoder_scheduler.state_dict(), 
                        'decoder': decoder.state_dict(), 
                        'decoder_optimizer': decoder_optimizer.state_dict(), 
                        'decoder_scheduler': decoder_scheduler.state_dict(), 
                        'epoch': epoch,
                        'best_score': best_score,
                        'step': 0,
                       },
                        f'logs/{cfg.version}/fold{fold}_best.pth')
        if score < 3.0:
            torch.save({'encoder': encoder.state_dict(), 
                        'encoder_optimizer': encoder_optimizer.state_dict(), 
                        'encoder_scheduler': encoder_scheduler.state_dict(), 
                        'decoder': decoder.state_dict(), 
                        'decoder_optimizer': decoder_optimizer.state_dict(), 
                        'decoder_scheduler': decoder_scheduler.state_dict(), 
                        'epoch': epoch,
                        'best_score': best_score,
                        'step': 0,
                       },
                        f'logs/{cfg.version}/fold{fold}_ep{epoch}_score={score:.4f}.pth')


seed_everything(seed=cfg.seed)

parser = argparse.ArgumentParser(description='bms')
parser.add_argument('-v', '--version', default='debug', type=str)
parser.add_argument('--info', default='', type=str)
args = parser.parse_args()
cfg.version = args.version

os.makedirs(f'./logs/{cfg.version}/code', exist_ok=True)
os.system(f'cp -r *.py ./logs/{cfg.version}/code/')

# preprocess
if not os.path.exists('data/train2.pkl'):
    preprocess()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOGGER = init_logger(log_file=f'./logs/{cfg.version}/train.log')
writer = SummaryWriter(f'logs/{cfg.version}')

data_train = pd.read_pickle('data/train2.pkl')
data_train['file_path'] = data_train['image_id'].apply(get_train_file_path)
LOGGER.info(f'data_train.shape: {data_train.shape}')

tokenizer = torch.load('data/tokenizer2.pth')
LOGGER.info(f"tokenizer.stoi: {tokenizer.stoi}")
assert tokenizer.stoi["<pad>"] == 192
max_length = data_train['InChI_length'].max()
LOGGER.info(f'max length: {max_length}')


if 'debug' in cfg.version:
    cfg.epochs = 2
    data_train = data_train.sample(n=1000, random_state=cfg.seed).reset_index(drop=True)

folds = data_train.copy()
Fold = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

idxes = {}
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['InChI_length'])):
    idxes[n] = [train_index, val_index]
    folds.loc[val_index, 'fold'] = int(n)
torch.save(idxes, f'logs/{cfg.version}/folds.pth')
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
states = torch.load(f'logs/{cfg.version}/fold1_best.pth', map_location=torch.device('cpu'))
# states = torch.load('logs/v0/resnet34_fold0_best.pth', map_location=torch.device('cpu'))

encoder = Encoder(cfg.model_name, pretrained=False)
encoder.load_state_dict(states['encoder'])
encoder.to(device)

decoder = DecoderWithAttention(attention_dim=cfg.attention_dim,
                               embed_dim=cfg.embed_dim,
                               decoder_dim=cfg.decoder_dim,
                               vocab_size=len(tokenizer),
                               encoder_dim=encoder.n_features,
                               dropout=cfg.dropout,
                               device=device,
                               rnn=cfg.rnn)
decoder.load_state_dict(states['decoder'])
decoder.to(device)

del states; gc.collect()

test_dataset = TestDataset(test, transform=get_transforms(data='valid', cfg=cfg))
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size*4, shuffle=False, num_workers=cfg.num_workers)
predictions = inference(test_loader, encoder, decoder, tokenizer, device)

del test_loader, encoder, decoder, tokenizer; gc.collect()

# submission
test['InChI'] = [f"InChI=1S/{text}" for text in predictions]
test[['image_id', 'InChI']].to_csv(f'logs/{cfg.version}/sub_fold1.csv', index=False)
