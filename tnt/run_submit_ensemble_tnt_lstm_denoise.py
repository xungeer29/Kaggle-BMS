import os
import gc

from common import *
from bms import *

from lib.net.lookahead import *
from lib.net.radam import *

from dataset_224 import *

import torch.nn as nn
from lstm.models import Encoder, Attention, DecoderWithAttention
from lstm.configs import config as cfg
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import torch.cuda.amp as amp
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



import cv2
import numpy as np


def denoise(src):
    shape = src.shape
    if len(shape) == 3:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    _, img_bw = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)

    img_opening = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    img_opening = 255 - img_opening
    kernel = np.ones((5, 5), np.float32) / (5 * 5)
    dst = cv2.filter2D(img_opening, -1, kernel)
    dst = 255 - dst
    _, dst_bw = cv2.threshold(dst, 240, 255, cv2.THRESH_BINARY)
    dst = src | dst_bw
    if len(shape) == 3:
        dst = np.stack((dst, dst, dst), axis=2) 
    return dst

def fast_remote_unrotate_augment(r):
    image = r['image']
    index = r['index']
    h, w = image.shape
    image = denoise(image)
    if h > w:
        image = np.rot90(image, 1)
        img_rot90 = np.rot90(image, -1)
    else:
        # Symmetrical image
        img = np.rot90(image, 2)
        loss = np.mean((img-image)**2)
        if loss < 0.01:
            img_rot90 = img
        else:
            img_rot90 = image

    # original entire image
    img0 = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    img0 = img0.astype(np.float16) / 255
    img0 = torch.from_numpy(img0).unsqueeze(0).repeat(3,1,1)

    # original entire image   rot90
    img1 = cv2.resize(img_rot90, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    img1 = img1.astype(np.float16) / 255
    img1 = torch.from_numpy(img1).unsqueeze(0).repeat(3,1,1)

    # crop2rect
    img_crop = crop_image_to_rect(image)
    img_crop = cv2.resize(img_crop, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    img_crop = img_crop.astype(np.float16) / 255
    img_crop = torch.from_numpy(img_crop).unsqueeze(0).repeat(3,1,1)

    # crop2rect rot90
    image1 = crop_image_to_rect(img_rot90)
    image1 = cv2.resize(image1, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    image1 = image1.astype(np.float16) / 255
    image1 = torch.from_numpy(image1).unsqueeze(0).repeat(3,1,1)

    # lstm
    img_lstm = np.stack((image, image, image), axis=2)
    img_lstm_rot = np.stack((img_rot90, img_rot90, img_rot90), axis=2)
    img_lstm_denoise = np.stack((img_denoise, img_denoise, img_denoise), axis=2)
    augs = A.Compose([
            A.Resize(cfg.size, cfg.size),
            # A.RandomResizedCrop(cfg.size, cfg.size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),])
    img_lstm = augs(image=img_lstm)['image']
    img_lstm_rot = augs(image=img_lstm_rot)['image']

    r={}
    r['img_crop'] = img_crop # orig
    r['img_crop_rot'] = image1

    r['img'] = img0
    r['img_rot'] = img1

    r['img_lstm'] = img_lstm
    r['img_lstm_rot'] = img_lstm_rot

    return r


def do_predict(nets, tokenizer, valid_loader, ckpts):
    print(f'ensemble {len(nets)} models!!')

    image_dim = 384
    text_dim = 384
    decoder_dim = 384
    num_layer = 3
    num_head = 8
    ff_dim = 1024

    STOI = {
        '<sos>': 190,
        '<eos>': 191,
        '<pad>': 192,
    }

    image_size = 224
    vocab_size = 193
    max_length = 300  # 275 300

    text = []
    start_timer = timer()
    valid_num = 0
    for t, batch in enumerate(valid_loader):
        img_crop = batch['img_crop'].cuda() # crop
        real_bs = img_crop.shape[0]

        # tta
        img_crop_rot = batch['img_crop_rot'].cuda() # entire
        img_crop = torch.cat((img_crop, img_crop_rot), dim=0)

        img = batch['img'].cuda() # entire
        img_rot = batch['img_rot'].cuda()
        img = torch.cat((img, img_rot), dim=0)

        batch_size = img_crop.shape[0]
        # lstm
        img_lstm = batch['img_lstm'].cuda()
        img_lstm_rot = batch['img_lstm_rot'].cuda()
        img_lstm = torch.cat((img_lstm, img_lstm_rot), dim=0)

        with torch.no_grad():
            with amp.autocast():
                device = img_crop.device
                batch_size = len(img_crop)
                token = torch.full((batch_size, max_length), STOI['<pad>'],dtype=torch.long, device=device)
                text_pos = nets[0].text_pos.pos
                token[:,0] = STOI['<sos>']
                eos = STOI['<eos>']
                pad = STOI['<pad>']

                # tnt
                image_embeds, incremental_states = [], []
                for i, net in enumerate(nets[:(len(ckpts[0]+ckpts[1]+ckpts[2]))]):
                    image_embed = net.cnn(img_crop) if i < len(ckpts[0]) else net.cnn(img)
                    image_embed = net.image_encode(image_embed).permute(1,0,2).contiguous()
                    incremental_state = {}
                    image_embeds.append(image_embed)
                    incremental_states.append(incremental_state)

                # lstm
                encoder_outs, hs, cs, embeddings = [], [], [], []
                for i, net in enumerate(nets[(len(ckpts[0]+ckpts[1]+ckpts[2])):]):
                    encoder_out = net['encoder'](img_lstm)
                    encoder_dim = encoder_out.size(-1)
                    vocab_size = net['decoder'].vocab_size
                    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
                    start_tockens = torch.ones(batch_size, dtype=torch.long).to(device) * STOI["<sos>"]
                    embedding = net['decoder'].embedding(start_tockens)
                    # initialize hidden state and cell state of LSTM cell
                    h, c = net['decoder'].init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
                    encoder_outs.append(encoder_out)
                    hs.append(h)
                    cs.append(c)
                    embeddings.append(embedding)

                for t in range(max_length-1):
                    last_token = token[:, t]

                    xs = []
                    # tnt
                    for i, net in enumerate(nets[:(len(ckpts[0]+ckpts[1]+ckpts[2]))]):
                        text_embed = net.token_embed(last_token)
                        text_embed = text_embed + text_pos[:,t] #
                        text_embed = text_embed.reshape(1,batch_size,text_dim)
                        x = net.text_decode.forward_one(text_embed, image_embeds[i], incremental_states[i])
                        x = x.reshape(batch_size, decoder_dim)
                        x = net.logit(x)
                        x = F.softmax(x, dim=1)
                        xs.append(x)

                    # lstm
                    for i, net in enumerate(nets[(len(ckpts[0]+ckpts[1]+ckpts[2])):]):
                        attention_weighted_encoding, alpha = net['decoder'].attention(encoder_outs[i], hs[i])
                        gate = net['decoder'].sigmoid(net['decoder'].f_beta(hs[i]))  # gating scalar, (batch_size_t, encoder_dim)
                        attention_weighted_encoding = gate * attention_weighted_encoding
                        h, c = net['decoder'].decode_step(torch.cat([embeddings[i], attention_weighted_encoding], dim=1), (hs[i], cs[i]))  # (batch_size_t, decoder_dim) LSTM
                        x = net['decoder'].fc(net['decoder'].dropout(h))  # (batch_size_t, vocab_size)
                        x = F.softmax(x, dim=1)
                        xs.append(x)
                        hs[i] = h
                        cs[i] = c

                    # ensemble
                    x = sum(xs) / len(xs)
                    # tta
                    assert x.shape[0] == batch_size
                    x = (x[:real_bs] + x[real_bs:2*real_bs]) / (batch_size/real_bs)
                    x = torch.cat((x, x), dim=0)

                    k = torch.argmax(x, -1)  # predict max
                    token[:, t+1] = k
                    if ((k == eos) | (k == pad)).all():  break
                    # lstm
                    for i, net in enumerate(nets[(len(ckpts[0]+ckpts[1]+ckpts[2])):]):
                        embeddings[i] = net['decoder'].embedding(k)

                predict = token[:real_bs, 1:]

                predict = predict.data.cpu().numpy()
                predict = tokenizer.predict_to_inchi(predict)
                text.extend(predict)

        valid_num += real_bs
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)

    assert(valid_num == len(valid_loader.dataset))
    print('')
    return text


def run_submit(args):
    gpu_no = int(os.environ['CUDA_VISIBLE_DEVICES'])
    print(gpu_no)

    ckpts00 = [
    'logs/tnt-s-224-crop/fold0/checkpoint/01276000_model.pth',
    'logs/tnt-s-224-crop/fold1/checkpoint/01513000_model.pth',
    'logs/tnt-s-224-crop/fold2/checkpoint/01294000_model.pth',
    'logs/tnt-s-224-crop/fold3/checkpoint/01871000_model.pth',
    'logs/tnt-s-224-crop/fold4/checkpoint/01251000_model.pth',
    'logs/tnt-s-224-crop/fold5/checkpoint/01168000_model.pth',
    ]
    ckpts01 = [
    'logs/tnt-s-224-fairseq/fold0/checkpoint/00910000_model.pth',
    'logs/tnt-s-224-fairseq/fold1/checkpoint/01027000_model.pth',
    'logs/tnt-s-224-fairseq/fold2/checkpoint/00659000_model.pth',
    'logs/tnt-s-224-fairseq/fold3/checkpoint/01117000_model-lb=2.31.pth',
    'logs/tnt-s-224-fairseq/fold4/checkpoint/00994000_model.pth',
    'logs/tnt-s-224-fairseq/fold5/checkpoint/00993000_model.pth',
    ]
    ckpts02 = [
    'logs/tnt-s-224-luozhengbo/fold3/checkpoint/01949000_model.pth',
    ]
    ckpts03 = [
    '../logs/lr/fold0_best.pth',
    '../logs/lr/fold1_best.pth',
    '../logs/lr/fold2_best.pth',
    '../logs/lr/fold4_best.pth',
    ]
    ckpts = [ckpts00, ckpts01, ckpts02, ckpts03]
    is_norm_ichi = False #True
    # out_dir = 'logs/ensemble/entire-crop-lstm'
    out_dir = 'logs/ensemble/tnt-crop224(012345)-entire(012345)-lstm-lr'
    os.makedirs(out_dir, exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('is_norm_ichi = %s\n' % is_norm_ichi)
    log.write('\n')

    #
    ## dataset ------------------------------------
    tokenizer = load_tokenizer()
    df_valid = make_fold('test')
    df_valid = df_valid.sort_values('length').reset_index(drop=True)
    # df_valid = df_valid[-500:] # debug
    # df_valid = df_valid[:500000] # 0
    # df_valid = df_valid[500000:1000000] # 1
    # df_valid = df_valid[1000000:1300000] # 2
    # df_valid = df_valid[1300000:] # 3
    df_valids = [df_valid[:350000], df_valid[350000:700000], df_valid[700000:1000000],
                 df_valid[1000000:1200000], df_valid[1200000:1400000], df_valid[1400000:],
                 df_valid[-100:]]
    df_valid = df_valids[args.which]
    print(df_valid)
    valid_dataset = BmsDataset(df_valid, tokenizer, augment=fast_remote_unrotate_augment)
    valid_loader  = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = 46, #32,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        #collate_fn  = lambda batch: null_collate(batch,False),
    )
    log.write('mode : ensemble folds')
    log.write('valid_dataset : \n%s\n'%(valid_dataset))

    ## net ----------------------------------------
    nets = []
    from fairseq_model import Net
    for ckpt in ckpts00+ckpts01:
        net = Net()
        net.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        net.eval()
        nets.append(net.cuda())
        del net
        gc.collect()

    for ckpt in ckpts02:
        net = Net(rew=True)
        net.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        net.eval()
        nets.append(net.cuda())
        del net
        gc.collect()

    for ckpt in ckpts03:
        net = {}
        states = torch.load(ckpt, map_location=torch.device('cpu'))

        encoder = Encoder(cfg.model_name, pretrained=False)
        encoder.load_state_dict(states['encoder'])
        encoder.cuda()
        encoder.eval()

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
        decoder.eval()

        net['encoder'] = encoder
        net['decoder'] = decoder
        nets.append(net)
        del net, encoder, decoder, states
        gc.collect()

    start_timer = timer()
    predict = do_predict(nets, tokenizer, valid_loader, ckpts)
    log.write('time %s \n' % time_to_str(timer() - start_timer, 'min'))

    # if is_norm_ichi:
    #     predict = [normalize_inchi(t) for t in predict]

    df_submit = pd.DataFrame()
    df_submit.loc[:,'image_id'] = df_valid.image_id.values
    df_submit.loc[:,'InChI'] = predict #
    df_submit.to_csv(out_dir + f'/submit_{args.which}.csv', index=False)
    print(out_dir + f'/submit_{args.which}.csv')

    if is_norm_ichi:
        predict_norm = [normalize_inchi(t) for t in predict]
        df_submit.loc[:,'InChI'] = predict_norm #
        df_submit.to_csv(out_dir + '/submit_norm.csv', index=False)

    log.write('submit_dir : %s\n' % (out_dir))
    log.write('initial_checkpoint : %s\n' % (ckpts))
    log.write('df_submit : %s\n' % str(df_submit.shape))
    log.write('%s\n' % str(df_submit))
    log.write('\n')


def cat_submit():
    df =[]
    for i in [3,2,1,0]:
        file=\
            '/root/share1/kaggle/2021/bms-moleular-translation/result/try10/tnt-s-224-fairse/fold3/valid/remote-00266000_model-gpu%d/submit.csv'%i
        d = pd.read_csv(file)
        df.append(d)

    df = pd.concat(df)
    print(df)
    print(df.shape)
    df.to_csv('/root/share1/kaggle/2021/bms-moleular-translation/result/try10/tnt-s-224-fairse/fold3/valid/submission-00266000_model.csv',index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', type=int, default=-1, help='debug')
    args = parser.parse_args()
    run_submit(args)
