import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
from tqdm import tqdm
import math

from models import Encoder, Attention, DecoderWithAttention
from utils import get_score, init_logger, seed_everything, AverageMeter, timeSince
from dataset import Tokenizer, get_train_file_path, BMSDataset, TestDataset, get_transforms, bms_collate, get_test_file_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def beamSearch(encoder, decoder, dataloader, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    vocab_size = len(word_map)

    seq_all, alphas_all, text_all = [], [], []

    tk0 = tqdm(dataloader, total=len(dataloader))
    for image in tk0:
        k = beam_size
        image = image.to(device)
        with torch.no_grad():
            encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(3)

            # Flatten encoding
            encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)

            # We'll treat the problem as having a batch size of k
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word_map['<sos>']]] * k).to(device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            # Tensor to store top k sequences' alphas; now they're just 1s
            seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

            # Lists to store completed sequences, their alphas and scores
            complete_seqs = list()
            complete_seqs_alpha = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            h, c = decoder.init_hidden_state(encoder_out)

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe
                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
                scores = decoder.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences, alphas
                # print(seqs, prev_word_inds, next_word_inds)
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)  # (s, step+1, enc_image_size, enc_image_size)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<eos>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                seqs_alpha = seqs_alpha[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 600:
                    break
                step += 1

            try:
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
                alphas = complete_seqs_alpha[i]
                text = tokenizer.predict_caption(seq)

                seq_all.append(seq)
                alphas_all.append(alphas)
                text_all.append(text.replace('<sos>', 'InChI=1S/'))
            except:
                seq_all.append([])
                alphas_all.append([])
                text_all.append('ERROR')

    return seq_all, alphas_all, text_all


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--start', '-s', type=int, default=0)
    parser.add_argument('--end', '-e', type=int, default=-1)
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    from logs.lr.code.configs import config as cfg

    tokenizer = torch.load('data/tokenizer2.pth')
    word_map = tokenizer.stoi

    # Load model
    encoder = Encoder(cfg.model_name, pretrained=False)
    decoder = DecoderWithAttention(attention_dim=cfg.attention_dim,
                                   embed_dim=cfg.embed_dim,
                                   decoder_dim=cfg.decoder_dim,
                                   vocab_size=len(tokenizer),
                                   encoder_dim=encoder.n_features,
                                   dropout=cfg.dropout,
                                   device=device,
                                   rnn=cfg.rnn)

    states = torch.load(args.model, map_location=torch.device('cpu'))
    encoder.load_state_dict(states['encoder'])
    decoder.load_state_dict(states['decoder'])
    encoder.to(device)
    decoder.to(device)
    decoder.eval()
    encoder.eval()

    test = pd.read_csv('data/sample_submission.csv')
    test['file_path'] = test['image_id'].apply(get_test_file_path)
 
    test_dataset = TestDataset(test, transform=get_transforms(data='valid', cfg=cfg))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    seqs, alphas, texts = beamSearch(encoder, decoder, test_loader, word_map, args.beam_size)
    exit()

    # Encode, decode with attention and beam search
    cnt = math.ceil(len(test)/10000)
    for i in range(cnt//6*args.start, cnt//6*args.end if args.end!=6 else cnt):
        sub = test[i*10000:(i+1)*10000]
        test_dataset = TestDataset(sub, transform=get_transforms(data='valid', cfg=cfg))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        seqs, alphas, texts = beamSearch(encoder, decoder, test_loader, word_map, args.beam_size)
        sub['InChI'] = texts
        sub[['image_id', 'InChI']].to_csv(f'logs/lr/beamsearch-fold1_{i}.csv', index=False)

