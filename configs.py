

# -*- coding:utf-8 -*- 
class DefaultConfigs(object):
    seed = 42
    num_workers = 4
    n_fold = 5
    print_freq = 1000
    max_len = 275

    train = True
    continue_train = False
    trn_fold = [0,1,2,3,4]

    # model
    model_name = 'tf_efficientnet_b4_ns' # [efficientnet_b3, resnet34]
    rnn = 'lstm' # ['lstm', 'gru']
    attention_dim = 512 # [256 512]
    embed_dim = 512 # [256 512]
    decoder_dim = 512
    dropout = 0.5

    size = 352 # image size 352
    batch_size = 48 # 32 64 128
    weight_decay = 1e-6


    # lr
    scheduler = 'CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    warmup = False
    warmup_ep = 1
    epochs = 20
    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    T_max = epochs # CosineAnnealingLR
    #T_0=4 # CosineAnnealingWarmRestarts
    encoder_lr = 1e-4*10 / 2
    decoder_lr = 4e-4*10 / 2
    min_lr = 1e-6*10 / 2


    gradient_accumulation_steps = 1
    max_grad_norm = 5

    # regularization parameter for 'doubly stochastic attention', as in the paper 
    alpha_c = 0. # 0 is close




config = DefaultConfigs()

    
