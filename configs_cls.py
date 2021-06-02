

# -*- coding:utf-8 -*- 
class DefaultConfigs(object):
    seed = 42
    num_workers = 4
    n_fold = 5
    print_freq = 1000
    max_len = 275

    train = True
    trn_fold = [0] # [0, 1, 2, 3, 4]

    # model
    model_name = 'resnet34' # [efficientnet_b3, resnet34]
    dropout = 0.5

    size = 256 # image size 352
    batch_size = 128 # 32 64 128
    weight_decay = 1e-6


    # lr
    scheduler = 'CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs = 2
    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    T_max = epochs # CosineAnnealingLR
    #T_0=4 # CosineAnnealingWarmRestarts
    lr = 1e-3
    min_lr = 1e-6


    gradient_accumulation_steps = 1
    max_grad_norm = 5

    alpha_c = 1.0 # regularization parameter for 'doubly stochastic attention', as in the paper




config = DefaultConfigs()

    
