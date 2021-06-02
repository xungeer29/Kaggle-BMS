STOI = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}


image_size = 224
vocab_size = 193
max_length = 300 #275


# ------------------------------patch_tnt--------------------------------
patch_size   = 16
pixel_pad    = 3
pixel_stride = 4
num_pixel    = (patch_size // pixel_stride)**2
pixel_scale  = 0.8  #1.0  #0.62=36/58 #1.0

pixel_dim  = 24
patch_dim  = 384

text_dim    = 384
decoder_dim = 384
num_layer = 3
num_head  = 8
ff_dim = 1024
