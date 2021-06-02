import os

from common import *
from bms import *
from configure import *

from lib.net.lookahead import *
from lib.net.radam import *

from dataset_224 import *
from fairseq_model import *


# ----------------
is_mixed_precision = True #False  #


###################################################################################################
import torch.cuda.amp as amp
# if is_mixed_precision:
#     class AmpNet(Net):
#         @torch.cuda.amp.autocast()
#         def forward(self, image):
#             #return super(AmpNet, self).forward(image)
#             return super(AmpNet, self).forward_argmax_decode(image)
# else:
#     AmpNet = Net
AmpNet = Net

def image_padding(image):
    h, w = image.shape[:2]
    maxlen = max(h ,w)
    zero = np.zeros((maxlen, maxlen), dtype=image.dtype)
    if h > w:
        num_zero = (h-w) // 2
        zero[:, num_zero:num_zero+w] = image
    else:
        num_zero = (w-h)//2
        zero[num_zero:num_zero+h, :] = image
    return zero

def crop_image_to_rect(image):
    h,w = image.shape[:2]
    if h > w:
        image = np.rot90(image, 1)
    if w / h < 2:
        return image
    h,w = image.shape[:2]
    num_h = round((w*h)**0.5 / h)
    patchH = h
    patchW = round(w*h/(num_h*h))
    num_w = round(w / patchW)
    patchW = w // num_w

    image = cv2.resize(image, (patchW*num_w, h))

    patchs = [image[:, i*patchW:(i+1)*patchW] for i in range(num_w)]
    img = np.concatenate(patchs, axis=0)

    return img

# start here ! ###################################################################################
def fast_remote_unrotate_augment(r):
    image = r['image']
    index = r['index']
    h, w = image.shape

    if h > w:
        # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = np.rot90(image, 1)
    #imgae = image_padding(image)
    # image = crop_image_to_rect(image)
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    #l= r['d'].orientation
    #if l == 1:
    #    image = np.rot90(image, -1)
    #if l == 2:
    #    image = np.rot90(image, 1)
    #if l == 3:
    #    image = np.rot90(image, 2)

    #image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    #assert (image_size==224)


    image = image.astype(np.float16) / 255
    image = torch.from_numpy(image).unsqueeze(0).repeat(3,1,1)

    r={}
    r['image'] = image
    return r


def do_predict(net, tokenizer, valid_loader):

    text = []

    start_timer = timer()
    valid_num = 0
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['image'])
        image = batch['image'].cuda()


        net.eval()
        with torch.no_grad():
            with amp.autocast():
                k = net.forward_argmax_decode(image)

                # token = batch['token'].cuda()
                # length = batch['length']
                # logit = data_parallel(net,(image, token, length))
                # k = logit.argmax(-1)

                k = k.data.cpu().numpy()
                k = tokenizer.predict_to_inchi(k)
                text.extend(k)

        valid_num += batch_size
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)

    assert(valid_num == len(valid_loader.dataset))
    print('')
    return text


def run_submit():
    gpu_no = int(os.environ['CUDA_VISIBLE_DEVICES'])

    fold = 3
    out_dir = 'logs/tnt-s-224-luozhengbo/fold%d' % fold
    initial_checkpoint = out_dir + '/checkpoint/01359000_model.pth'
    is_norm_ichi = False #True
    mode = 'remote' # remote local

    submit_dir = out_dir + '/valid/%s-%s-gpu%d'%(mode, initial_checkpoint[-18:-4],gpu_no)
    os.makedirs(submit_dir, exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('is_norm_ichi = %s\n' % is_norm_ichi)
    log.write('\n')

    #
    ## dataset ------------------------------------
    tokenizer = load_tokenizer()

    if 'remote' in mode: #1_616_107 testset
        df_valid = make_fold('test')

    elif 'local' in mode:  #484_837 valid
        df_train, df_valid = make_fold('train-%d' % fold)
        # df_valid = df_valid[:10_000]

    df_valid = df_valid.sort_values('length').reset_index(drop=True)
    valid_dataset = BmsDataset(df_valid, tokenizer, augment=fast_remote_unrotate_augment)
    valid_loader  = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = 512, #32,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        #collate_fn  = lambda batch: null_collate(batch,False),
    )
    log.write('mode : %s\n'%(mode))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))

    ## net ----------------------------------------
    tokenizer = load_tokenizer()
    net = AmpNet().cuda()
    net.load_state_dict(torch.load(initial_checkpoint)['state_dict'], strict=True)
    net = torch.jit.script(net)

    start_timer = timer()
    predict = do_predict(net, tokenizer, valid_loader)
    log.write('time %s \n' % time_to_str(timer() - start_timer, 'min'))

    #np.save(submit_dir + '/probability.uint8.npy',probability)
    #write_pickle_to_file(submit_dir + '/predict.pickle', predict)
    #exit(0)

    #----
    if is_norm_ichi:
        predict = [normalize_inchi(t) for t in predict]

    df_submit = pd.DataFrame()
    df_submit.loc[:,'image_id'] = df_valid.image_id.values
    df_submit.loc[:,'InChI'] = predict #
    df_submit.to_csv(submit_dir + '/submit.csv', index=False)

    log.write('submit_dir : %s\n' % (submit_dir))
    log.write('initial_checkpoint : %s\n' % (initial_checkpoint))
    log.write('df_submit : %s\n' % str(df_submit.shape))
    log.write('%s\n' % str(df_submit))
    log.write('\n')

    if 'local' in mode:
        truth = df_valid['InChI'].values.tolist()
        lb_score = compute_lb_score(predict, truth)
        #print(lb_score)
        log.write('lb_score  = %f\n'%lb_score.mean())
        log.write('is_norm_ichi = %s\n' % is_norm_ichi)
        log.write('\n')

        df_eval = df_submit.copy()
        df_eval.loc[:,'truth']=truth
        df_eval.loc[:,'lb_score']=lb_score
        df_eval.loc[:,'length'] = df_valid['length']
        df_eval.to_csv(submit_dir + '/df_eval.csv', index=False)
         # df_valid.to_csv(submit_dir + '/df_valid', index=False)


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
    run_submit()
