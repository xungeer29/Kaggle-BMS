import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import get_train_file_path, get_test_file_path

# train = pd.read_csv('data/train_labels.csv')
# train['file_path'] = train['image_id'].apply(get_train_file_path)

# os.makedirs('logs/train', exist_ok=True)
# for path in tqdm(train['file_path'].values):
#     img = cv2.imread(path)
#     h, w, c = img.shape
#     if h > w:
#         print(path, h, w)
#         os.system(f'cp {path} logs/train/')

test = pd.read_csv('data/sample_submission.csv')
test['file_path'] = test['image_id'].apply(get_test_file_path)

os.makedirs('logs/symmetric/test1', exist_ok=True)
for path in tqdm(test['file_path'].values):
    img = cv2.imread(path)
    img1 = np.rot90(img, 2)
    loss = np.mean((img-img1)**2)
    # print(loss);exit()
    if loss < 0.01:
        cv2.imwrite(f'logs/symmetric/test1/{os.path.basename(path)}', img)