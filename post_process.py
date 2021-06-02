import numpy as np
import pandas as pd
from tqdm import tqdm

subs = [
        # 'subs/tnt(crop-entire)-lstm(eb4-eb0)_norm-lb1.05.csv',
        'subs/tnt-crop224(012345)-entire(01345)-lstm-lr_norm-lb1.04.csv',
        # 'subs/tnt-crop224(012345)-entire224(3)-lstm(0124)-norm-lb1.05.csv', # 0
        # 'subs/submit_tnt_lstm_norm-lb1.37.csv', # 1
        # 'subs/submit_crop_entire_2tta_norm_correct-lb1.48.csv', # tnt-crop224(fold012345)-entire224(fold3) 2
        # 'subs/tnt-crop-224-ensemble012345-2tta-norm-lb1.53.csv', # tnt-crop224(fold012345) 3
        'subs/tnt-crop(012345)-3tta_norm-lb1.29.csv', # tnt-crop224(fold012345) 3
        # 'subs/sub_ensemble_fold01_norm.csv', # lstm-lr ensemble fold[0,1] 4
        'subs/submit_lstm_fold0124_norm-lb1.42.csv', # lstm-lr ensemble fold[0,1,2,4] 5
        'subs/sub_beam5_norm.csv', # lstm-fold0-beamsearch5 6
        # 'subs/submit_norm.csv', # tnt 7
        # 'subs/submit_norm (1).csv', # tnt 8
        # 'subs/submit_norm (2).csv', # tnt 9
        'subs/tnt-320-00917000.csv', # tnt-320 10
        ]

for ii in range(len(subs)):
    vars()['sub'+str(ii)] = pd.read_csv(subs[ii])
    vars()['sub'+str(ii)] = vars()['sub'+str(ii)].sort_values(by='image_id', ascending=True)
    vars()['sub'+str(ii)].reset_index(inplace=True)
    vars()['sub'+str(ii)] = vars()['sub'+str(ii)].drop('index',axis=1)

sums = []
subfinal = sub0.copy()
for ii in range(len(subs)):
    subfinal[f'InChI{ii}'] = vars()['sub'+str(ii)]['InChI']
    subfinal[f'InChI{ii}sum'] = 0
    sums.append(f'InChI{ii}')

for ii in range(len(subs)):
    sums2 = sums.copy()
    sums2.remove(f'InChI{ii}')
    for obj in sums2:
        subfinal[f'InChI{ii}sum'][subfinal[f'InChI{ii}'] == subfinal[obj]] += 1

# original
# subfinal['InChI'][subfinal['InChI1sum'] > subfinal['InChI0sum']] = subfinal['InChI1']
# subfinal[['image_id', 'InChI']].to_csv('submission.csv', index=False)


# Modified
inchis = []
for i in range(len(subfinal)):
    maxsum = 0
    inchi = subfinal['InChI'][i]
    for j in range(len(subs)):
        if subfinal[f'InChI{j}sum'][i] > maxsum:
            maxsum = subfinal[f'InChI{j}sum'][i]
            inchi = subfinal[f'InChI{j}'][i]
    inchis.append(inchi)

subfinal['InChI'] = inchis
subfinal[['image_id', 'InChI']].to_csv('tnt-crop224(012345)-entire(01345)-lstm-lr_norm_pp.csv', index=False)
