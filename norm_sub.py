from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pathlib import Path
import argparse

# def normalize_inchi(inchi):
#     try:
#         mol = Chem.MolFromInchi(inchi)
#         return inchi if (mol is None) else Chem.MolToInchi(mol)
#     except: return inchi

def normalize_inchi(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            try:
                inchi = Chem.MolToInchi(mol)
            except:
                pass
    except:
        pass
    return inchi

# Segfault in rdkit taken care of, run it with:
# ```while [ 1 ]; do python normalize_inchis.py && break; done```
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='normlize inchi')
    parser.add_argument('-p', '--path', default='submission.csv', type=str)
    args = parser.parse_args()
    # Input & Output
    orig_path = Path(f'{args.path}')
    norm_path = orig_path.with_name(orig_path.stem+'_norm.csv')
    
    # Do the job
    N = norm_path.read_text().count('\n') if norm_path.exists() else 0
    print(N, 'number of predictions already normalized')

    r = open(str(orig_path), 'r')
    w = open(str(norm_path), 'a', buffering=1)

    for _ in range(N):
        r.readline()
    line = r.readline()  # this line is the header or is where it segfaulted last time
    w.write(line)

    pbar = tqdm()
    while True:
        line = r.readline()
        if not line:
            break  # done
        image_id = line.split(',')[0]
        inchi = ','.join(line[:-1].split(',')[1:]).replace('"','')
        inchi_norm = normalize_inchi(inchi)
        w.write(f'{image_id},"{inchi_norm}"\n')
        pbar.update(1)

    r.close()
    w.close()

# import pandas as pd
# import edlib
# from tqdm import tqdm

# sub_df = pd.read_csv('submission.csv')
# sub_norm_df = pd.read_csv('submission_norm.csv')

# lev = 0
# N = len(sub_df)
# for i in tqdm(range(N)):
#     inchi, inchi_norm = sub_df.iloc[i,1], sub_norm_df.iloc[i,1]
#     lev += edlib.align(inchi, inchi_norm)['editDistance']

# print(lev/N)