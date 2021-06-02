# !pip -q install cairosvg==2.5.2
# !pip -q install reportlab==3.5.65
# !pip -q install cssutils==2.2.0
# !conda install -q -y -c rdkit rdkit=2020_03_6

import cv2
import time
from pathlib import Path
from io import BytesIO
import copy
import logging

import numpy as np
import pandas as pd

import lxml.etree as et
import cssutils

from PIL import Image
import cairosvg
from skimage.transform import resize

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolDrawOptions


INPUT_DIR = Path('data')
TMP_DIR = Path('logs/tmp')
TRAIN_DATA_PATH = INPUT_DIR / 'train'
TRAIN_LABELS_PATH = INPUT_DIR / 'train_labels.csv'
TMP_DIR.mkdir(exist_ok=True)

cssutils.log.setLevel(logging.CRITICAL)

np.set_printoptions(edgeitems=30, linewidth=180)
print('RDKit version:', rdkit.__version__)
# Use a specific version of RDKit with known characteristics so that we can reliably manipulate output SVG.
# assert rdkit.__version__ == '2020.03.6'

TRAIN_LABELS = pd.read_csv(TRAIN_LABELS_PATH)
print(f'Read {len(TRAIN_LABELS)} training labels.')

def one_in(n):
    return np.random.randint(n) == 0 and True or False

def yesno():
    return one_in(2)


def svg_to_image(svg, convert_to_greyscale=True):
    svg_str = et.tostring(svg)
    # TODO: would prefer to convert SVG dirrectly to a numpy array.
    png = cairosvg.svg2png(bytestring=svg_str)
    image = np.array(Image.open(BytesIO(png)), dtype=np.float32)
    # Naive greyscale conversion.
    if convert_to_greyscale:
        image = image.mean(axis=-1)
    return image


def elemstr(elem):
    return ', '.join([item[0] + ': ' + item[1] for item in elem.items()])


# Streches the value range of an image to be exactly 0, 1, unless the image appears to be blank.
def stretch_image(img, blank_threshold=1e-2):
    img_min = img.min()
    img = img - img_min
    img_max = img.max()
    if img_max < blank_threshold:
        # seems to be blank or close to it
        return img
    img_max = img.max()
    if img_max < 1.0:
        img = img/img_max
    return img


def random_molecule_image(inchi, drop_bonds=True, add_noise=True, render_size=1200, margin_fraction=0.2):
    # Note that the original image is returned as two layers: one for atoms and one for bonds.
    #mol = Chem.MolFromSmiles(smiles)
    mol = Chem.inchi.MolFromInchi(inchi)
    d = Draw.rdMolDraw2D.MolDraw2DSVG(render_size, render_size)
    options = MolDrawOptions()
    options.useBWAtomPalette()
    options.additionalAtomLabelPadding = np.random.uniform(0, 0.3)
    options.bondLineWidth = int(np.random.uniform(1, 4))
    options.multipleBondOffset = np.random.uniform(0.05, 0.2)
    options.rotate = np.random.uniform(0, 360)
    options.fixedScale = np.random.uniform(0.05, 0.07)
    options.minFontSize = 20
    options.maxFontSize = options.minFontSize + int(np.round(np.random.uniform(0, 36)))
    d.SetFontSize(100)
    d.SetDrawOptions(options)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg_str = d.GetDrawingText()
    # Do some SVG manipulation
    svg = et.fromstring(svg_str.encode('iso-8859-1'))
    atom_elems = svg.xpath(r'//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'})
    bond_elems = svg.xpath(r'//svg:path[starts-with(@class,"bond-")]', namespaces={'svg': 'http://www.w3.org/2000/svg'})
    # Change the font.
    font_family = np.random.choice([
        'serif',
        'sans-serif'
    ])
    for elem in atom_elems:
        style = elem.attrib['style']
        css = cssutils.parseStyle(style)
        css.setProperty('font-family', font_family)
        css_str = css.cssText.replace('\n', ' ')
        elem.attrib['style'] = css_str
    # Create the original image layers.
    # TODO: separate atom and bond layers
    bond_svg = copy.deepcopy(svg)
    # remove atoms from bond_svg
    for elem in bond_svg.xpath(r'//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
        parent_elem = elem.getparent()
        if parent_elem is not None:
            parent_elem.remove(elem)
    orig_bond_img = svg_to_image(bond_svg)
    atom_svg = copy.deepcopy(svg)
    # remove bonds from atom_svg
    for elem in atom_svg.xpath(r'//svg:path', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
        parent_elem = elem.getparent()
        if parent_elem is not None:
            parent_elem.remove(elem)
    orig_atom_img = svg_to_image(atom_svg)
    if drop_bonds:
        num_bond_elems = len(bond_elems)
        if one_in(3):
            while True:
                # drop a bond
                # Let's leave at least one bond!
                if num_bond_elems > 1:
                    bond_elem_idx = np.random.randint(num_bond_elems)
                    bond_elem = bond_elems[bond_elem_idx]
                    bond_parent_elem = bond_elem.getparent()
                    if bond_parent_elem is not None:
                        bond_parent_elem.remove(bond_elem)
                        num_bond_elems -= 1
                else:
                    break
                if not one_in(4):
                    break
    img = svg_to_image(svg) > 254
    img = 1*img  # bool → int
    # Calculate the margins.
    black_indices = np.where(img == 0)
    row_indices, col_indices = black_indices
    if len(row_indices) >= 2:
        min_y, max_y = row_indices.min(), row_indices.max() + 1
    else:
        min_y, max_y = 0, render_size
    if len(col_indices) >= 2:
        min_x, max_x = col_indices.min(), col_indices.max() + 1
    else:
        min_x, max_x = 0, render_size
    margin_size = int(np.random.uniform(0.8*margin_fraction, 1.2*margin_fraction)*max(max_y - min_y, max_x - min_x))
    min_y, max_y = max(min_y - margin_size, 0), min(max_y + margin_size, render_size)
    min_x, max_x = max(min_x - margin_size, 0), min(max_x + margin_size, render_size)
    img = img[min_y:max_y, min_x:max_x]
    img = img.reshape([img.shape[0], img.shape[1]]).astype(np.float32)
    orig_bond_img = orig_bond_img[min_y:max_y, min_x:max_x]
    orig_atom_img = orig_atom_img[min_y:max_y, min_x:max_x]
    scale = np.random.uniform(0.2, 0.4)
    sz = (np.array(orig_bond_img.shape[:2], dtype=np.float32)*scale).astype(np.int32)
    orig_bond_img = resize(orig_bond_img, sz, anti_aliasing=True)
    orig_atom_img = resize(orig_atom_img, sz, anti_aliasing=True)
    img = resize(img, sz, anti_aliasing=False)
    img = img > 0.5
    if add_noise:
        # Add "salt and pepper" noise.
        salt_amount = np.random.uniform(0, 0.3)
        salt = np.random.uniform(0, 1, img.shape) < salt_amount
        img = np.logical_or(img, salt)
        pepper_amount = np.random.uniform(0, 0.001)
        pepper = np.random.uniform(0, 1, img.shape) < pepper_amount
        img = np.logical_or(1 - img, pepper)
    
    img = img.astype(np.uint8)  # boolean -> uint8
    orig_bond_img = 1 - orig_bond_img/255
    orig_atom_img = 1 - orig_atom_img/255
    # Stretch the range of the atom and bond images so tha tthe min is 0 and the max. is 1
    orig_bond_img = stretch_image(orig_bond_img)
    orig_atom_img = stretch_image(orig_atom_img)
    return img, orig_bond_img, orig_atom_img

def image_widget(a, greyscale=True):
    img_bytes = BytesIO()
    img_pil = Image.fromarray(a)
    if greyscale:
        img_pil = img_pil.convert("L")
    else:
        img_pil = img_pil.convert("RGB")
    img_pil.save(img_bytes, format='PNG')
    return widgets.Image(value=img_bytes.getvalue())


def test_random_molecule_image(n=4, graphics=True):
    t_sum = 0
    for imol in range(n):
        #smiles = np.random.choice(some_smiles)
        mol_index = np.random.randint(len(TRAIN_LABELS))
        mol_id, inchi = TRAIN_LABELS['image_id'][mol_index], TRAIN_LABELS['InChI'][mol_index]
        mol_train_img_path = TRAIN_DATA_PATH / mol_id[0] /mol_id[1] / mol_id[2] / (mol_id + '.png')
        train_img = Image.open(mol_train_img_path)
        t0 = time.time()
        img, orig_bond_img, orig_atom_img = random_molecule_image(inchi)
        tt = (time.time() - t0)*1000
        t_sum += tt
        print(f'Time: {tt:.2f}ms')

        if graphics:
            print('+-------------------------------------------------------------------------------')
            print(f'Molecule #{imol + 1}: {mol_id}: {inchi}')
            print('Training image path:', mol_train_img_path)
            print('Size:', img.shape)
            combined_orig_img =  np.clip(np.stack([orig_atom_img, orig_bond_img, np.zeros_like(orig_bond_img)], axis=-1), 0.0, 1.0)
            combined_orig_img = (255*combined_orig_img).astype(np.uint8)
            img = (255*(1 - img)).astype(np.uint8)
            # cv2.imwrite(f'logs/tmp/combined_orig_img_{imol}.png', combined_orig_img[..., ::-1])
            cv2.imwrite(f'logs/tmp/img_{imol}.png', img)
            print('train Size:', train_img.size, 'gen size:', img.shape)
    print(f'time/img: {t_sum/n:.2f}ms')
    return


# %timeit -n1 -r10 test_random_molecule_image(n=1, graphics=False)
test_random_molecule_image(n=100, graphics=True)
    