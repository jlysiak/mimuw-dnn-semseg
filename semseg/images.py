import os
from PIL import Image
import numpy as np

def save_image(outdir, name, arr):
    if name is None:
        return None
    if type(name) == bytes:
        name = name.decode()
    path = os.path.join(outdir, name)
    img = arr.astype(np.uint8)
    image = Image.fromarray(img)
    image.save(path)
    return path


def save_predictions(outdir, name, pred_arr, cnts_arr):
    """
    Save generated predictions array as png file.
    """
    if name is None:
        return None

    if type(name) == bytes:
        name = name.decode()
    
    name = name.split(".")[0] + ".png"
    path = os.path.join(outdir, name)
    img = pred_arr / cnts_arr
    img = img.astype(np.uint8)
    image = Image.fromarray(img)
    image.save(path)
    return path


def calc_accuracy(pred_arr, truth_arr, cnts_arr):
    if pred_arr is None:
        return None
    preds = pred_arr / cnts_arr
    preds = preds.astype(np.uint8)
    truth = truth_arr / cnts_arr
    truth = truth.astype(np.uint8)
    ok = np.equal(preds, truth).astype(np.uint8)
    ok = np.sum(ok)

    sz = np.prod(pred_arr.shape)
    return ok / sz

