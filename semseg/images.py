from PIL import Image
import numpy as np


def save_predictions(outdir, name, pred_arr, cnts_arr):
    """
    Save generated predictions array as png file.
    """
    name = name.split(".")[0] + ".png"
    path = os.path.join(outdir, name)
    img = pred_arr / cnts_arr
    img = img.astype(np.uint8)
    image = Image.fromarray(img)
    image.save(path)

