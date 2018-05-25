#!/usr/bin/env python3.5

from PIL import Image
import numpy as np

palette = np.random.uniform(0, 256, (16,3))
palette = np.floor(palette)
print(palette)

img = [[palette[i//32] for _ in range(512)] for i in range(512)]
img = np.array(img)
print(img.shape)
img = Image.fromarray(img, mode="RGB")
img.show()
