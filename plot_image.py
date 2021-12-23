import os, sys
import natsort
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


os.chdir(sys.path[0])

image_path = os.listdir('./inference_images_beam=1')[:-2]
image_path =natsort.natsorted(image_path, reverse=False)
print(f'{len(image_path)}: {image_path}')

images = []
for img_path in image_path:
    image = os.path.join('./inference_images_beam=1', img_path)
    image = plt.imread(image)
    images.append(image)



figure = plt.figure(figsize=(10,10), dpi=200)
grid = ImageGrid(figure, 111, nrows_ncols=(3, 4), axes_pad=0.1,)

for ax, im in zip(grid, images):
    ax.set_axis_off()
    ax.imshow(im)

plt.tight_layout()
plt.savefig('./inference_images.png')