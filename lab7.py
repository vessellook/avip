from pathlib import Path

import numpy as np
from PIL import Image

from texture import get_matrix, log_contrast, corr, save_histogram


def lab7():
    textures_dir = Path('textures')
    output_dir = Path('reports/lab7/')
    output_dir.mkdir(exist_ok=True)
    for texture_path in textures_dir.iterdir():
        name = texture_path.name.split('.')[0]

        image = Image.open(str(texture_path)).convert('L')
        np_image = np.asarray(image, dtype=np.uint8)
        image.save(str(output_dir / f'original.{name}.bmp'))
        np_matrix = get_matrix(np_image, d=1, diag=True)
        np_matrix = 512 * np_matrix / np_matrix.max()
        matrix = Image.fromarray(np_matrix).convert('L')
        matrix.save(str(output_dir / f'matrix.{name}.bmp'))

        np_contrasted_image = log_contrast(np_image)
        contrasted_image = Image.fromarray(np_contrasted_image).convert('L')
        contrasted_image.save(str(output_dir / f'contrasted.{name}.bmp'))
        np_contrasted_matrix = get_matrix(np_contrasted_image, d=1, diag=True)
        np_contrasted_matrix = 512 * np_contrasted_matrix / np_contrasted_matrix.max()
        contrasted_matrix = Image.fromarray(np_contrasted_matrix).convert('L')
        contrasted_matrix.save(str(output_dir / f'contrasted-matrix.{name}.bmp'))
        print(name, corr(np_matrix), corr(np_contrasted_matrix))

        save_histogram(np_image.flatten(), output_dir / f'histogram.{name}.png')
        save_histogram(np_contrasted_image.flatten(), output_dir / f'contrasted-histogram.{name}.png')


if __name__ == '__main__':
    lab7()
