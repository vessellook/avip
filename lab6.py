import json
from pathlib import Path

import numpy as np
from PIL import Image
from mdutils import MdUtils

from lab5 import symbol_segmentation
from features import density, relative_center, relative_axial_moments


def get_scalar_features(binary):
    return density(binary), *relative_center(binary), *relative_axial_moments(binary)


def distance(binary1, binary2):
    feature_pairs = zip(*map(get_scalar_features, (binary1, binary2)))
    square = sum(map(lambda pair: abs(pair[0] ** 2 - pair[1] ** 2), feature_pairs))
    return square ** 0.5


def get_symbol_paths(input_dir):
    symbol_image_paths = []
    for file in input_dir.iterdir():
        if file.name.endswith('bmp') and file.name.startswith('symbol.'):
            symbol_image_paths.append(file)
    symbol_image_paths.sort(key=lambda p: int(p.name.split('.')[1][5:]))
    return symbol_image_paths


def get_alphabet_paths(alphabet_dir):
    alphabet_paths = []
    for file in alphabet_dir.iterdir():
        if file.name.endswith('bmp'):
            alphabet_paths.append(file)
    alphabet_paths.sort(key=lambda p: p.name)
    return alphabet_paths


def get_hypotheses(symbol_paths, alphabet_paths):
    sorted_distances = []
    for image_path in symbol_paths:
        image = Image.open(str(image_path)).convert('L')
        binary = np.asarray(image) / 255
        distances = {}
        for alphabet_path in alphabet_paths:
            letter_name = alphabet_path.name.rsplit('.', 2)[0]
            letter_binary = np.asarray(Image.open(str(alphabet_path)).convert('L')) / 255
            distances[letter_name] = distance(binary, letter_binary)
        sorted_distances.append(sorted(distances.items(), key=lambda pair: pair[1]))
    return sorted_distances


def lab6():
    input_dir = Path('reports/lab5/')
    alphabet_dir = Path('reports/lab4/')
    output_dir = Path('reports/lab6/')
    new_symbols_dir = output_dir / 'symbols'
    output_dir.mkdir(exist_ok=True)
    new_symbols_dir.mkdir(exist_ok=True)
    symbol_image_paths = get_symbol_paths(input_dir)
    alphabet_paths = get_alphabet_paths(alphabet_dir)

    hypotheses = get_hypotheses(symbol_image_paths, alphabet_paths)
    with (output_dir / 'distances.txt').open('w') as f:
        for arr in hypotheses:
            f.write(json.dumps(arr, ensure_ascii=False))
            f.write('\n')
    for name, _ in next(zip(*hypotheses)):
        print(name)
    print()

    image = Image.open('big-hello.Flavius.png').convert('L')
    symbol_segmentation(image, new_symbols_dir, 'Flavius')
    new_symbol_paths = get_symbol_paths(new_symbols_dir)
    new_hypotheses = get_hypotheses(new_symbol_paths, alphabet_paths)
    print('Новые гипотезы')
    for (name, _), (new_name, _) in zip(next(zip(*hypotheses)), next(zip(*new_hypotheses))):
        print(name, new_name)


if __name__ == '__main__':
    lab6()
