from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mdutils import MdUtils
from prettytable import PrettyTable

import slavic
from slavic import LetterCase, LOWERCASE, UPPERCASE
from features import (center, relative_center, axial_moments, relative_axial_moments, orientation_angle, weight,
                      density, vertical_projection, horizontal_projection, draw_projection)
from font_utils import Font
from slavic import slavic_font_names, crop_white
from thresholding import balanced_thresholding


def same(a: Image.Image, b: Image.Image):
    if a.height == b.height and a.width == b.width:
        if (np.asarray(a) - np.asarray(b)).sum() == 0:
            return True
    return False


@dataclass
class LetterImage:
    letter_case: slavic.LetterCase
    letter_name: str
    font_name: str
    data: Image.Image

    def attributes(self):
        return self.letter_case, self.letter_name, self.font_name, self.data


# def draw_all_letters(report: MdUtils, alphabet, font: Font, get_path: Callable[[str, str, Font], Path],
#                      bad_image: Image.Image) -> Iterable[ImageData]:
def draw_all_letters(alphabet, font: Font, bad_image: Image.Image) -> List[LetterImage]:
    # выбери, какие шрифты нужны
    # в некоторых шрифтах отсутствуют некоторые буквы

    results = []
    # report.new_line(f'Шрифт {font.name}')
    for letter in alphabet:
        images = dict()
        for word_case in LetterCase:
            image = font.try_draw_glyph(letter[word_case], 100, 100)
            if image is None or same(image, bad_image):
                continue
            image = balanced_thresholding(image)
            image = crop_white(image)
            images[word_case] = image

        if {UPPERCASE, LOWERCASE}.issubset(images.keys()) and same(images[UPPERCASE], images[LOWERCASE]):
            del images[LOWERCASE]

        if len(images) == 0:
            continue

        # report.new_line(f"Буква {letter['name']}")
        for word_case, image in images.items():
            image_data = LetterImage(
                letter_case=word_case,
                letter_name=letter['name'],
                font_name=font.name,
                data=image,
            )
            results.append(image_data)
            # filename = get_path(letter['name'], word_case, font)
            # if word_case == 'small':
            #     report.new_line('Нижний регистр')
            #     report.new_line(report.new_inline_image(text='Нижний регистр', path=str(filename)))
            # elif word_case == 'capital':
            #     report.new_line('Верхний регистр')
            #     report.new_line(report.new_inline_image(text='Верхний регистр', path=str(filename)))
            # # image.save(output_dir / f"{letter['name']}.{word_case}.{font.name}.bmp")
            # image.save(filename)

    return results


def get_scalar_features(data: Iterable[LetterImage]) -> PrettyTable:
    t = PrettyTable(['letter', 'case', 'font', 'width', 'height', 'weight', 'density',
                     'x_center', 'y_center',
                     'x_rel_center', 'y_rel_center',
                     'x_axial', 'y_axial',
                     'x_rel_axial', 'y_rel_axial',
                     'orientation_angle'])
    for li in data:
        letter_case, letter_name, font_name, image = li.attributes()
        np_image = np.asarray(image) / 255
        h, w = np_image.shape
        x_center, y_center = center(np_image)
        x_rel_center, y_rel_center = relative_center(np_image)
        x_axial, y_axial = axial_moments(np_image)
        x_rel_axial, y_rel_axial = relative_axial_moments(np_image)
        orientation = orientation_angle(np_image)
        t.add_row(
            [letter_name, letter_case.value, font_name, h, w, weight(np_image), density(np_image),
             x_center, y_center,
             x_rel_center, y_rel_center,
             x_axial, y_axial,
             x_rel_axial, y_rel_axial,
             orientation])
    # for directory in sorted(list(Path(__file__).parent.joinpath('slavic_letters').iterdir())):
    #     if not directory.is_dir():
    #         continue
    #     for file in sorted(list(directory.iterdir())):
    #         word_case = file.name.split('.')[0]
    #         font = file.name.split('.')[1]

    return t


def draw_projections(letters_dir: Path, output_dir: Path):
    for file in letters_dir.iterdir():
        if file.is_dir():
            continue
        if not file.name.endswith('bmp'):
            continue
        letter_name, letter_case, font_name, _ = file.name.split('.')

        image = Image.open(str(file))
        # image.show()
        np_image = np.asarray(image) / 255

        projection = vertical_projection(np_image)
        indices = np.arange(0, len(projection))
        filename = f'{letter_name}.{letter_case}.{font_name}.vertical.png'
        draw_projection(projection, indices, str(output_dir / filename))

        projection = horizontal_projection(np_image)
        indices = np.arange(0, len(projection))
        filename = f'{letter_name}.{letter_case}.{font_name}.horizontal.png'
        draw_projection(projection, indices, str(output_dir / filename))


def lab4(font: str):
    output_dir = Path('reports/lab4')
    output_dir.mkdir(parents=True, exist_ok=True)
    report = MdUtils(file_name=str(output_dir / 'README.md'))
    report.new_header(level=1, title="Выделение контуров на изображении "
                                     "(Прюитт с матрицами 5x5 и Манхеттенским расстоянием)")
    report.new_line('Выполнил Васелюк Артём Б19-514')

    bad_image = Image.open('assets/bad-letter-image.bmp')
    font = Font(Path('fonts') / 'Flavius.ttf', 'Flavius', 50)
    images = draw_all_letters(slavic.slavic_alphabet, font, bad_image)
    report.new_line(f'Шрифт {font.name}')
    for image in images:
        filename = f'{image.letter_name}.{image.letter_case.value}.{image.font_name}.bmp'.replace(' ', '_')
        image.data.save(output_dir / filename)
        if image.letter_case == UPPERCASE:
            text_case = 'Верхний регистр'
        else:
            text_case = 'Нижний регистр'
        report.new_line(report.new_inline_image(text=f'{text_case} {image.letter_name}',
                                                path=str(filename)))
    projections_dir = output_dir / 'projections'
    projections_dir.mkdir(exist_ok=True)
    draw_projections(output_dir, projections_dir)
    scalar_features = get_scalar_features(images)
    with (output_dir / 'scalar_features.csv').open('w', newline='') as f:
        f.write(scalar_features.get_csv_string())
    report.create_md_file()


if __name__ == '__main__':
    lab4(slavic_font_names)
