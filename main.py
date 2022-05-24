from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from mdutils import MdUtils

from features import vertical_projection, horizontal_projection, draw_projection
from segmentation import get_symbol_boxes


def symbol_segmentation(report: MdUtils, image, output_dir, font_name):
    image = image.copy()
    np_image: np.ndarray = np.asarray(image) / 255
    rects = get_symbol_boxes(np_image, threshold=2)
    paths = []
    d = ImageDraw.ImageDraw(image)
    for num, (top, right, bottom, left) in enumerate(rects, 1):
        prefix = f'image{num}.{font_name}'
        file_name = f'symbol.{prefix}.bmp'
        path = output_dir / file_name
        paths.append(path)
        region = image.crop((left, top, right, bottom))
        region.save(str(path))
        draw_projections(report, np_image[top:bottom, left:right], output_dir, prefix)
        report.new_line(f'Границы сегмента {num}: сверху {top}, снизу {bottom}, слева {left}, справа {right}')
        report.new_line(report.new_inline_image(text=f'Сегмент {num}',
                                                path=file_name))
        d.rectangle(((left, top), (right, bottom)), outline=128)
    return rects, paths, image


def draw_projections(report, binary: np.ndarray, output_dir, prefix):
    filename = f'{prefix}.vertical.png'
    projection = vertical_projection(binary)
    indices = np.arange(0, len(projection))
    draw_projection(projection, indices, str(output_dir / filename))
    report.new_line(report.new_inline_image(text=f'Вертикальная проекция',
                                            path=filename))

    filename = f'{prefix}.horizontal.png'
    projection = horizontal_projection(binary)
    indices = np.arange(0, len(projection))
    draw_projection(projection, indices, str(output_dir / filename))
    report.new_line(report.new_inline_image(text=f'Горизонтальная проекция',
                                            path=filename))


def lab5():
    output_dir = Path('reports/lab5/')
    output_dir.mkdir(exist_ok=True)
    report = MdUtils(file_name=str(output_dir / 'README.md'))
    report.new_header(level=1, title="Сегментация текста")
    report.new_line('Выполнил Васелюк Артём Б19-514')
    font_name = 'Flavius'
    image_line_path = Path('hello.Flavius.png')

    report.new_line(f'Шрифт {font_name}')
    image_line = Image.open(str(image_line_path)).convert('L')
    filename = f'original.' + image_line_path.name
    image_line.save(str(output_dir / filename))
    report.new_line(report.new_inline_image(text=f'Исходное изображение',
                                            path=filename))

    rects, paths, image = symbol_segmentation(report, image_line, output_dir, font_name)
    report.new_line(f'Сегменты')
    filename = f'with-rects.' + image_line_path.name
    image.save(str(output_dir / filename))
    report.new_line(report.new_inline_image(text=f'Изображение с сегментами',
                                            path=filename))


if __name__ == '__main__':
    lab5()
