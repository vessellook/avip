from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from mdutils import MdUtils

from features import vertical_projection, horizontal_projection, draw_projection
from segmentation import get_symbol_boxes


def symbol_segmentation(image: Image.Image, output_dir, font_name, report: MdUtils = None):
    image = image.copy()
    binary: np.ndarray = np.asarray(image) / 255
    boxes = get_symbol_boxes(binary, threshold=2)
    d = ImageDraw.ImageDraw(image)
    for num, box in enumerate(boxes, 1):
        prefix = f'image{num}.{font_name}'
        file_name = f'symbol.{prefix}.bmp'
        path = output_dir / file_name
        region = image.crop((box.left, box.top, box.right, box.bottom))
        region.save(str(path))
        if report:
            report.new_line(f'Границы сегмента {num}: сверху {box.top}, снизу {box.bottom}, '
                            f'слева {box.left}, справа {box.right}')
            report.new_line(report.new_inline_image(text=f'Сегмент {num}',
                                                    path=file_name))
            draw_projections(report, binary[box.top:box.bottom, box.left:box.right], output_dir, prefix)
        d.rectangle(((box.left, box.top), (box.right, box.bottom)), outline=128)
    return image


def draw_projections(report: MdUtils, binary: np.ndarray, output_dir, prefix):
    filename = f'{prefix}.vertical.png'
    projection = vertical_projection(binary)
    indices = np.arange(0, len(projection))
    draw_projection(projection, indices, str(output_dir / filename))
    report.new_line(report.new_inline_image(text=f'Вертикальная проекция',
                                            path=filename))

    filename = f'{prefix}.horizontal.png'
    projection = horizontal_projection(binary)
    indices = np.arange(0, len(projection))
    draw_projection(indices, projection, str(output_dir / filename))
    report.new_line(report.new_inline_image(text=f'Горизонтальная проекция',
                                            path=filename))


def lab5():
    output_dir = Path('reports/lab5/')
    output_dir.mkdir(exist_ok=True)
    report = MdUtils(file_name=str(output_dir / 'README.md'))
    report.new_header(level=1, title="Сегментация текста")
    report.new_line('Выполнил Васелюк Артём Б19-514')
    font_name = 'Flavius'

    report.new_line(f'Шрифт {font_name}')
    image_line = Image.open('hello.Flavius.png').convert('L')
    filename = 'original.hello.Flavius.png'
    image_line.save(str(output_dir / filename))
    report.new_line(report.new_inline_image(text=f'Исходное изображение',
                                            path=filename))

    image = symbol_segmentation(image_line, output_dir, font_name, report=report)
    report.new_line(f'Сегменты')
    filename = 'with-rects.hello.Flavius.png'
    image.save(str(output_dir / filename))
    report.new_line(report.new_inline_image(text=f'Изображение с сегментами',
                                            path=filename))
    report.create_md_file()


if __name__ == '__main__':
    lab5()
