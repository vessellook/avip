from pathlib import Path

import numpy as np
from PIL import Image
from mdutils import MdUtils

import grayscale
import gradient


def save_and_add_to_report(report: MdUtils, text: str, image: Image, path: Path, core=None):
    image.save(path)
    report.new_line(text)
    if core is not None:
        report.new_line(report.insert_code(str(core), language='python'))
    report.new_line(report.new_inline_image(text, path=path.name))


def lab3(threshold: int):
    assets_dir = Path('assets')
    output_dir = Path('reports/lab3')
    output_dir.mkdir(parents=True, exist_ok=True)
    report = MdUtils(file_name=str(output_dir / 'README.md'))
    report.new_header(level=1, title="Выделение контуров на изображении "
                                     "(Прюитт с матрицами 5x5 и Манхеттенским расстоянием)")
    report.new_line('Выполнил Васелюк Артём Б19-514')

    asset_names = ['cat.bmp', 'city.bmp', 'text.bmp', 'panda.bmp', 'road.bmp', 'slavic-book.bmp', 'woman.bmp']
    for name in asset_names:
        report.new_header(level=2, title=f'Файл "{name}"')
        asset = Image.open(assets_dir / name).convert('RGB')
        grayscaled = grayscale.transform(asset, method='average')
        save_and_add_to_report(report, 'Исходная картинка', asset, output_dir / f'grayscaled.{name}')
        np_img = np.asarray(grayscaled).astype(np.int8)
        matrix_x, matrix_y = gradient.get_pruitt(5)
        g_x = gradient.get_partial(np_img, matrix_x)
        save_and_add_to_report(report, 'Градиентная матрица Gx',
                               Image.fromarray(gradient.normalize(g_x)),
                               output_dir / f'gx.{name}')
        g_y = gradient.get_partial(np_img, matrix_y)
        save_and_add_to_report(report, 'Градиентная матрица Gy',
                               Image.fromarray(gradient.normalize(g_y)),
                               output_dir / f'gy.{name}')
        g_norm = gradient.normalize(gradient.get_gradient_abs(g_x, g_y))
        save_and_add_to_report(report, 'Градиентная матрица G',
                               Image.fromarray(g_norm),
                               output_dir / f'g.{name}')
        binarized = np.zeros_like(g_norm)
        binarized[g_norm > threshold] = 255
        save_and_add_to_report(report, 'Бинаризованная градиентная матрица G',
                               Image.fromarray(binarized),
                               output_dir / f'binarized.{name}')
    report.create_md_file()


if __name__ == '__main__':
    lab3(200)
