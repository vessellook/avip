from pathlib import Path

import numpy as np
from PIL import Image
from mdutils import MdUtils

from filtering import erosion
import grayscale


def save_and_add_to_report(report: MdUtils, text: str, image: Image, path: Path, core=None):
    image.save(path)
    report.new_line(text)
    if core is not None:
        report.insert_code(str(core))
    report.new_line(report.new_inline_image(text, path=path.name))


def lab2():
    assets_dir = Path('assets')
    output_dir = Path('reports/lab2')
    output_dir.mkdir(parents=True, exist_ok=True)
    report = MdUtils(file_name=str(output_dir / 'README.md'))
    report.new_header(level=1, title="Фильтрация изображений и морфологические операции")
    report.new_line('Выполнил Васелюк Артём Б19-514')

    asset_names = ['cat.bmp', 'city.bmp', 'text.bmp', 'panda.bmp', 'road.bmp', 'slavic-book.bmp', 'woman.bmp']
    cores = {'rect': np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]]),
             'circle': np.array([[0, 1, 1, 1, 0],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [0, 1, 1, 1, 0]]),
             'ring': np.array([[1, 1, 1, 1, 1],
                               [1, 0, 0, 0, 1],
                               [1, 0, 0, 0, 1],
                               [1, 0, 0, 0, 1],
                               [1, 1, 1, 1, 1]])
             }
    for name in asset_names:
        report.new_header(level=2, title=f'Файл "{name}"')
        asset = Image.open(assets_dir / name).convert('RGB')
        grayscaled = grayscale.transform(asset, method='average')
        save_and_add_to_report(report, 'Исходная картинка', asset, output_dir / f'grayscaled.{name}')
        for core_name, core in cores.items():
            eroded = erosion(grayscaled, core)
            save_and_add_to_report(report, 'Сжатие с ядром', eroded,
                                   output_dir / f'erosion.{core_name}.{name}', core)
            save_and_add_to_report(report, 'Разность', grayscale.difference(grayscaled, eroded),
                                   output_dir / f'difference.erosion.{core_name}.{name}')
    report.create_md_file()


if __name__ == '__main__':
    lab2()
