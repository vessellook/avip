from fractions import Fraction
from pathlib import Path

from PIL import Image
from mdutils import MdUtils

from resampling import interpolation, decimation, resampling_single_pass
import grayscale
import thresholding


def save_and_add_to_report(report: MdUtils, text: str, image: Image, path: Path):
    image.save(path)
    report.new_line(text)
    report.new_line(report.new_inline_image(text, path=path.name))


def lab1(M: int, N: int):
    assets_dir = Path('assets')
    output_dir = Path('reports/lab1')
    output_dir.mkdir(parents=True, exist_ok=True)
    report = MdUtils(file_name=str(output_dir / 'README.md'))
    report.new_header(level=1, title="Передискретизация, обесцвечивание и бинаризация растровых изображений")
    report.new_line('Выполнил Васелюк Артём Б19-514')

    asset_names = ['cat.bmp', 'city.bmp', 'text.bmp', 'panda.bmp', 'road.bmp', 'slavic-book.bmp', 'woman.bmp']
    for name in asset_names:
        report.new_header(level=2, title=f'Файл "{name}"')
        asset = Image.open(assets_dir / name).convert('RGB')
        save_and_add_to_report(report, 'Исходная картинка', asset, output_dir / f'original.{name}')
        report.new_header(level=3, title=f'Передискретизация')
        save_and_add_to_report(report, f'Интерполяция с M={M}',
                               interpolation(asset, M),
                               output_dir / f'interpolation.{name}')
        save_and_add_to_report(report, f'Децимация с N={N}',
                               decimation(asset, N),
                               output_dir / f'decimation.{name}')
        save_and_add_to_report(report, f'Децимация с N={N} после интерполяции с M={M}',
                               decimation(interpolation(asset, M), N),
                               output_dir / f'decimation.interpolation.{name}')
        save_and_add_to_report(report, f'Интерполяция с M={M} после децимации с N={N}',
                               interpolation(decimation(asset, N), M),
                               output_dir / f'interpolation.decimation.{name}')
        save_and_add_to_report(report, f'Однопроходная передискретизация с K={M}/{N}',
                               resampling_single_pass(asset, Fraction(M, N)),
                               output_dir / f'single_pass.{name}')

        report.new_header(level=3, title=f'Обесцвечивание и бинаризация')
        grayscaled = grayscale.transform(asset, method='average')
        save_and_add_to_report(report, f'Усреднение каналов',
                               grayscaled,
                               output_dir / f'grayscale.{name}')
        save_and_add_to_report(report, f'Сбалансированное пороговое отсечение гистограммы',
                               thresholding.balanced_thresholding(grayscaled),
                               output_dir / f'balance.{name}')
        save_and_add_to_report(report, f'Бинаризация Отцу',
                               thresholding.transform_otsu(grayscaled),
                               output_dir / f'otsu.{name}')
    report.create_md_file()


if __name__ == '__main__':
    lab1(3, 2)
