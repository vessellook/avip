from enum import Enum

import numpy as np
from PIL import Image


class LetterCase(Enum):
    UPPER = 'capital'
    LOWER = 'small'


UPPERCASE = LetterCase.UPPER
LOWERCASE = LetterCase.LOWER

# Русские буквы написаны как есть, латинские и кириллические нерусские буквы написаны через их имена в Unicode
slavic_alphabet = [
    {'name': 'азъ', UPPERCASE: 'А', LOWERCASE: 'а'},
    {'name': 'буки', UPPERCASE: 'Б', LOWERCASE: 'б'},
    {'name': 'веди', UPPERCASE: 'В', LOWERCASE: 'в'},
    {'name': 'глаголь', UPPERCASE: 'Г', LOWERCASE: 'г'},
    {'name': 'добро', UPPERCASE: 'Д', LOWERCASE: 'д'},
    {'name': 'есть', UPPERCASE: '\N{cyrillic capital letter ie}',
     LOWERCASE: '\N{cyrillic small letter ie}'},
    {'name': 'живите', UPPERCASE: 'Ж', LOWERCASE: 'ж'},
    {'name': 'сало', UPPERCASE: '\N{cyrillic capital letter dze}',
     LOWERCASE: '\N{cyrillic small letter dze}'},
    {'name': 'земля', UPPERCASE: 'З', LOWERCASE: 'з'},
    {'name': 'иже', UPPERCASE: 'И', LOWERCASE: 'и'},
    {'name': 'и', UPPERCASE: '\N{cyrillic capital letter byelorussian-ukrainian i}',
     LOWERCASE: '\N{cyrillic small letter byelorussian-ukrainian i}'},
    {'name': 'како', UPPERCASE: 'К', LOWERCASE: 'к'},
    {'name': 'люди', UPPERCASE: 'Л', LOWERCASE: 'л'},
    {'name': 'мыслите', UPPERCASE: 'М', LOWERCASE: 'м'},
    {'name': 'нашъ', UPPERCASE: 'Н', LOWERCASE: 'н'},
    {'name': 'онъ',
     UPPERCASE: [
         '\N{cyrillic capital letter round omega}',
         '\N{latin capital letter O}'
     ],
     LOWERCASE: [
         '\N{cyrillic small letter round omega}',
         '\N{latin small letter O}'
     ]
     },
    {'name': 'омега',
     UPPERCASE: [
         '\N{cyrillic capital letter omega}',
         '\N{latin capital letter W}'
     ],
     LOWERCASE: [
         '\N{cyrillic small letter omega}',
         '\N{latin small letter W}'
     ],
     },
    {'name': 'покой', UPPERCASE: 'П', LOWERCASE: 'п'},
    {'name': 'рцы', UPPERCASE: 'Р', LOWERCASE: 'р'},
    {'name': 'слово', UPPERCASE: 'С', LOWERCASE: 'с'},
    {'name': 'твердо', UPPERCASE: 'Т', LOWERCASE: 'т'},
    {'name': 'укъ',
     UPPERCASE: [
         '\N{cyrillic capital letter uk}',
         '\N{latin capital letter U}'
     ],
     LOWERCASE: [
         '\N{cyrillic small letter uk}',
         '\N{latin small letter U}'
     ],
     },
    {'name': 'ферть', UPPERCASE: 'Ф', LOWERCASE: 'ф'},
    {'name': 'херъ', UPPERCASE: 'Х', LOWERCASE: 'х'},
    {'name': 'оть',
     UPPERCASE: [
         '\N{cyrillic capital letter ot}',
         '\N{latin capital letter T}'
     ],
     LOWERCASE: [
         '\N{cyrillic small letter ot}',
         '\N{latin small letter T}'
     ]
     },
    {'name': 'цы', UPPERCASE: 'Ц', LOWERCASE: 'ц'},
    {'name': 'червь', UPPERCASE: 'Ч', LOWERCASE: 'ч'},
    {'name': 'ша', UPPERCASE: 'Ш', LOWERCASE: 'ш'},
    {'name': 'ща', UPPERCASE: 'Щ', LOWERCASE: 'щ'},
    {'name': 'еръ', UPPERCASE: 'Ъ', LOWERCASE: 'ъ'},
    {'name': 'еры', UPPERCASE: 'Ы', LOWERCASE: 'ы'},
    {'name': 'ерь', UPPERCASE: 'Ь', LOWERCASE: 'ь'},
    {'name': 'ять',
     UPPERCASE: [
         '\N{cyrillic capital letter yat}',
         'Э'
     ],
     LOWERCASE: [
         '\N{cyrillic small letter yat}',
         'э'
     ]
     },
    {'name': 'ю', UPPERCASE: 'Ю', LOWERCASE: 'ю'},
    {'name': 'йотированный юсъ малый',
     UPPERCASE: [
         '\N{cyrillic capital letter iotified little yus}',
         '\N{latin capital letter K}'
     ],
     LOWERCASE: [
         '\N{cyrillic small letter iotified little yus}',
         '\N{latin small letter K}'
     ]
     },
    {'name': 'юсъ малый',
     UPPERCASE: [
         '\N{cyrillic capital letter little yus}',
         '\N{latin capital letter Z}'
     ],
     LOWERCASE: [
         '\N{cyrillic small letter little yus}',
         '\N{latin small letter Z}'
     ]
     },
    {'name': 'кси',
     UPPERCASE: [
         '\N{cyrillic capital letter ksi}',
         '\N{latin capital letter X}'
     ],
     LOWERCASE: [
         '\N{cyrillic small letter ksi}',
         '\N{latin small letter X}'
     ]
     },
    {'name': 'пси',
     UPPERCASE: [
         '\N{cyrillic capital letter psi}',
         '\N{latin capital letter P}'
     ],
     LOWERCASE: [
         '\N{cyrillic small letter psi}',
         '\N{latin small letter P}'
     ]
     },
    {'name': 'фита', UPPERCASE: '\N{cyrillic capital letter fita}',
     LOWERCASE: '\N{cyrillic small letter fita}'},
    {'name': 'ижица',
     UPPERCASE: [
         '\N{cyrillic capital letter izhitsa}',
         '\N{latin capital letter V}'
     ],
     LOWERCASE: [
         '\N{cyrillic small letter izhitsa}',
         '\N{latin small letter V}'
     ]
     },
]

slavic_font_names = [
    'AkathUcs8.ttf',
    'bukvica.ttf',
    'evngucs.ttf',
    'FedorovskUnicode.otf',
    'feofanucs.ttf',
    'Flavius.ttf',
    'HirmUcs8.ttf',
    'IndUcs.ttf',
    'IrmUcs.ttf',
    'KathUcs8.ttf',
    'MenaionUnicode.otf',
    'OglUcs8.ttf',
    'OstgUcs8.ttf',
    'PochUcs.ttf',
    'PosUcs8.ttf',
    'PsalUcs.ttf',
    'SlavUcs.ttf',
    'StusUcs.ttf',
    'TriUcs.ttf',
    'VertUcs.ttf',
    'Vilnius-Regular.otf',
    'ZlatUcs.ttf'
]


def crop_white(img: Image.Image) -> Image.Image:
    np_img = 255 - np.asarray(img)
    top, bottom = None, None
    left, right = None, None
    for i, value in enumerate(np_img.sum(axis=1)):
        if value > 0 and top is None:
            top = i
        if value > 0 and bottom is not None:
            bottom = None
        if value == 0 and bottom is None and top is not None:
            bottom = i
    if bottom is None:
        bottom = np_img.shape[0]
    for i, value in enumerate(np_img.sum(axis=0)):
        if value > 0 and left is None:
            left = i
        if value > 0 and right is not None:
            right = None
        if value == 0 and right is None and left is not None:
            right = i
    if right is None:
        right = np_img.shape[1]
    if left is None or top is None:
        raise Exception(f'{left}-{right}-{top}-{bottom}')
    return Image.fromarray(255 - np_img[top:bottom, left:right])
