from pathlib import PurePath
from typing import Union, Optional, Iterable

from PIL import ImageFont, Image, ImageDraw
from fontTools.ttLib import TTFont


class Font:
    def __init__(self, path: Union[str, PurePath], name: str, size: int):
        self.ttFont = TTFont(str(path))
        self.name = name
        self.imageFont = ImageFont.truetype(str(path), size=size)
        self.size = size

    def try_draw_glyph(self, glyphs: Iterable[str], width: int = None, height: int = None) -> Optional[Image.Image]:
        if width is None:
            width = self.size * 2
        if height is None:
            height = self.size * 2
        for glyph in glyphs:
            if self.has_glyph(glyph):
                img = Image.new('L', (width, height), color=255)
                d = ImageDraw.Draw(img)
                w, h = d.textsize(glyph, font=self.imageFont)
                point = (width - w) // 2, (height - h) // 2
                d.text(point, glyph, font=self.imageFont, stroke_fill=0, align="center")
                return img
        return None

    def has_glyph(self, glyph: str):
        """Found here:
        https://stackoverflow.com/questions/47948821/python-imagefont-and-imagedraw-check-font-for-character-support"""
        for table in self.ttFont['cmap'].tables:
            if ord(glyph) in table.cmap.keys():
                return True
        return False
