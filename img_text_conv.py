import logging
import re

import click
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from spellchecker import SpellChecker

logger = logging.getLogger(__name__)
logging.basicConfig()


@click.command()
@click.option('-i', '--input', help="Set path to the input image")
@click.option('-o', '--output', help="Set path to the output text file")
@click.option('--verbose', is_flag=True, help='Prints verbose output')
def main(input, output, verbose):
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    assert input is not None, 'Input file path is required'

    in_ext = input.split('.')[-1]
    if in_ext == 'pdf':
        pages = convert_from_path(input)
        result = '\n'.join([convert_image_to_text(np.array(page)) for page in pages])
    else:
        image = cv2.imread(input)
        assert image is not None, f'Cant read {input} as image'
        result = convert_image_to_text(image)
    if output:
        with open(output, 'w') as f:
            f.write(result)
    else:
        logger.info(result)


def image_preprocessing(image):
    # Remove horizontal lines
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    contours, hierarchy = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"{len(contours)} horizontal lines found")
    for c in contours:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    # Smoothing
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (1, 1), 0)

    return image


def spell_correction(text):
    spell = SpellChecker()

    def correct_word(match):
        word = match.group(0)
        if len(word) > 2 and not any(c.isupper() for c in word) and not any(c.isdigit() for c in word):
            corrected = spell.correction(word)
            if corrected != word:
                logger.debug(f"spellchecker change {word} to {corrected}")
            return corrected
        return word

    return re.sub(r'\w+', correct_word, text)


def convert_image_to_text(image):
    image = image_preprocessing(image)
    text = pytesseract.image_to_string(image)
    return spell_correction(text)


if __name__ == '__main__':
    main()
