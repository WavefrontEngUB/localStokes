
import sys, os
import numpy as np
import imageio
from glob import glob

def crop(image, size, center=None):
    """ Crop the image to a square of size size centered on center.
        If certer is None, the center of the image is used.
    """
    y0, x0 = center if center else (image.shape[0]//2, image.shape[1]//2)
    yf, xf = y0 + size//2, x0 + size//2
    yi, xi = y0 - size//2, x0 - size//2

    return image[yi:yf, xi:xf]


if __name__ == "__main__":
    """ Test the crop function.
        It expects a bash pattern as first argument and 
        the desired size as second.
    """

    args = sys.argv[1:]

    if len(args) < 2:
        print("Usage: python crop.py <pattern> <size>")
        sys.exit(1)

    replace = False
    if args[2] == "--replace":
        replace = True
        args.remove("--replace")

    if len(args) > 2:
        print("Warning: ignoring extra arguments")

    pattern = args[0]
    size = int(args[1])

    for file in glob(pattern):
        print("Processing file %s" % file)
        image = imageio.imread(file)
        cropped = crop(image, size)
        new_fn = file.replace(".png", "_recrop.png") if not replace else file
        imageio.imwrite(new_fn, cropped)
