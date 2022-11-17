
import sys, os

import matplotlib.pyplot as plt
import numpy as np
import imageio
from glob import glob

def crop(image, size, center=None):
    """ Crop the image to a square of size size centered on center.
        If certer is None, the center of the image is used.
    """
    out = np.zeros((size, size))
    y0, x0 = center if center else (image.shape[0]//2, image.shape[1]//2)
    yf, xf = y0 + size//2, x0 + size//2
    yi, xi = y0 - size//2, x0 - size//2

    # print(f"yf={yf}, xf={xf}, yi={yi}, xi={xi} : MAX: {image.shape}")

    out_yi = 0
    if yi < 0:
        out_yi = abs(yi)
        out[0:out_yi, :] = 0
        yi = 0

    out_xi = 0
    if xi < 0:
        out_xi = abs(xi)
        out[:, 0:out_xi] = 0
        xi = 0

    out_yf = size
    if yf > image.shape[0]:
        out_yf = yf - image.shape[0]
        out[out_yf:, :] = 0
        yf = image.shape[0]

    out_xf = size
    if xf > image.shape[1]:
        out_xf = xf - image.shape[1]
        out[:, out_xf:] = 0
        xf = image.shape[1]

    # print(f"out_yi={out_yi}, out_xi={out_xi}, out_yf={out_yf}, out_xf={out_xf}")
    out[out_yi:out_yf, out_xi:out_xf] = image[yi:yf, xi:xf]
    # plt.imshow(out)
    # plt.show()

    return out

def find_center(image, template):
    """
    Find the center of the template in the image.
    """
    # Compute the correlation between the template and the image
    ft_image = np.fft.fft2(image)
    ft_template = np.fft.fft2(template, s=image.shape)
    ft_corr = ft_image * np.conj(ft_template)
    corr = np.fft.ifft2(ft_corr)
    corr2 = corr*np.conj(corr)
    # Find the location of the maximum correlation
    yloc, xloc = np.where(corr2 == corr2.max())

    return yloc[0]+template.shape[0]//2, xloc[0]+template.shape[1]//2

if __name__ == "__main__":
    """ Test the crop function.
        It expects a bash pattern as first argument and 
        the desired size as second.
    """

    args = sys.argv[1:]

    if len(args) < 2 or '-h' in args or '--help' in args:
        print("Usage: python crop.py <pattern> <size> [--replace|--out <crop_dir>] "
              "[--tmp <templates dir>] [--verbose|-v]")
        sys.exit(1)

    replace = False
    if "--replace" in args:
        replace = True
        args.remove("--replace")

    tmp_dir = None
    if "--tmp" in args:
        tmp_dir = args[args.index("--tmp") + 1]
        args.remove("--tmp")
        args.remove(tmp_dir)
        templates = os.listdir(tmp_dir)

    out_dir = None
    if "--out" in args:
        out_dir = args[args.index("--out") + 1]
        args.remove("--out")
        args.remove(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    verbose = True if "-v" in args or "--verbose" in args else False

    if len(args) > 2:
        print("Warning: ignoring extra arguments")

    pattern = args[0]
    size = int(args[1])
    center = None
    for file in glob(pattern):
        print("Processing file %s" % file) if verbose else None
        image = imageio.imread(file)
        if tmp_dir:
            tmp_basename = [t for t in templates if
                            t.startswith(os.path.split(os.path.basename(file)))][0]
            print(f"Template file {os.path.join(tmp_dir, tmp_basename)}") if verbose else None
            template = imageio.imread(os.path.join(tmp_dir, tmp_basename))
            center = find_center(image[:, :800], template)
            print(f"Center found at {center[::-1]}") if verbose else None
            if verbose:
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(image)
                axs[1].imshow(template)
                plt.show()

        cropped = crop(image, size, center)

        new_fn = file.replace(".png", "_crop.png") if not replace else file
        if out_dir:
            new_fn = os.path.join(out_dir, os.path.basename(new_fn))
        imageio.imwrite(new_fn, cropped.astype(np.uint16))
