import argparse
import matplotlib.pyplot as plt
import numpy as np


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


###############################################################################################################

class LogMeter(object):
    """Logging class used to count and stores aggregates and means"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


###############################################################################################################

def plot_scans_and_reconstructions(output, target):
    output1 = output.detach().cpu().numpy()
    target1 = target.detach().cpu().numpy()

    # Plotting the ground truth scans
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

    trgt_gt_1 = ((target1[0, :, :, :] - np.min(target1[0, :, :, :])) / (
        np.max(target1[0, :, :, :] - np.min(target1[0, :, :, :])))).copy()
    im1 = ax1.imshow(trgt_gt_1.transpose(1, 2, 0)[:, :, 0])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    trgt_gt_2 = ((target1[1, :, :, :] - np.min(target1[1, :, :, :])) / (
        np.max(target1[1, :, :, :] - np.min(target1[1, :, :, :])))).copy()
    im2 = ax2.imshow(trgt_gt_2.transpose(1, 2, 0)[:, :, 0])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    trgt_gt_3 = ((target1[2, :, :, :] - np.min(target1[2, :, :, :])) / (
        np.max(target1[2, :, :, :] - np.min(target1[2, :, :, :])))).copy()
    im3 = ax3.imshow(trgt_gt_3.transpose(1, 2, 0)[:, :, 0])
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Plotting the reconstructed scans
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

    recon_1 = ((output1[0, :, :, :] - np.min(output1[0, :, :, :])) / (
        np.max(output1[0, :, :, :] - np.min(output1[0, :, :, :])))).copy()
    im1 = ax1.imshow(recon_1.transpose(1, 2, 0)[:, :, 0])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    recon2 = ((output1[1, :, :, :] - np.min(output1[1, :, :, :])) / (
        np.max(output1[1, :, :, :] - np.min(output1[1, :, :, :])))).copy()
    im2 = ax2.imshow(recon2.transpose(1, 2, 0)[:, :, 0])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    recon3 = ((output1[2, :, :, :] - np.min(output1[2, :, :, :])) / (
        np.max(output1[2, :, :, :] - np.min(output1[2, :, :, :])))).copy()
    im3 = ax3.imshow(recon3.transpose(1, 2, 0)[:, :, 0])
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


###############################################################################################################

def phantom(n=256, p_type='Modified Shepp-Logan', ellipses=None):
    """
     phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)

    Create a Shepp-Logan or modified Shepp-Logan phantom.

    A phantom is a known object (either real or purely mathematical)
    that is used for testing image reconstruction algorithms.  The
    Shepp-Logan phantom is a popular mathematical model of a cranial
    slice, made up of a set of ellipses.  This allows rigorous
    testing of computed tomography (CT) algorithms as it can be
    analytically transformed with the radon transform (see the
    function `radon').

    Inputs
    ------
    n : The edge length of the square image to be produced.

    p_type : The type of phantom to produce. Either
      "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
      if `ellipses' is also specified.

    ellipses : Custom set of ellipses to use.  These should be in
      the form
          [[I, a, b, x0, y0, phi],
           [I, a, b, x0, y0, phi],
           ...]
      where each row defines an ellipse.
      I : Additive intensity of the ellipse.
      a : Length of the major axis.
      b : Length of the minor axis.
      x0 : Horizontal offset of the centre of the ellipse.
      y0 : Vertical offset of the centre of the ellipse.
      phi : Counterclockwise rotation of the ellipse in degrees,
            measured as the angle between the horizontal axis and
            the ellipse major axis.
      The image bounding box in the algorithm is [-1, -1], [1, 1],
      so the values of a, b, x0, y0 should all be specified with
      respect to this box.

    Output
    ------
    P : A phantom image.

    Usage example
    -------------
      import matplotlib.pyplot as pl
      P = phantom ()
      pl.imshow (P)

    References
    ----------
    Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
    from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
    Feb. 1974, p. 232.

    Toft, P.; "The Radon Transform - Theory and Implementation",
    Ph.D. thesis, Department of Mathematical Modelling, Technical
    University of Denmark, June 1996.

    """

    ## Copyright (C) 2010  Alex Opie  <lx_op@orcon.net.nz>
    ##
    ## This program is free software; you can redistribute it and/or modify it
    ## under the terms of the GNU General Public License as published by
    ## the Free Software Foundation; either version 3 of the License, or (at
    ## your option) any later version.
    ##
    ## This program is distributed in the hope that it will be useful, but
    ## WITHOUT ANY WARRANTY; without even the implied warranty of
    ## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
    ## General Public License for more details.
    ##
    ## You should have received a copy of the GNU General Public License
    ## along with this program; see the file COPYING.  If not, see
    ## <https://urldefense.proofpoint.com/v2/url?u=http-3A__www.gnu.org_licenses_&d=DwIGAg&c=-35OiAkTchMrZOngvJPOeA&r=vhoDr5aF9OD2iT57HuYhi_snow1O_xxPTTiW0lJoQpc&m=JE3ZbDt-JQ5v6oF7wYj_3Et5nhBJAd6gmIM1dYvI1-5INn0y7lUKCdGDYDTjfpps&s=N-tb2eLvEqp5Y8uz4vPnByxHY1C6qfmGfL5Ir0clNY0&e= >.

    if (ellipses is None):
        ellipses = _select_phantom(p_type)
    elif (np.size(ellipses, 1) != 6):
        raise AssertionError("Wrong number of columns in user phantom")

    # Blank image
    p = np.zeros((n, n))

    # Create the pixel grid
    ygrid, xgrid = np.mgrid[-1:1:(1j * n), -1:1:(1j * n)]

    for ellip in ellipses:
        I = ellip[0]
        a2 = ellip[1] ** 2
        b2 = ellip[2] ** 2
        x0 = ellip[3]
        y0 = ellip[4]
        phi = ellip[5] * np.pi / 180  # Rotation angle in radians

        # Create the offset x and y values for the grid
        x = xgrid - x0
        y = ygrid - y0

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # Find the pixels within the ellipse
        locs = (((x * cos_p + y * sin_p) ** 2) / a2
                + ((y * cos_p - x * sin_p) ** 2) / b2) <= 1

        # Add the ellipse intensity to those pixels
        p[locs] += I

    return p


def _select_phantom(name):
    if (name.lower() == 'shepp-logan'):
        e = _shepp_logan()
    elif (name.lower() == 'modified shepp-logan'):
        e = _mod_shepp_logan()
    else:
        raise ValueError("Unknown phantom type: %s" % name)

    return e


def _shepp_logan():
    #  Standard head phantom, taken from Shepp & Logan
    return [[2, .69, .92, 0, 0, 0],
            [-.98, .6624, .8740, 0, -.0184, 0],
            [-.02, .1100, .3100, .22, 0, -18],
            [-.02, .1600, .4100, -.22, 0, 18],
            [.01, .2100, .2500, 0, .35, 0],
            [.01, .0460, .0460, 0, .1, 0],
            [.02, .0460, .0460, 0, -.1, 0],
            [.01, .0460, .0230, -.08, -.605, 0],
            [.01, .0230, .0230, 0, -.606, 0],
            [.01, .0230, .0460, .06, -.605, 0]]


def _mod_shepp_logan():
    #  Modified version of Shepp & Logan's head phantom,
    #  adjusted to improve contrast.  Taken from Toft.
    return [[1, .69, .92, 0, 0, 0],
            [-.80, .6624, .8740, 0, -.0184, 0],
            [-.20, .1100, .3100, .22, 0, -18],
            [-.20, .1600, .4100, -.22, 0, 18],
            [.10, .2100, .2500, 0, .35, 0],
            [.10, .0460, .0460, 0, .1, 0],
            [.10, .0460, .0460, 0, -.1, 0],
            [.10, .0460, .0230, -.08, -.605, 0],
            [.10, .0230, .0230, 0, -.606, 0],
            [.10, .0230, .0460, .06, -.605, 0]]


# def ?? ():
#	# Add any further phantoms of interest here
#	return np.array (
#	 [[ 0, 0, 0, 0, 0, 0],
#	  [ 0, 0, 0, 0, 0, 0]])

###############################################################################################################

def phantom_modified(n=256, p_type='Modified Shepp-Logan', ellipses=None):
    """
        phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)

    Create a Shepp-Logan or modified Shepp-Logan phantom.

    A phantom is a known object (either real or purely mathematical)
    that is used for testing image reconstruction algorithms.  The
    Shepp-Logan phantom is a popular mathematical model of a cranial
    slice, made up of a set of ellipses.  This allows rigorous
    testing of computed tomography (CT) algorithms as it can be
    analytically transformed with the radon transform (see the
    function `radon').

    Inputs
    ------
    n : The edge length of the square image to be produced.

    p_type : The type of phantom to produce. Either
        "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
        if `ellipses' is also specified.

    ellipses : Custom set of ellipses to use.  These should be in
        the form
        [[I, a, b, x0, y0, phi],
            [I, a, b, x0, y0, phi],
            ...]
        where each row defines an ellipse.
        I : Additive intensity of the ellipse.
        a : Length of the major axis.
        b : Length of the minor axis.
        x0 : Horizontal offset of the centre of the ellipse.
        y0 : Vertical offset of the centre of the ellipse.
        phi : Counterclockwise rotation of the ellipse in degrees,
            measured as the angle between the horizontal axis and
            the ellipse major axis.
        The image bounding box in the algorithm is [-1, -1], [1, 1],
        so the values of a, b, x0, y0 should all be specified with
        respect to this box.

    Output
    ------
    P : A phantom image.

    Usage example
    -------------
        import matplotlib.pyplot as pl
        P = phantom ()
        pl.imshow (P)

    References
    ----------
    Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
    from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
    Feb. 1974, p. 232.

    Toft, P.; "The Radon Transform - Theory and Implementation",
    Ph.D. thesis, Department of Mathematical Modelling, Technical
    University of Denmark, June 1996.

    """

    ## Copyright (C) 2010  Alex Opie  <lx_op@orcon.net.nz>
    ##
    ## This program is free software; you can redistribute it and/or modify it
    ## under the terms of the GNU General Public License as published by
    ## the Free Software Foundation; either version 3 of the License, or (at
    ## your option) any later version.
    ##
    ## This program is distributed in the hope that it will be useful, but
    ## WITHOUT ANY WARRANTY; without even the implied warranty of
    ## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
    ## General Public License for more details.
    ##
    ## You should have received a copy of the GNU General Public License
    ## along with this program; see the file COPYING.  If not, see
    ## <https://urldefense.proofpoint.com/v2/url?u=http-3A__www.gnu.org_licenses_&d=DwIGAg&c=-35OiAkTchMrZOngvJPOeA&r=vhoDr5aF9OD2iT57HuYhi_snow1O_xxPTTiW0lJoQpc&m=JE3ZbDt-JQ5v6oF7wYj_3Et5nhBJAd6gmIM1dYvI1-5INn0y7lUKCdGDYDTjfpps&s=N-tb2eLvEqp5Y8uz4vPnByxHY1C6qfmGfL5Ir0clNY0&e= >.

    if (ellipses is None):
        ellipses = _select_phantom(p_type)
    elif (np.size(ellipses, 1) != 6):
        raise AssertionError("Wrong number of columns in user phantom")

    # Blank image
    p = np.zeros((n, n))

    # Create the pixel grid
    ygrid, xgrid = np.mgrid[-1:1:(1j * n), -1:1:(1j * n)]

    for ellip in ellipses:
        p_temp = np.zeros((n, n))

        I = ellip[0]
        a2 = ellip[1] ** 2
        b2 = ellip[2] ** 2
        x0 = ellip[3]
        y0 = ellip[4]
        phi = ellip[5] * np.pi / 180  # Rotation angle in radians

        # Create the offset x and y values for the grid
        x = xgrid - x0
        y = ygrid - y0

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # Find the pixels within the ellipse
        locs = (((x * cos_p + y * sin_p) ** 2) / a2
                + ((y * cos_p - x * sin_p) ** 2) / b2) <= 1

        # Add the ellipse intensity to those pixels
        p_temp[locs] = I
        p_temp[p > 0.01] = 0
        p += p_temp
    return p


###############################################################################################################

def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img
