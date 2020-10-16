import numpy as np

def generate_birdcage_sensitivities(matrix_size = 256, number_of_coils = 8, relative_radius = 1.5, rotation = 0.0, normalize=True) -> np.ndarray:
    """ 
    Generates birdcage coil sensitivites.

    :param matrix_size: size of imaging matrix in pixels (default ``256``)
    :param number_of_coils: Number of simulated coils (default ``8``)
    :param relative_radius: Relative radius of birdcage (default ``1.5``)

    This function is heavily inspired by the mri_birdcage.m Matlab script in
    Jeff Fessler's IRT package: http://web.eecs.umich.edu/~fessler/code/
    """

    out = np.zeros((number_of_coils,matrix_size,matrix_size),dtype=np.complex64)
    for c in range(0,number_of_coils):
        coilx = relative_radius*np.cos(c*(2*np.pi/number_of_coils)+rotation)
        coily = relative_radius*np.sin(c*(2*np.pi/number_of_coils)+rotation)
        coil_phase = -c*(2*np.pi/number_of_coils)

        for y in range(0,matrix_size):
            y_co = float(y-matrix_size/2)/float(matrix_size/2)-coily
            for x in range(0,matrix_size):
                x_co = float(x-matrix_size/2)/float(matrix_size/2)-coilx
                rr = np.sqrt(x_co**2+y_co**2)
                phi = np.arctan2(x_co, -y_co) + coil_phase
                out[c,y,x] =  (1/rr) * np.exp(1j*phi)

    if normalize:
         rss = np.squeeze(np.sqrt(np.sum(abs(out) ** 2, 0)))
         out = out / np.tile(rss,(number_of_coils,1,1))

    return out


def phantom(matrix_size = 256, phantom_type = 'Modified Shepp-Logan', ellipses = None):
    """
    Create a Shepp-Logan, modified Shepp-Logan phantom, or random elipsis phantom::

        phantom (n = 256, phantom_type = 'Modified Shepp-Logan', ellipses = None)

    :param matrix_size: size of imaging matrix in pixels (default 256)

    :param phantom_type: The type of phantom to produce.
        Either "Modified Shepp-Logan", "Shepp-Logan", or "Random". This is overridden
        if ``ellipses`` is also specified.

    :param ellipses: Custom set of ellipses to use.  These should be in
        the form::

            [[I, a, b, x0, y0, phi],
            [I, a, b, x0, y0, phi],
                            ...]

        where each row defines an ellipse.

        :I: Additive intensity of the ellipse.
        :a: Length of the major axis.
        :b: Length of the minor axis.
        :x0: Horizontal offset of the centre of the ellipse.
        :y0: Vertical offset of the centre of the ellipse.
        :phi: Counterclockwise rotation of the ellipse in degrees,
            measured as the angle between the horizontal axis and
            the ellipse major axis.

    The image bounding box in the algorithm is ``[-1, -1], [1, 1]``,
    so the values of ``a``, ``b``, ``x0``, ``y0`` should all be specified with
    respect to this box.

    :returns: Phantom image

    References:

    Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
    from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
    Feb. 1974, p. 232.

    Toft, P.; "The Radon Transform - Theory and Implementation",
    Ph.D. thesis, Department of Mathematical Modelling, Technical
    University of Denmark, June 1996.
    """

    if (ellipses is None):
        ellipses = _select_phantom (phantom_type)
    elif (np.size (ellipses, 1) != 6):
        raise AssertionError ("Wrong number of columns in user phantom")

    ph = np.zeros ((matrix_size, matrix_size),dtype=np.float32)

    # Create the pixel grid
    ygrid, xgrid = np.mgrid[-1:1:(1j*matrix_size), -1:1:(1j*matrix_size)]

    for ellip in ellipses:
        I   = ellip [0]
        a2  = ellip [1]**2
        b2  = ellip [2]**2
        x0  = ellip [3]
        y0  = ellip [4]
        phi = ellip [5] * np.pi / 180  # Rotation angle in radians

        # Create the offset x and y values for the grid
        x = xgrid - x0
        y = ygrid - y0

        cos_p = np.cos (phi)
        sin_p = np.sin (phi)

        # Find the pixels within the ellipse
        locs = (((x * cos_p + y * sin_p)**2) / a2
        + ((y * cos_p - x * sin_p)**2) / b2) <= 1

        # Add the ellipse intensity to those pixels
        ph [locs] += I

    return ph

def phase_roll(matrix_size = 256, rotation = 0.0, center = (0.0, 0.0), roll=np.pi):
    def calc_phase(rel_x, rel_y, roll = np.pi, rotation = 0.0, center = (0,0)):
        rot_x = (rel_x-center[0])*np.cos(rotation) - (rel_y-center[1])*np.sin(rotation) + center[0]
        rot_y = (rel_x-center[0])*np.sin(rotation) + (rel_y-center[1])*np.cos(rotation) + center[1]
        return np.cos(rot_x*roll) + 1j*np.sin(rot_x*roll)

    phase = np.array([
        [calc_phase((x-matrix_size/2.0)/matrix_size,(y-matrix_size/2.0)/matrix_size, center=center, rotation=rotation,roll=roll)
             for x in range(matrix_size)]
             for y in range(matrix_size)])

    return phase

def random_phase_roll(matrix_size = 256):
    return phase_roll(matrix_size=matrix_size,
                      rotation=(np.random.random()-0.5)*np.pi,
                      center=(np.random.random()-0.5,np.random.random()-0.5),
                      roll=np.random.random()*np.pi*4)

def _select_phantom(name):
    if (name.lower() == 'shepp-logan'):
        e = _shepp_logan()
    elif (name.lower() == 'modified shepp-logan'):
        e = _mod_shepp_logan()
    elif (name.lower() == 'random'):
        e = _random_phantom()
    else:
        raise ValueError("Unknown phantom type: %s" % name)
    return e

def _shepp_logan():
    #  Standard head phantom, taken from Shepp & Logan
    return [[   2,   .69,   .92,    0,      0,   0],
            [-.98, .6624, .8740,    0, -.0184,   0],
            [-.02, .1100, .3100,  .22,      0, -18],
            [-.02, .1600, .4100, -.22,      0,  18],
            [ .01, .2100, .2500,    0,    .35,   0],
            [ .01, .0460, .0460,    0,     .1,   0],
            [ .02, .0460, .0460,    0,    -.1,   0],
            [ .01, .0460, .0230, -.08,  -.605,   0],
            [ .01, .0230, .0230,    0,  -.606,   0],
            [ .01, .0230, .0460,  .06,  -.605,   0]]

def _mod_shepp_logan():
    #  Modified version of Shepp & Logan's head phantom,
    #  adjusted to improve contrast.  Taken from Toft.
    return [[   1,   .69,   .92,    0,      0,   0],
            [-.80, .6624, .8740,    0, -.0184,   0],
            [-.20, .1100, .3100,  .22,      0, -18],
            [-.20, .1600, .4100, -.22,      0,  18],
            [ .10, .2100, .2500,    0,    .35,   0],
            [ .10, .0460, .0460,    0,     .1,   0],
            [ .10, .0460, .0460,    0,    -.1,   0],
            [ .10, .0460, .0230, -.08,  -.605,   0],
            [ .10, .0230, .0230,    0,  -.606,   0],
            [ .10, .0230, .0460,  .06,  -.605,   0]]

def _random_phantom():
    return [_random_ellipsis() for _ in range(np.random.randint(1,20))]

def _random_ellipsis():
    I = np.random.random()
    a = np.random.random()
    b = np.random.random()
    x0  = np.random.random()*(1-max(a,b))
    y0  = np.random.random()*(1-max(a,b))
    phi = (np.random.random()-0.5)*360.0
    return [I,a,b,x0,y0,phi]

def generate_sampling_pattern(dimensions, rate = 1, random = True, centerlines = 8, seed = None):
    """
    Creates a sampling pattern for undersampling MRI data.

    :param dimensions: tuple (size_ky, size_kx)
    :param rate: acceleration rate
    :param random: boolean
    :param centerlines: fully sampled lines in the center
    :param seed: random seed for random lines
    
    """
    if random == False and np.mod(dimensions[0], rate) != 0:
        raise Exeption('Acceleration k-y dimension must be a multiple of acceleration when unless random sampling')

    if random == False:
        lines = np.mod(range(dimensions[0]), rate) == 0
    else:
        if seed is not None:
            np.random.seed(seed)
        lines = np.random.random(dimensions[0]) <= (1.0/rate)

    if centerlines > 0:
        leftmargin = int(dimensions[0]/2-centerlines/2)
        rightmargin = leftmargin+centerlines
        lines[leftmargin:rightmargin] = 1

    sp = np.zeros(dimensions)
    sp[lines, :] = 1

    return sp