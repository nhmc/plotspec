import os

import numpy as np
import matplotlib.transforms as mtransforms

from barak.absorb import findtrans, readatom
from barak.io import readtabfits, readtxt, parse_config
from barak.utilities import adict, get_data_path
from barak.pyvpfit import readf26
from barak.constants import c_kms
from barak.convolve import convolve_constant_dv
from barak.sed import make_constant_dv_wa_scale
from barak.convolve import convolve_psf

try:
    from COS import convolve_with_COS_FOS
except:
    convolve_with_COS_FOS = None

# regions of atmospheric absorption
ATMOS = [(5570, 5590),
         (5885, 5900),
         (6275, 6325),
         (6870, 6950),
         (7170, 7350),
         (7580, 7690),
         (8130, 8350),
         (8900, 9200),
         (9300, 9850),
         (11100, 11620),
         (12590, 12790),
         (13035, 15110),
         (17435, 20850),
         (24150, 24800)]


def lines_from_f26(f26):
    """ Convert a f26-file list of lines into a list we can pass to
    find_tau.
    """
    if f26 is None:
        return []

    if f26.lines is None:
        f26.lines = []

    lines = []
    for l in f26.lines:
        print l['name']
        if l['name'].strip() in ('<<', '>>'):
            #print "skipping!"
            continue
        lines.append((l['name'].replace(' ', ''), l['z'], l['b'], l['logN']))
    return lines


def plot_tick_vel(ax, vpos, offset, t, tickz=None):
    """ plot a single velocity tick
    """
    label = '%s %.0f %.3f' % (t['name'], t['wa0'], t['z'])
    label = label.replace('NeVII', 'NeVIII')

    T = ax.plot(
        [vpos, vpos], [1.05 + offset, 1.4 + offset],
        color='k', alpha=0.7, lw=1.5)

    Tlabels = []
    if (tickz is not None and
        not (1e-5 < abs(t['z'] - tickz) < 1e-2)) or tickz is None:
        Tlabels.append(ax.text(vpos, 1.05 + offset, label, rotation=60,
                               fontsize=8, va='bottom', alpha=0.7))

    return T, Tlabels


def plot_tick_wa(ax, wa, fl, height, t, tickz=None):
    """ plot a single tick on a wavelength scale
    """
    label = '%s %.0f %.3f' % (t.name, t.wa0, t.z)
    label = label.replace('NeVII', 'NeVIII')

    fl = fl * 1.1
    T = ax.plot([wa, wa], [fl, fl + height], color='k', alpha=0.7, lw=1.5)

    Tlabels = []
    if tickz is not None and not (1e-5 < abs(t.z - tickz) < 1e-2) or \
           tickz is None:
        Tlabels.append(ax.text(wa, fl + 1.4 * height, label, rotation=60,
                    fontsize=8, va='bottom', alpha=0.7))

    return T, Tlabels


def plotregions(ax, wmin, wmax):
    """ Plot a series of fitting regions on the matplotlib axes `ax`.
    """
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    regions = []
    for w0, w1 in zip(wmin, wmax):
        r, = ax.plot([w0, w1], [0.8, 0.8], color='r', lw=3, alpha=0.7,
                     transform=trans)
        regions.append(r)
    return regions


def plot_velocity_regions(wmin, wmax, w0, w1, obswa, ax, offset):
    """ wmin, wmax is minimum and maximum wavelengths of the plot.

    w0 and w1 are the min and max wavelengths of the fitting
    regions. obswa is the wavelength of the transition for this plot.
    """
    cond = ((w1 >= wmax) & (w0 < wmax)) | \
           ((w1 <= wmax) & (w0 >= wmin)) | \
           ((w1 > wmin) & (w0 <= wmin))
    regions = []

    if not cond.any():
        return regions

    vel0 = (w0[cond] / obswa - 1) * c_kms
    vel1 = (w1[cond] / obswa - 1) * c_kms
    for v0, v1 in zip(vel0, vel1):
        yoff = 1.1 + offset
        R, = ax.plot([v0, v1], [yoff, yoff], 'r', lw=3, alpha=0.7)
        regions.append(R)

    return regions


def print_example_options():
    print """\
    Rfwhm = 6.67
    features = features_filename
    f26 = lines.f26
    transitions = transitions/z0p56
    taulines = taulines_filename
    co_is_sky = False
    tickz = 0.567311
    ticklabels = True
    showticks = True
    z = 3
    wadiv = 6
    nsmooth = 1
    show_oscillator_strength = False
    dv = 1000
    residuals = False
    """

def process_options(opt_args):
    opt = adict()
    filename = os.path.abspath(__file__).rsplit('/', 1)[0] + '/default.cfg'
    opt = parse_config(filename)
    if os.path.lexists('./plot.cfg'):
        opt = parse_config('./plot.cfg', opt)

    opt.update(opt_args)

    opt.atom = readatom(molecules=True)
    if opt.Rfwhm is not None:
        if isinstance(opt.Rfwhm, basestring):
            if opt.Rfwhm == 'convolve_with_COS_FOS':
                if convolve_with_COS_FOS is None:
                    raise ValueError('convolve_with_COS_FOS() not available')
                print 'Using tailored FWHM for COS/FOS data'
                opt.Rfwhm = 'convolve_with_COS_FOS'
            elif opt.Rfwhm.endswith('fits'):
                print 'Reading Resolution FWHM from', opt.Rfwhm
                res = readtabfits(opt.Rfwhm)
                opt.Rfwhm = res.res / 2.354
            else:
                print 'Reading Resolution FWHM from', opt.Rfwhm
                fh = open(opt.Rfwhm)
                opt.Rfwhm = 1 / 2.354 * np.array([float(r) for r in fh])
                fh.close()
        else:
            opt.Rfwhm = float(opt.Rfwhm)

    if opt.features is not None:
        print 'Reading feature list from', opt.features
        opt.features = readtabfits(opt.features)

    if opt.f26 is not None:
        name = opt.f26
        print 'Reading ions and fitting regions from', name
        opt.f26 = readf26(name)
        opt.f26.filename = name

    if opt.transitions is not None:
        print 'Reading transitions from', opt.transitions
        fh = open(opt.transitions)
        trans = list(fh)
        fh.close()
        temp = []
        for tr in trans:
            tr = tr.strip()
            if tr and not tr.startswith('#'):
                junk = tr.split()
                tr = junk[0] + ' ' + junk[1]
                t = findtrans(tr, atomdat=opt.atom)
                temp.append(dict(name=t[0], wa=t[1][0], tr=t[1]))
        opt.linelist = temp
    else:
        #opt.linelist = readtxt(get_data_path() + 'linelists/galaxy_lines',
        #                names='wa,name,select')
        opt.linelist = readtxt(get_data_path() + 'linelists/qsoabs_lines',
                        names='wa,name,select')

    if opt.f26 is None and opt.taulines is not None:
        print 'Reading ions from', opt.taulines
        fh = open(opt.taulines)
        lines = []
        for row in fh:
            if row.lstrip().startswith('#'):
                continue
            items = row.split()
            lines.append([items[0]] + map(float, items[1:]))
        fh.close()
        opt.lines = lines

    return opt


def process_Rfwhm(Rfwhm, wa, model, models):
    """ Convolve the input models using the Rfwhm option

    Return the new models.

    wa:  wavelength array, shape (N,)
    model: model flux array, shape (N,)
    models: list of model flux arrays each with shape (N,)

    Rfwm is one of:

      'convolve_with_COS_FOS'
      a float
      an array floats with shape (N,)

    Returns
    -------
    new_model, new_models
    """

    model_out = None
    models_out = []

    #import pdb; pdb.set_trace()
    if Rfwhm is None:
        return model, models

    elif Rfwhm == 'convolve_with_COS_FOS':
        #print 'convolving with COS/FOS instrument profile'
        #import pdb; pdb.set_trace()
        model_out = convolve_with_COS_FOS(model, wa, use_COS_nuv=True)
        for m in models:
            #if m.min() < (1 - 1e-2):
            m = convolve_with_COS_FOS(m, wa, use_COS_nuv=True)
            models_out.append(m)

    elif isinstance(Rfwhm, float):
        #print 'Convolving with fwhm %.2f km/s' % Rfwhm
        # use a pixel velocity width 4 times smaller than the FWHM
        ndiv = 4.
        try:
            wa_dv = make_constant_dv_wa_scale(wa[0], wa[-1], Rfwhm / ndiv)
        except:
            import pdb
            pdb.set_trace()
        model_out = convolve_constant_dv(wa, model, wa_dv, ndiv)
        # do the same to every model if there's more than one
        for m in models:
            #if m.min() < (1 - 1e-2):
            m = convolve_constant_dv(wa, m, wa_dv, ndiv)
            models_out.append(m)
    else:
        raise ValueError('Unknown value for Rfwhm option')

    return model_out, models_out


def process_args(args):
    out = []
    option = {}
    for arg in args:
        if '=' in arg:
            key, val = arg.split('=')
            try:
                val = float(val)
            except ValueError:
                pass
            option[key] = val
        else:
            out.append(arg)

    return out, option
