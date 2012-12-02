from barak.pyvpfit import readf26
from barak.absorb import findtrans, readatom, find_tau
from barak.constants import c_kms
from barak.spec import make_constant_dv_wa_scale, convolve_constant_dv
from barak.convolve import convolve_psf

import pylab as pl
import numpy as np

f26 = readf26(f26name)
atomdat = readatom(molecules=1)
spfilename = ''
trfilename = ''
vmin = vmax = 399
wadiv = None
Rfwhm = 6.6
osc = False
residuals = True
redshift = 0.56

unrelated = []
#         ( 3863.451, 3865.529), 
#         ( 3855.399, 3859.330), 
#         ( 4075.662, 4076.668), 
#         ( 4079.430, 4082.960), 
#         ( 3906.116, 3908.287), 
#         ( 3898.097, 3899.218), 
#         ( 4509.955, 4512.281), 
#         ( 4503.099, 4507.387), 
#         ( 4532.218, 4544.106), 
#         ( 4314.625, 4315.922), 
#         ( 4317.588, 4321.694), 
#         ( 5021.751, 5023.484), 
#         ( 5994.466, 5995.031), 
#         ( 9063.195, 9064.171), 
#         ( 9071.901, 9073.122), 
#         ( 9074.830, 9075.766), 
#         ( 9077.149, 9078.126), 
#         ( 4209.905, 4211.247), 
#         ( 4009.795, 4011.467), 
#         ( 4019.675, 4021.803), 
#         ( 3882.978, 3884.388), 
#         ( 3878.539, 3880.001), 
#         (5017.818, 5019.365),
#         ]


def get_fig_axes(nrows, ncols, npar, width=12):
    fig = pl.figure(figsize=(width, width*nrows/ncols))    
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.95)
    axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(npar)]
    return fig, axes

def get_nrows_ncols(npar):
    nrows = max(int(np.sqrt(npar)), 1)
    ncols = nrows
    while npar > (nrows * ncols):
        ncols += 1

    return nrows, ncols

def plot_tick_vel(ax, vpos, offset, t, tickz=None, ticklabels=True):
    """ plot a single velocity tick
    """

    repr(tickz), repr(ticklabels)
    label = '%s %.0f %.2f' % (t.name, t.wa0, t.z)
    label = label.replace('NeVII', 'NeVIII')

    T = ax.plot(
        [vpos, vpos], [1.05 + offset, 1.4 + offset],
        color='k', alpha=0.7, lw=1.5)

    if ticklabels and ((tickz is not None and
                       not (1e-5 < abs(t.z - tickz) < 1e-2)) or tickz is None):
        T.append(ax.text(vpos, 1.05 + offset, label, rotation=60,
                    fontsize=8, va='bottom',alpha=0.7))
    return T


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
    for v0,v1 in zip(vel0, vel1):
        yoff = 1.1 + offset
        R, = ax.plot([v0, v1], [yoff,yoff],'r',lw=3, alpha=0.7)
        regions.append(R)

    return regions

def process_Rfwhm(Rfwhm, wa, model, models):
    """ Convolve the input models using the Rfwhm option

    Return the new models.

    wa:  wavelength array, shape (N,)
    model: model flux array, shape (N,)
    models: list of model flux arrays each with shape (N,)

    Rfwm is one of:

      'convolve_with_COS_FOS'
      a float

    Returns
    -------
    new_model, new_models
    """

    model_out = None
    models_out = []

    if Rfwhm is None:
        return model, models

    elif Rfwhm == 'convolve_with_COS_FOS':
        print 'convolving with COS/FOS instrument profile'
        model_out = convolve_with_COS_FOS(model, wa, wa[1] - wa[0])
        models_out = [convolve_with_COS_FOS(m, wa, wa[1] - wa[0]) for m
                      in models]

    elif isinstance(Rfwhm, float):
        print 'Convolving with fwhm %.2f km/s' % Rfwhm
        # use a pixel velocity width 4 times smaller than the FWHM
        ndiv = 4.
        wa_dv = make_constant_dv_wa_scale(wa[0], wa[-1], Rfwhm / ndiv)
        model_out = convolve_constant_dv(wa, model, wa_dv, ndiv)
        # do the same to every model if there's more than one
        for m in models:
            models_out.append(convolve_constant_dv(wa, m, wa_dv, ndiv))
    else:
        raise ValueError('Unknown value for Rfwhm option')

    return model_out, models_out

def read_transitions(filename, atomdat):
    print 'Reading transitions from', filename

    with open(filename) as fh:
        linelist = []
        for tr in fh:
            tr = tr.strip()
            if tr and not tr.startswith('#'):
                name, t = findtrans(tr, atomdat=atomdat))
                temp.append(dict(name=name, wa=t['wa'], )

    return linelist

if 1:
    linelist = read_transitions(trfilename)
    ntrans = len(linelist)
    sp = barak.spec.read(spfilename)
    wa = sp.wa
    nfl = sp.fl / sp.co
    ner = sp.fl / sp.co
    edges = barak.spec.find_wa_edges(wa)    # pixel edges
    dwa = edges[1:] - edges[:-1]             # width per pixel

    dw = np.median(dwa)
    if wadiv is not None:
        dw1 = dw / wadiv
        wa1 = np.arange(wa[0], wa[-1]+0.5*dw1, dw1)
        tau, ticks = find_tau(wa1, f26.lines, atomdat)
    else:
        tau, ticks = find_tau(wa, f26.lines, atomdat)

    model = np.exp(-tau)
    models = []

    #models = [np.exp(-t) for t in taus]

    if wadiv is not None:
        model, models = process_Rfwhm(Rfwhm, wa1, model, models)
    else:
        model, models = process_Rfwhm(Rfwhm, wa, model, models)

    if wadiv is not None:
        model = np.interp(wa, wa1, model)
        models = [np.interp(wa, wa1, m) for m in models]

if 0:
    # actual plotting
    nrows, ncols = get_nrows_ncols(ntrans)
    fig, axes = get_fig_axes(nrows, ncols, ntrans)

    colours = ('b','r', 'g', 'orangered', 'c', 'purple')
    ions = [tr['name'].split()[0] for tr in transitions]
    # want an ordered set
    ionset = []
    for ion in ions:
        if ion not in ionset:
            ionset.append(ion)

    colour = dict(zip(ionset, colours * (len(ions) // len(colours) + 1)))

    n = len(transitions)

    zp1 = redshift + 1
    betamin = vmin / c_kms
    betamax = vmax / c_kms

    fig.subplots_adjust(wspace=0.0001, left=0.03, right=0.97, top=0.95,
                        bottom=0.07)

    # plot top down, so we need reversed()
    for i,trans in enumerate(reversed(transitions)):
        ax = axes[i]
        ion = trans['name'].split()[0]
        watrans = trans['wa']
        obswa = watrans * zp1
        wmin = obswa * (1 + 3*betamin)
        wmax = obswa * (1 + 3*betamax)

        cond = between(wa, wmin, wmax)
        
        fl = nfl[cond]
        co = nco[cond]
        vel = (wa[cond] / obswa - 1) * c_kms
        ax.axhline(0, color='gray', lw=0.5)
        ax.plot(vel, fl, color=colour[ion], lw=0.5, ls='steps-mid')
        ax.plot(vel, co, color='gray', lw=0.5, ls='dashed')

        for m in models:
            ax.plot(vel, m, 'k', lw=0.2)

        ax.plot(vel, model, 'k', lw=0.5)

        if residuals:
            ax.plot([], [], '.', ms=1, color='k')
            ax.axhline(0.05, color='k', lw=0.3)
            ax.axhline(0.15, color='k', lw=0.3)

        bbox = dict(facecolor='w', edgecolor='None')
        transf = mtransforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        name = trans['name']
        if 'tr' in trans and osc:
            name = name + ' %.3g' % trans['tr']['osc']
        ax.text(0.03, 0.5, name, fontsize=15, bbox=bbox, transform=transf)

    for ax in axes:
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticklabels([])
        #ax.set_xlabel('Velocity offset (km s$^{-1}$)', fontsize=16)

    pl.suptitle('$z = %.5f$' % z, fontsize=18)    

    pl.show()
