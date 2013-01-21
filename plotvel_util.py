#!/usr/bin/env python
import sys
from math import ceil

import numpy as np
import matplotlib.pyplot as pl
import matplotlib.transforms as mtransforms

import barak.spec
from barak.plot import A4PORTRAIT
from barak.utilities import indexnear, between
from barak.convolve import convolve_psf
from barak.absorb import calc_Wr, calc_iontau, guess_logN_b
from barak.pyvpfit import readf26
from plotspec.utils import \
     process_options, plot_velocity_regions, process_Rfwhm, plot_tick_vel, \
     ATMOS, process_args, lines_from_f26

np.seterr(divide='ignore', invalid='ignore')
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


# read in one or more spectra.

usage = """Usage:  plotvel filename [redshift]

The file called plot.cfg contains key-value pairs of options.  All of
these can also be given a command line options on the form
key=value. In this case they override any values in the config file.
"""

help = """
space        Mark a new redshift position and redraw.
e            Measure the observed equivalent width over a region. Place the
             cursor on one region edge and press 'e', then repeat for the
             other edge.
s            Smooth spectrum.
b            Print fitting region at position of cursor.
l            Print line at position of cursor, guessing N and b.
H            Same as 'l', but the line is always HI.
p            Print the wavelength at the cursor position.

P            Make a plot of the current figure.

L            Add an HI Lya transition (only) at the cursor.
R            Re-read the f26 file and generate new models.

f5           Toggle tick labels on and off.
"""

wlya = 1215.6701
c_kms = 299792.458         # speed of light km/s, exact


def calc_abs(wa, ion, zp1, resolution, logN=13.0, b=10.0, maxdv=10000):
    """ Generate a normalised continuum over some wavelength range
    including absorption from an ion.

    Includes instrumental broadening. Splits the wavelength array up
    into smaller bins if necessary to sample the instrumental profile
    properly.

    Parameters
    ----------
    wa: array of floats
        The array of wavelengths (Angstroms) where the continuum will
        be generated.
    ion: atom.dat entry
        The full atom.dat list of transitions for given ion,
        i.e. at['HI']
    zp1: float
        One plus the redshift of the ion.
    resolution:
        The spectrum resolution.
    logN: float (13.0)
        log10 of column density in absorbers per cm**2.
    b: float (10.)
        b parameter in km/s of the ion.
    maxdv: float (1000)
        For performance reasons, only calculate the Voigt profile for
        a single line to +/- maxdv.  Increase this if you expect
        DLA-type extended wings.

    Returns
    -------
    co: array of floats
        normalised continuum including absorption
    """
    # show expected lines with sample N, b.

    # use smaller wave divisions so voigt profiles are sampled properly
    # new array
    wa = np.asarray(wa)
    dv = np.diff(wa).mean() / wa.mean() * c_kms
    ndiv = int(np.ceil(dv / (b / 2.)))
    print 'dividing', ndiv, 'times'
    n = len(wa)
    wa0 = np.interp(np.linspace(0, n - 1, (n - 1) * ndiv), range(n), wa)

    tau = calc_iontau(wa0, ion, zp1, logN, b, maxdv=maxdv)
    co0 = np.exp(-tau)

    # instrumental broadening
    fwhmwa = wa.mean() / resolution
    i = indexnear(wa, wa.mean())
    fwhmpix = fwhmwa / (wa[i + 1] - wa[i])
    print 'fwhmpix', fwhmpix
    co1 = convolve_psf(co0, fwhmpix * ndiv)

    # re-bin back to original wav array
    co2 = np.empty_like(wa)
    for i in range(len(wa)):
        co2[i] = co1[i * ndiv:i * ndiv + ndiv].mean()

    co = np.where(np.isnan(co2), 1, co2)
    return co


def initvelplot(wa, nfl, ner, nco, transitions, z, fig, atom,
                vmin=-1000., vmax=1000., nmodels=0,
                osc=False, residuals=False):
    """ Vertical stacked plots of expected positions of absorption
    lines at the given redshift on a velocity scale."""

    colours = ('b', 'r', 'g', 'orangered', 'c', 'purple')

    ions = [tr['name'].split()[0] for tr in transitions]
    # want an ordered set
    ionset = []
    for ion in ions:
        if ion not in ionset:
            ionset.append(ion)

    colour = dict(zip(ionset, colours * (len(ions) // len(colours) + 1)))

    zp1 = z + 1
    betamin = vmin / c_kms
    betamax = vmax / c_kms

    fig.subplots_adjust(wspace=0.0001, left=0.03, right=0.97, top=0.95,
                        bottom=0.07)
    ax1 = pl.subplot(122)
    ax2 = pl.subplot(121)
    axes = [ax1, ax2]
    for ax in axes:
        ax.set_autoscale_on(0)
    # plot top down, so we need reversed()
    offsets = []
    artists = dict(fl=[], er=[], co=[], resid=[], text=[], model=[], models=[],
                   ew=None, regions=[], ticklabels=[], ticks=[])

    num_per_panel = int(ceil(0.5 * len(transitions)))
    for i, trans in enumerate(reversed(transitions)):
        ax = ax1
        if i >= num_per_panel:
            i -= num_per_panel
            ax = ax2
        ion = trans['name'].split()[0]
        offset = i * 1.5
        #print i, offset
        offsets.append(offset)
        watrans = trans['wa']
        obswa = watrans * zp1
        wmin = obswa * (1 + 3 * betamin)
        wmax = obswa * (1 + 3 * betamax)

        cond = between(wa, wmin, wmax)
        #good = ~np.isnan(fl) & (er > 0) & ~np.isnan(co)

        fl = nfl[cond]
        #er = ner[cond]
        co = nco[cond]
        vel = (wa[cond] / obswa - 1) * c_kms
        #import pdb; pdb.set_trace()
        ax.axhline(offset, color='gray', lw=0.5)
        #artists['er'].extend(
        #    ax.plot(vel, er + offset, lw=1, color='orange', alpha=0.5) )
        artists['fl'].extend(
            ax.plot(vel, fl + offset, color=colour[ion], lw=0.5,
                    ls='steps-mid'))
        artists['co'].extend(
            ax.plot(vel, co + offset, color='gray', lw=0.5, ls='dashed'))

        if nmodels > 0:
            artists['models'].append([ax.plot(vel, co + offset, 'k', lw=0.2)[0]
                                      for j in range(nmodels)])
        artists['model'].extend(
            ax.plot(vel, co + offset, 'k', lw=0.5, zorder=12))
        if residuals:
            artists['resid'].extend(
                ax.plot([], [], '.', ms=3, mew=0, alpha=0.4, color='forestgreen'))
            #ax.axhline(offset-0.1, color='k', lw=0.3)
            ax.axhline(offset - 0.05, color='k', lw=0.3)
            ax.axhline(offset - 0.15, color='k', lw=0.3)
        bbox = dict(facecolor='w', edgecolor='None')
        transf = mtransforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        name = trans['name']
        if 'tr' in trans and osc:
            name = name + ' %.3g' % trans['tr']['osc']
        artists['text'].append(
            ax.text(0.03, offset + 0.5, name, fontsize=15, bbox=bbox,
                    transform=transf, zorder=20))

    for ax in axes:
        ax.axvline(0, color='k', lw=0.5, zorder=20)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(-0.5, num_per_panel * 1.5)
        ax.set_yticks([])
        ax.set_xlabel('Velocity offset (km s$^{-1}$)', fontsize=16)
    artists['title'] = pl.suptitle('$z = %.5f$' % z, fontsize=18)
    artists['sky'] = []

    return artists, np.array(offsets[:num_per_panel]), num_per_panel, axes


class VelplotWrap(object):
    def __init__(self, wa, nfl, ner, fig, filename, options, smoothby=1):
        self.opt = options
        self.filename = filename
        self.smoothby = smoothby
        self.cids = dict()
        # disable any existing key press callbacks
        cids = list(fig.canvas.callbacks.callbacks['key_press_event'])
        for cid in cids:
            fig.canvas.callbacks.disconnect(cid)
        self.connect(fig)
        self.wa = wa
        self.ner = ner
        self.nfl = nfl
        self.co = np.ones(len(wa))
        edges = barak.spec.find_wa_edges(wa)    # pixel edges
        dwa = edges[1:] - edges[:-1]            # width per pixel
        self.dwa = dwa
        self.ew = (1 - nfl) * dwa               # equiv. width
        self.ewer = dwa * ner
        self.fig = fig
        self.vmin = -options.dv
        self.vmax = options.dv
        self.prev_wa = None
        self.indprev = None

        dw = np.median(dwa)
        if options.wadiv is not None:
            dw1 = dw / options.wadiv
            wa1 = np.arange(wa[0], wa[-1] + 0.5 * dw1, dw1)
            self.wa1 = wa1

        self.taus = {}
        self.models = {}
        self.ticks = {}

        if self.opt.f26 is not None:
            self.update_model()
            self.convolve_LSF()
            self.apply_zero_offsets()

        artists, offsets, num_per_panel, axes = initvelplot(
            wa, nfl, ner, self.co, options.linelist, options.z, fig,
            options.atom, vmin=-options.dv, vmax=options.dv,
            nmodels=0, osc=options.show_oscillator_strength,
            residuals=options.residuals)
        self.offsets = offsets
        self.num_per_panel = num_per_panel
        self.artists = artists
        self.axes = axes
        self.update(options.z)
        self.prev_keypress = None

    def update_model(self):
        lines = lines_from_f26(self.opt.f26)

        wa = self.wa
        if self.opt.wadiv is not None:
            wa = self.wa1

        # first remove models that no longer present in lines
        for l in set(self.taus).difference(lines):
            del self.taus[l]
            del self.models[l]
            del self.ticks[l]

        dtype = np.dtype([('name', 'S10'), ('wa', 'f8'), ('z', 'f8'),
                          ('wa0', 'f8'), ('ind', 'i4')])
        # now add the new models from lines
        new_lines = set(lines).difference(self.taus)
        print len(new_lines), 'new lines'
        for l in new_lines:
            #print 'z, logN, b', z, logN, b
            ion, z, b, logN = l
            maxdv = 20000 if logN > 18 else 500
            t, tick = calc_iontau(wa, self.opt.atom[ion], z + 1, logN, b,
                                 ticks=True, maxdv=maxdv)
            self.taus[l] = t
            self.models[l] = np.exp(-t)
            temp = np.empty(0, dtype=dtype)
            if tick:
                ions = [ion] * len(tick)
                temp = np.rec.fromarrays([ions] + zip(*tick), dtype=dtype)
            self.ticks[l] = temp

        self.allticks = np.concatenate(
            [self.ticks[l] for l in self.ticks]).view(np.recarray)
        self.allticks.sort(order='wa')

        tau = np.zeros_like(wa)
        for line in self.taus:
            tau += self.taus[line]

        self.tau = tau
        self.model = np.exp(-tau)

    def convolve_LSF(self):
        print 'convolving'
        if self.opt.wadiv is not None:
            self.model, _ = process_Rfwhm(
                self.opt.Rfwhm, self.wa1, self.model, [])
        else:
            self.model, _ = process_Rfwhm(
                self.opt.Rfwhm, self.wa, self.model, [])

        if self.opt.wadiv is not None:
            self.model = np.interp(self.wa, self.wa1, self.model)
            #self.models = [np.interp(wa, self.wa1, m) for m in self.models]
        print 'done!'

    def apply_zero_offsets(self):
        l = self.opt.f26.lines
        regions = self.opt.f26.regions
        isort = regions.wmin.argsort()
        zeros = l[l.name == '__']
        print 'applying zero offsets for', len(zeros), 'regions'
        for val in zeros:
            wa = self.opt.atom['__'].wa[0] * (1 + val['z'])
            i0 = regions.wmin[isort].searchsorted(wa)
            i1 = regions.wmax[isort].searchsorted(wa)
            #print i0, i1
            #import pdb; pdb.set_trace()
            assert i0 - 1 == i1
            c0 = between(self.wa, regions.wmin[isort[i0 - 1]],
                         regions.wmax[isort[i1]])
            model = self.model[c0] * (1. - val['logN']) + val['logN']
            #import pdb; pdb.set_trace()
            self.model[c0] = model

    def update(self, z):
        if self.opt.f26 is not None:
            wa, nfl, ner, nco, model, artists, options = (
                self.wa, self.nfl, self.ner, self.co, self.model,
                self.artists, self.opt)
        else:
            wa, nfl, ner, nco, artists, options = (
                self.wa, self.nfl, self.ner, self.co,
                self.artists, self.opt)

        self.z = z

        zp1 = z + 1
        betamin = self.vmin / c_kms
        betamax = self.vmax / c_kms

        for t in artists['ticklabels']:
            t.remove()
        for t in artists['ticks']:
            t.remove()
        artists['ticklabels'] = []
        artists['ticks'] = []
        if artists['ew'] is not None:
            artists['ew'].remove()
            artists['ew'] = None
        for r in artists['regions']:
            r.remove()
        for s in artists['sky']:
            s.remove()
        artists['regions'] = []
        artists['sky'] = []

        # want plots to appear from  top down, so we need reversed()
        for i, trans in enumerate(reversed(options.linelist)):
            ax = self.axes[0]
            offset = i * 1.5
            if i >= self.num_per_panel:
                offset = (i - self.num_per_panel) * 1.5
                ax = self.axes[1]

            watrans = trans['wa']
            obswa = watrans * zp1
            wmin = obswa * (1 + 3 * betamin)
            wmax = obswa * (1 + 3 * betamax)
            if self.opt.f26 is not None:
                if self.opt.showticks:
                    ticks = self.allticks
                    tickwmin = obswa * (1 + betamin)
                    tickwmax = obswa * (1 + betamax)
                    wticks = ticks.wa
                    cond = between(wticks, tickwmin, tickwmax)
                    if cond.any():
                        vel = (wticks[cond] / obswa - 1) * c_kms
                        for j, t in enumerate(ticks[cond]):
                            T, Tlabels = plot_tick_vel(ax, vel[j], offset, t,
                                                      tickz=options.tickz)
                            artists['ticklabels'].extend(Tlabels)
                            artists['ticks'].extend(T)
            if options.f26 is not None and options.f26.regions is not None:
                artists['regions'].extend(plot_velocity_regions(
                    wmin, wmax, options.f26.regions.wmin,
                    options.f26.regions.wmax,
                    obswa, ax, offset))

            cond = between(wa, wmin, wmax)
            #good = ~np.isnan(fl) & (er > 0) & ~np.isnan(co)

            fl = nfl[cond]
            co = nco[cond]

            vel = (wa[cond] / obswa - 1) * c_kms
            vranges = []
            for w0, w1 in unrelated:
                c0 = between(wa[cond], w0, w1)
                if c0.any():
                    vranges.append(c0)

            for w0, w1 in ATMOS:
                c0 = between(wa[cond], w0, w1)
                if c0.any():
                    artists['sky'].append(
                        ax.fill_between(vel[c0], offset,
                                        offset + 1.5, facecolor='0.9', lw=0))

            if self.smoothby > 1:
                if len(fl) > 3 * self.smoothby:
                    fl = convolve_psf(fl, self.smoothby)

            artists['fl'][i].set_xdata(vel)
            artists['fl'][i].set_ydata(fl + offset)

            artists['co'][i].set_xdata(vel)
            artists['co'][i].set_ydata(co + offset)

            #pdb.set_trace()

            if self.opt.residuals:
                resid = (fl - model[cond]) / ner[cond]
                c0 = np.abs(resid) < 5
                artists['resid'][i].set_xdata(vel[c0])
                artists['resid'][i].set_ydata(resid[c0] * 0.05 + offset - 0.1)

            if self.opt.f26 is not None:
                artists['model'][i].set_xdata(vel)
                artists['model'][i].set_ydata(model[cond] + offset)

        for ax in self.axes:
            ax.set_xlim(self.vmin, self.vmax)
            ax.set_ylim(-0.5, self.num_per_panel * 1.5)
        self.artists['title'].set_text('$z = %.5f$' % self.z)

        if not self.opt.ticklabels:
            for t in artists['ticklabels']:
                t.set_visible(False)

        self.fig.canvas.draw()

        self.artists = artists

    def refresh_f26(self):
        name = self.opt.f26.filename
        print 're-reading from', name
        self.opt.f26 = readf26(name)
        self.opt.f26.filename = name

    def on_keypress(self, event):
        """ Process a keypress event.
        """
        if event.key == 'S':
            pl.savefig('junk.png', dpi=300)

        elif event.key == 'f5':
            if self.opt.ticklabels:
                for t in self.artists['ticklabels']:
                    t.set_visible(False)
                self.opt.ticklabels = False
            else:
                self.opt.ticklabels = True
                for t in self.artists['ticklabels']:
                    t.set_visible(True)

            self.fig.canvas.draw()

        elif event.key == 'R':
            self.refresh_f26()
            self.update_model()
            self.convolve_LSF()
            self.apply_zero_offsets()
            self.update(self.z)
        elif event.key == ' ' and event.inaxes is not None:
            z = self.z
            # get new redshift
            dz = (event.xdata / c_kms) * (1 + z)
            # and new axis limits, if any
            vmin, vmax = event.inaxes.get_xlim()
            self.vmin = min(0, vmin)
            self.vmax = max(0, vmax)
            self.update(z + dz)
        if event.key == 'z':
            c = raw_input('Enter redshift: ')
            self.update(float(c))
        if event.key in 'blepH':
            i = self.offsets.searchsorted(event.ydata)
            if i == 0:
                i = len(self.opt.linelist)
            off = self.offsets[i - 1]
            if event.inaxes == self.axes[1]:
                i += self.num_per_panel
            tr = self.opt.linelist[-i]
            z = (1 + event.xdata / c_kms) * (1 + self.z) - 1
            wa = tr['wa'] * (1 + z)
            ind = indexnear(self.wa, wa)
            #print 'transition', tr['name'], tr['wa'], wa
        if event.key == 'b':
            # print fitting region
            if self.prev_keypress == 'b' and self.prev_wa is not None:
                wmin = self.prev_wa
                wmax = wa
                if wmin > wmax:
                    wmin, wmax = wmax, wmin
                vsig = 6.38
                if isinstance(self.opt.Rfwhm, float):
                    vsig = self.opt.Rfwhm / 2.35
                print '%%%% %s 1 %.3f %.3f vsig=%.3f' % (
                    self.filename, wmin, wmax, vsig)
                self.prev_keypress = None
                self.prev_wa = None
            else:
                self.prev_wa = wa
        elif event.key == 'p':
            s = ''
            if self.opt.f26 is not None:
                i = indexnear(self.allticks.wa, wa)
                tick = self.allticks[i]
                s = ', closest tick: %-s %.1f z=%.5f' % (
                    tick['name'], tick['wa0'], tick['z'])
            print '%12s dv=%.1f wa=%.3f' % (tr['name'], event.xdata, wa) + s
        elif event.key == 'l':
            # guess a line N, z and b and print to screen need to know
            # which transition we want, and what the central index is.
            ner = self.ner[ind - 1:ind + 1]
            nfl = self.nfl[ind - 1:ind + 1]
            c0 = (ner > 0) & ~np.isnan(nfl)
            if not c0.sum():
                print 'No good pixels!'
                return
            f = np.median(nfl[c0])
            e = np.median(ner[c0])
            if f < e:
                f = e
            elif f > 1 - e:
                f = 1 - e
            tau0 = -np.log(f)
            logN, b = guess_logN_b(tr['name'].split()[0], tr['wa'], tr['tr'][1], tau0)
            print '%-6s %8.6f 0.0 %4.1f 0.0 %4.1f 0.0' % (
                tr['name'].split()[0], z, b, logN)
        elif event.key == 'H':
            # same as 'l' but always HI
            ner = self.ner[ind - 1:ind + 1]
            nfl = self.nfl[ind - 1:ind + 1]
            c0 = (ner > 0) & ~np.isnan(nfl)
            if not c0.sum():
                print 'No good pixels!'
                return
            f = np.median(nfl[c0])
            e = np.median(ner[c0])
            if f < e:
                f = e
            elif f > 1 - e:
                f = 1 - e
            tau0 = -np.log(f)
            logN, b = guess_logN_b('HI', 1215.6701, 0.416400, tau0)
            redshift = wa / 1215.6701 - 1.
            print '%-6s %8.6f 0.0 %4.1f 0.0 %4.1f 0.0' % ('HI', redshift, b, logN)
        elif event.key == 'e':
            wa = self.wa
            if self.indprev is not None:
                i0, i1 = self.indprev, ind
                if i0 > i1:
                    i0, i1 = i1, i0
                f = calc_Wr(i0, i1, wa, tr['tr'], self.ew, self.ewer)
                print '%s z=%.6f ngoodpix=%3i  wa=%.3f-%.3f ' % (
                    tr['name'], f.zp1 - 1, f.ngoodpix, wa[i0], wa[i1])
                print '   Wr=%.3f+/-%.3fA logN=%.4f (%.3f-%.3f) 5sig detect lim %.4f' % (
                    f.Wr, f.Wre, f.logN[1], f.logN[0], f.logN[2], f.Ndetlim)
                self.artists['ew'].remove()
                self.artists['ew'] = event.inaxes.fill_between(
                    (wa[i0:i1 + 1] / tr['wa'] / (1 + self.z) - 1) * c_kms,
                    self.nfl[i0:i1 + 1] + off, y2=off + 1,
                    color='r', alpha=0.5)
                pl.draw()
                self.indprev = None
                return
            else:
                if self.artists['ew'] is not None:
                    self.artists['ew'].remove()
                vpos = [(wa[ind] / tr['wa'] / (1 + self.z) - 1) * c_kms] * 2
                #print tr, [off, off+1], vpos
                self.artists['ew'], = event.inaxes.plot(
                    vpos, [off, off + 1], 'k', alpha=0.3)
                pl.draw()
                self.indprev = ind
        elif event.key == 'P':
            # save redshift, logN, and plot.
            f = self.filename.rsplit('.', 1)
            f = f[0] + '_z%.4f.' % self.z + f[1]
            print "printing to", f
            pl.savefig(f)
        elif event.key == 's':
            c = raw_input('New FWHM in pixels of Gaussian to convolve with? '
                          '(blank for no smoothing) ')
            if c == '':
                # restore spectrum
                self.smoothby = 1
            else:
                try:
                    fwhm = float(c)
                except TypeError:
                    print 'FWHM must be a float'
                if fwhm < 1:
                    self.smoothby = 1
                else:
                    self.smoothby = fwhm

            self.update(self.z)

        self.prev_keypress = event.key

    def connect(self, fig):
        cids = dict()
        cids['print'] = fig.canvas.mpl_connect(
            'key_press_event', self.on_keypress)
        self.cids.update(cids)


def main(args):
    Nargs = len(args)
    if Nargs == 0:
        print usage
        sys.exit(1)

    args, opt_args = process_args(args)
    options = process_options(opt_args)

    if options.z is None:
        if options.f26 is not None:
            options.z = np.median(options.f26.lines.z)
        else:
            options.z = 1

    filename = args[0]
    spec = barak.spec.read(filename)
    if np.isnan(spec.co).all():
        spec.co[:] = 1.

    fig = pl.figure(figsize=(12, 8))
    print help
    junk = VelplotWrap(spec.wa, spec.fl / spec.co, spec.er / spec.co, fig,
                       spec.filename, options)
    pl.show()
    return junk
