#!/usr/bin/env python
import os
from math import sqrt
from atpy import Table

import numpy as np
import matplotlib.pyplot as pl

import barak.spec
from barak.sed import get_SEDs
from barak.utilities import between, indexnear
from barak.convolve import convolve_psf
from barak.absorb import find_tau
from barak.plot import axvlines
from plotspec.utils import process_options, plotregions, process_Rfwhm, \
     plot_tick_wa, process_args, lines_from_f26

# read in one or more spectra.  left and right arrows to switch between them

np.seterr(divide='ignore', invalid='ignore')

usage = """\
Usage:  plotspec [options] filename1 filename2 ...

# name       wavelength
H-a          6564.0

and so on ...
"""

help = """Left and right arrows to switch between spectra.
i   Identify line to set redshift.
c   Fit a dodgy continuum.
m   Calculate S/N over a region.
v   Velocity plot (requires a continuum).
?   Print this message.

f5  Toggle tick labels.
"""

prefix = os.path.abspath(__file__).rsplit('/', 1)[0]
dla_spec = Table(prefix + '/dla.fits')


class Junk(object):
    def __init__(self, filenames, fig, options):
        """ Resolution is resolution FWHM in units of pixels.
        """
        self.opt = options
        self.filenames = list(filenames)
        self.i = 0
        if options.z is None:
            options.z = 0
        self.zp1 = options.z + 1
        self.n = len(self.filenames)
        self.spec = [None] * self.n          # cache spectra
        self.twa = []
        self.tfl = []
        self.models = [None] * self.n          # cache optical depths
        temp = range(1, len(self.opt.linelist) + 1)
        self.linehelp = '\n'.join('%i: %s %.4f' % (i, r['name'], r['wa']) for
                                  i, r in zip(temp, self.opt.linelist))
        self.linehelp += '\nEnter transition: '
        self.cids = dict()
        # disable any existing key press callbacks
        cids = list(fig.canvas.callbacks.callbacks['key_press_event'])
        for cid in cids:
            fig.canvas.callbacks.disconnect(cid)
        self.connect(fig)
        self.fig = fig
        self.ax = self.fig.add_subplot(111)
        self.ax.set_autoscale_on(False)
        self.artists = dict()
        self.artists['spec'] = None
        self.artists['zlines'] = []
        self.artists['mlines'] = []
        self.artists['ticklabels'] = []
        self.artists['ticks'] = []
        self.wlim1 = None
        self.prev_wa = None
        self.get_new_spec(0)
        print help

    def get_new_spec(self, i):
        """ Change to a new spectrum
        """
        self.i = i
        filename = self.filenames[i]
        if self.spec[i] is None:
            print 'Reading %s' % filename
            self.spec[i] = barak.spec.read(filename)
            self.models[i] = None
            self.ticks = None
            if self.opt.f26 is not None:
                self.calc_model()

        s = self.spec[i]
        self.ax.cla()
        co = s.co
        if self.opt.co_is_sky:
            co = s.co * np.median(s.fl) / np.median(s.co) * 0.1
        self.artists['lines'] = barak.spec.plot(
            s.wa, s.fl, s.er, co, ax=self.ax, show=0, yperc=0.90)
        if self.opt.smoothby > 1:
            sfl = convolve_psf(s.fl, self.opt.smoothby, edge='reflect')
            self.artists['lines'][0].set_data(s.wa, sfl)
        self.artists['template'], = self.ax.plot([], [], 'y')

    def calc_model(self):
        lines = lines_from_f26(self.opt.f26)
        wa = self.spec[self.i].wa
        dw = np.median(np.diff(wa))
        print 'finding tau'
        if self.opt.wadiv is not None:
            dw1 = dw / self.opt.wadiv
            wa1 = np.arange(wa[0], wa[-1] + 0.5 * dw1, dw1)
            tau, ticks = find_tau(wa1, lines, self.opt.atom)
        else:
            tau, ticks = find_tau(wa, lines, self.opt.atom)

        model = np.exp(-tau)
        # if we want to calculate the optical depth per line,
        # do it here.

        if self.opt.wadiv is not None:
            model, _ = process_Rfwhm(self.opt.Rfwhm, wa1, model, [])
        else:
            model, _ = process_Rfwhm(self.opt.Rfwhm, wa, model, [])

        if self.opt.wadiv is not None:
            model = np.interp(wa, wa1, model)

        self.models[self.i] = model
        self.ticks = ticks

    def apply_zero_offsets(self):
        model = self.spec[self.i].model
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
            c0 = between(self.spec[self.i].wa, regions.wmin[isort[i0 - 1]],
                         regions.wmax[isort[i1]])
            m = model[c0] * (1. - val['logN']) + val['logN']
            #import pdb; pdb.set_trace()
            model[c0] = m
        self.spec[self.i].model = model

    def update(self):
        i = self.i
        self.artists['mlines'] = []
        a = self.ax
        s = self.spec[i]
        wa, fl, er, co = s.wa, s.fl, s.er, s.co
        if self.models[i] is not None and (co > 0).any():
            temp = co * self.models[i]
            a.plot(wa, temp, color='orange')
        a.set_title(self.filenames[i])
        if self.zp1 is not None:
            self.plotlines(self.zp1)
        if self.ticks is not None:
            if not np.isnan(co).all() and not self.opt.co_is_sky:
                f = np.interp(self.ticks.wa, wa, co)
            else:
                f = np.interp(self.ticks.wa, wa, fl)

            height = 0.08 * np.percentile(fl, 90)
            for j, t in enumerate(self.ticks):
                if self.opt.showticks:
                    T, Tlabels = plot_tick_wa(a, t.wa, f[j], height, t,
                                     tickz=self.opt.tickz)
                    self.artists['ticks'].extend(T)
                    self.artists['ticklabels'].extend(Tlabels)

        if self.opt.f26 is not None and self.opt.f26.regions is not None:
            plotregions(a, self.opt.f26.regions.wmin,
                        self.opt.f26.regions.wmax)

        if self.opt.features is not None:
            f = self.opt.features
            sortfl = np.sort(fl[er > 0])
            ref = sortfl[int(len(sortfl) * 0.95)] - sortfl[int(len(sortfl) * 0.05)]
            wedge = np.concatenate([f.wa0, f.wa1])
            axvlines(wedge, ax=a, colors='k', alpha=0.7)
            temp = co[np.array([indexnear(wa, wav) for wav in f.wac])]
            ymin = temp + ref * 0.1
            ymax = ymin + ref * 0.2
            a.vlines(f.wac, ymin, ymax, colors='g')
            for j, f in enumerate(f):
                a.text(f['wac'], ymax[j] + ref * 0.05, f['num'], ha='center',
                       fontsize=12, color='g')

        fl = self.tfl * np.median(s.fl) / np.median(self.tfl)
        self.artists['template'].set_data(self.twa, fl)

        if not self.opt.ticklabels:
            for t in self.artists['ticklabels']:
                t.set_visible(False)
        barak.spec.plotatmos(a)
        pl.draw()

    def plotlines(self, zp1):
        for l in self.artists['zlines']:
            try:
                l.remove()
            except ValueError:
                # plot has been removed
                pass
        self.artists['zlines'] = barak.spec.plotlines(
            zp1 - 1, pl.gca(), lines=self.opt.linelist, labels=True)
        pl.draw()

    def on_keypress(self, event):
        if event.key == 'right':
            if self.i == self.n - 1:
                print 'At the last spectrum.'
                return
            self.artists['zlines'] = []
            self.get_new_spec(self.i + 1)
            self.update()

        elif event.key == 'left':
            if self.i == 0:
                print 'At the first spectrum.'
                return
            self.artists['zlines'] = []
            self.get_new_spec(self.i - 1)
            self.update()
        elif event.key == '?':
            print help
        elif event.inaxes is None:
            return
        elif event.key == ' ':
            print '%.4f  %i' % (
                event.xdata, indexnear(self.spec[self.i].wa, event.xdata))
        elif event.key == 'f5':
            if self.opt.ticklabels:
                for t in self.artists['ticklabels']:
                    t.set_visible(False)
                self.opt.ticklabels = False
            else:
                self.opt.ticklabels = True
                x0, x1 = self.ax.get_xlim()
                for t in self.artists['ticklabels']:
                    if x0 < t.get_position()[0] < x1:
                        t.set_visible(True)

            self.fig.canvas.draw()
        elif event.key == 'm':
            if self.wlim1 is not None:
                self.artists['mlines'].append(pl.gca().axvline(
                    event.xdata, color='k', alpha=0.3))
                pl.draw()
                w0, w1 = self.wlim1, event.xdata
                if w0 > w1:
                    w0, w1 = w1, w0
                sp = self.spec[self.i]
                good = between(sp.wa, w0, w1) & (sp.er > 0) & ~np.isnan(sp.fl)
                print ('median fl, rms fl, median er, pixel width, '
                       'SNR per pixel/Ang using rms, using er')
                if good.sum() < 2:
                    print 'Too few good pixels in range'
                    self.wlim1 = None
                    return
                medfl = np.median(sp.fl[good])
                stdfl = sp.fl[good].std()
                meder = np.median(sp.er[good])
                pixwidth = np.mean(sp.wa[good][1:] - sp.wa[good][:-1])
                mult = sqrt(1. / pixwidth)
                snr1 = medfl / stdfl
                snr2 = medfl / meder
                print '%9.3g, %.3g, %.3g, %.2g, %.2g/%.2g, %.2g/%.2g' % (
                    medfl, stdfl, meder, pixwidth, snr1, snr1 * mult,
                    snr2, snr2 * mult)
                self.wlim1 = None
            else:
                for l in self.artists['mlines']:
                    try:
                        l.remove()
                    except ValueError:
                        # plot has been removed
                        pass
                self.artists['mlines'].append(pl.gca().axvline(
                    event.xdata,  color='k', alpha=0.3))
                pl.draw()
                self.wlim1 = event.xdata
                print "press 'm' again..."

        elif event.key == 'c':
            # fit dodgy continuum
            sp = self.spec[self.i]
            co = barak.spec.find_cont(sp.fl)
            temp, = pl.gca().plot(sp.wa, co, 'm')
            pl.draw()
            c = raw_input('Accept continuum? (y) ')
            if (c + ' ').lower()[0] != 'n':
                print 'Accepted'
                sp.co = co
                self.update()
            else:
                temp.remove()

        elif event.key == 'b':
            # print fitting region
            wa = event.xdata
            if self.prev_wa != None:
                wmin = self.prev_wa
                wmax = wa
                if wmin > wmax:
                    wmin, wmax = wmax, wmin
                print '%%%% %s 1 %.3f %.3f vsig=x.x' % (self.filenames[self.i], wmin, wmax)
                self.prev_wa = None
            else:
                self.prev_wa = wa

        elif event.key == 'l':
            # print HI line
            wa = event.xdata
            z = wa / 1215.6701 - 1
            print '%-6s %8.6f 0.0 %3.0f 0.0 %4.1f 0.0' % ('HI', z, 20, 14.0)

        elif event.key == 't':
            # overplot a template
            c = '0'
            while c not in '1234':
                c = raw_input("""\
1: LBG
2: QSO
3: LRG
4: Starburst galaxy
""")
            temp = get_SEDs('LBG', 'lbg_em.dat')
            temp.redshift_to(self.zp1 - 1)
            self.twa = temp.wa
            self.tfl = temp.fl
            self.update()

        elif event.key == 'D':
            # print DLA spec (needs continuum)
            z = event.xdata / 1215.6701 - 1
            wa = dla_spec.wa * (1 + z)
            s = self.spec[self.i]
            fl = np.interp(wa, s.wa, s.co) * dla_spec.fl
            if self.artists['spec'] is not None:
                self.artists['spec'].remove()

            self.artists['spec'], = pl.plot(wa, fl, 'r')
            self.update()

    def on_keypress_plotz(self, event):
        ax = event.inaxes
        if ax is None:
            return
        if event.key == 'i':
            # id line to get redshift
            while True:
                c = raw_input(self.linehelp)
                print c
                try:
                    i = int(c) - 1
                    ion = self.opt.linelist[i]['name']
                    wa = self.opt.linelist[i]['wa']
                except (TypeError, IndexError):
                    continue
                else:
                    break
            zp1 = event.xdata / wa
            print 'z=%.3f, %s %.2f' % (zp1 - 1, ion, wa)
        else:
            return
        self.zp1 = zp1
        self.plotlines(zp1)

    def on_keypress_smooth(self, event):
        if event.key != 's':
            return
        s = self.spec[self.i]
        c = raw_input('New FWHM in pixels of Gaussian to convolve with? '
                      '(blank for no smoothing) ')
        if c == '':
            # restore spectrum
            self.opt.smoothby = 1
            sfl = s.fl
        else:
            try:
                fwhm = int(c)
            except TypeError:
                print 'FWHM must be an integer'
            if fwhm == 0:
                self.opt.smoothby = 1
            else:
                self.opt.smoothby = fwhm
            sfl = convolve_psf(s.fl, self.opt.smoothby, edge='reflect')

        self.artists['lines'][0].set_data(s.wa, sfl)
        pl.draw()

    def connect(self, fig):
        cids = dict()
        cids['misc'] = fig.canvas.mpl_connect(
            'key_press_event', self.on_keypress)
        cids['plotz'] = fig.canvas.mpl_connect(
            'key_press_event', self.on_keypress_plotz)
        cids['smooth'] = fig.canvas.mpl_connect(
            'key_press_event', self.on_keypress_smooth)

        self.cids.update(cids)


def main(args):
    if len(args) < 1:
        print usage
        return

    args, opt_args = process_args(args)
    options = process_options(opt_args)

    fig = pl.figure(figsize=(15, 6))
    fig.subplots_adjust(left=0.04, right=0.98)
    junk = Junk(args, fig, options)
    junk.update()
    pl.show()
