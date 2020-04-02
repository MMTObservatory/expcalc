#!./env/bin/python
#!/usr/bin/python
#Exposure time / Signal to Noise calculator for the red and blue channel spectrographs on the MMT.
#This is Version 0.0 of the code - expect substantial structural changes to integrate with the PHP
#for use online.

#For now, this *only* works for blue channel.  More additions to change.

#This file uses the command line inputs. The calling proceedure is
#spectime.py INSTRUMENT OUTTYPE DEPTH grating ORDER CENWAE FILTER SPATIALBINNING
#   SPECTRALBINNING SLIT SEEING ABMAG LUNARPHASE airmass
#
# OUTTYPE - 1 means to calcualte the time to a given S/N (depth)
#           2 means to caclculate snr given an exposure time (depth)


#The parameters are read from the text file spectime.par

import numpy as np
import os
import sys
import scipy.special
from scipy.interpolate import interp1d
import math
import json
import re
import datetime

#from  pylab import *


def extrap1d(interpolator):
    #This function takes the outputs of numpy's interpolator and extrapolates

    xs = interpolator.x
    ys = interpolator.y
    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))
    return ufunclike

def interpol(newx, oldx, oldy):

    xx = np.arange(3000, 1.1e4)
    f_i = interp1d(oldx, oldy)
    f_x = extrap1d(f_i)

    newy = f_x(newx)
    #By design for this code, kill negatives
    for ii in range(0, len(newy)):
        if newy[ii] < 0 : newy[ii] = 1e-8

    return newy

def getFilterCorrection(opt, lam):
    #Read the filter information for the given filter and return a
    #vector based on wavelength lam that interpolates and
    #(smartly) extrapolates the filter over the full range.
    filterName = opt['filter']

    #Read the proper throughput data
    if filterName == "Clear":
        filtThrough = []
        filtLam = np.arange(3000, 1e4) # we don't have a file for clear, so just construct it
        for ii in range(0, len(filtLam)):
            filtThrough.append(100.0)
    else:
        filtFilename = 'data/spectrodata/filters/' + filterName + '.dat'
        filtLam, filtThrough = readSpectroDataFile(filtFilename)

    #Now, we just need to interpolate through the array to lam
    filtInterpol = interpol(lam, filtLam, filtThrough)


    #We can do this because we sort the vectors
    filtInterpol = cleanThroughputCorrections(lam, filtInterpol, filtLam, filtThrough)


    return filtInterpol/100.


def getBlazeCorrection(opt, lam):
    #This function will look at the grating requested and make a correction between that and the nominal grating
    #that the throughput was defined for and calculate the correction term for the blaze function.

    instr =  opt['spectrograph']
    grating = opt['grating']

    referenceGrating = '300' #True for both red and blue

    #Trim 'gpm' from the grating name
    grating = re.sub('gpm', '', grating)

    #Check to see if this is either of the two gratings we use in 2nd order.
    if grating == '600' and opt['order'] == 2 :
        grating = '600_2nd'
    if grating == '832' and opt['order'] == 2:
        grating = '832_2nd'

    blazeFilename = 'data/spectrodata/grating/' + instr + '_' + grating + '.dat'
    refFilename = 'data/spectrodata/grating/' + instr + '_' + referenceGrating + '.dat'

    blazeLam, blazeThrough = readSpectroDataFile(blazeFilename)
    refLam, refThrough = readSpectroDataFile(refFilename)

    #Need to do the interpolation and extrapolation before the ratio to mae sure we're on the same
    #pixel coordinates
    blazeInterpol = interpol(lam, blazeLam, blazeThrough)
    blazeInterpol = cleanThroughputCorrections(lam, blazeInterpol, blazeLam, blazeThrough)

    refInterpol = interpol(lam, refLam, refThrough)
    refInterpol = cleanThroughputCorrections(lam, refInterpol, refLam, refThrough)

    #Now, we need to multiply by the blazeInterpol, but divide by the reference

    return blazeInterpol / refInterpol



def cleanThroughputCorrections(lam, fullInterpol, fullLam, fullThrough):
    #This is just a function to do a repeated task of extrapolating throughput currves

    for ii in range(0, len(lam)):
        if (lam[ii] < np.amin(fullLam)):
            fullInterpol[ii] = fullThrough[0]
        if (lam[ii] > np.amax(fullLam)):
            fullInterpol[ii] = fullThrough[-1]
        if (fullInterpol[ii] <= 0):
            fullInterpol[ii] = 1e-9 #I do this to remove divide by zero errors
    return fullInterpol


def readSpectroDataFile(filename):
    #Quick wrapper to read the points for the spectrograph data (x,y) whitespace separated.

    f = open(filename, 'r')

    xx = []
    yy = []

    for line in f.readlines():
        splitArray = line.split();
        if len(splitArray) == 2:
            x1, y1 = map(float, splitArray)
            xx.append(x1)
            yy.append(y1)

    xx = np.array(xx)
    yy = np.array(yy)

    order = np.argsort(xx)

    return xx[order], yy[order]


def read_options(filename):
    #Read the filename into a dictionary. The assumed format is white space separation
    #with no spaces in any of the keyword names

    options = {}
    with open(filename) as f:
        for line in f:
            (key, val) = line.split()
            options[key] = val
    return options

def read_skymodels(opt):
    #Read the ascii files with each of the sky models (for different lunar phases) in them
    #Based on the set options, interpolate the models to the correct lunar phase and
    #return

    files = ['0','39','90','129','180']
    files = ['data/sky/sky_' + f+ '.dat' for f in files]
    lunarphase = [0, 3.5, 7, 10.5, 14]
    index = [0, 1, 2, 3, 4]

    junk = np.interp(float(opt['lunar_phase']), lunarphase, index)
    below = int(math.floor(junk))

    if below == junk:
        #Since the index was exactly the same as one of the input indexes, then
        #we can simply read the proper file.

        lam = []
        flux = []
        with open(files[below]) as f:
            for line in f:
                (val1, val2) = line.split()
                lam.append(float(val1)*1e4)
                flux.append(float(val2))
    else :
        top = int(math.ceil(junk))
        lam = []
        flux = []
        flux1 = []
        with open(files[below]) as f:
            for line in f:
                (val1, val2) = line.split()
                lam.append(float(val1)*1e4)
                flux1.append(float(val2))
                flux.append(0.0)

        lam2 = []
        flux2 = []
        with open(files[top]) as f:
            for line in f:
                (val1, val2) = line.split()
                lam2.append(float(val1)*1e4)
                flux2.append(float(val2))



        for ii in range(len(flux1)):
            flux[ii] = (flux1[ii]-flux1[ii]) / (lunarphase[top]-lunarphase[below]) * (float(opt['lunar_phase'])-lunarphase[below]) + flux1[ii]

    return [lam,flux]

def grating_params(opt):

    #For each grating  / order pair, return the R value at blaze and
    #the dispersion of the spectrograph

    spec = opt['spectrograph']
    grat = opt['grating']

    if spec == 'blue':
        if grat == '300gpm':
            return [740, 1.96]
        if grat == '500gpm':
            return [1430, 1.19]
        if grat == '800gpm':
            return [1730, 0.75]
        if grat == '600gpm':
            if int(opt['order']) == 2 : return [3310, 0.5]
            if int(opt['order']) == 1 : return [3310, 1.0]
        if grat == '832gpm':
            if int(opt['order']) == 2 : return [3830, 0.36]
            if int(opt['order']) == 1 : return [3830, 0.72]
        if grat == '1200gpm':
            return [3340, 0.5]

    if spec == 'red':
        if grat == '270gpm':
            return [640,3.59]
        if grat == '150gpm' :
            return [230,6.37]
        if grat == '300gpm' :
            return [460,3.21]
        if grat == '600_4800':
            return [960,1.63]
        if grat == '600_6310':
            return [1290,1.64]
        if grat == '1200_7700':
            return [3930,0.80]
        if grat == '1200_9000':
            return [5097,0.78]



def gauss_convol(xx, yy, sig):
    #Convolve a funtion with a gaussian
    x = np.array(xx)
    y = np.array(yy)
    sig = float(sig)

    nsigma=2.5
    nsigma2 = nsigma*2
    n = len(x)
    conv = (np.amax(x)-np.amin(x))/(n-1)
    n_pts = math.ceil(nsigma2*sig/conv)

    #print "n_pts = %s" % n_pts

    #Make some restrictions on the number of points in the kernel
    if n_pts < 2 : n_pts = 2
    if n_pts > (n-2) : n_pts = (n-2)
    if (n_pts/2*2) == n_pts :
        n_pts = n_pts + 1

    n_pts = int(n_pts) # cast it to int, otherwise boom.
        
    xvar = (np.arange(n_pts, dtype='d') / (n_pts -1) - 0.5)*n_pts
    gaus = np.exp(-0.5*(xvar/(sig/conv)**2))
    value = np.convolve(y, gaus/np.sum(gaus), 'same')

    #print "value:%s\n" % value
    #print "n_pts:%s\n" % n_pts
    
    
    value[0:n_pts] = y[0:n_pts]
    value[-n_pts:] = y[-n_pts:]


    return value

def get_lam(opt):

    #Given the dispersion, determine the wavelength scale
    grating_param = grating_params(opt)
    disperse = grating_param[1]*float(opt['binning_spectral'])
    npix = 2048 / float(opt['binning_spectral'])

    wave = (np.arange(0, npix) - (npix/2.))*disperse + float(opt['cenwave'])

    #Limit to reasonable wavelegth range
    cond1 = (wave > 3350)
    wave2 = np.extract(cond1, wave)

    cond2 = (wave2 < 10000)
    wave3 = np.extract(cond2, wave2)

    return wave3


def get_resolution(opt):

    grating_param = grating_params(opt)
    #print "grating_param: %s" % grating_param;
    resolution = float(opt['cenwave']) / grating_param[0]*float(opt['slit_width'])
    return resolution

def calculate_sky(opt):

    [lam,flux] = read_skymodels(opt)
    grating_param = grating_params(opt)



    #Calculate the effective resolution (in Ang) at the central wavelength
    resolution = get_resolution(opt)
    skyspec= gauss_convol(lam, flux, resolution)

    #The current units on skyspec are phot/s/m^2/micron/arcsecond
    mmt_area = math.pi*( (6.5/2.0)**2 - (1.0/2.0)**2) #Square meter
    skyspec = skyspec * mmt_area # Photons/s/micron/"^2
    skyspec = 1e-4*skyspec # photons/s/Ang/"2

    #Correct for telescope efficiency
    #Here, we estimate the efficicy based on the count rate (since the count rate is of a known source
    #rate, this is allowed.)
    if opt['spectrograph'] == 'blue' :
        file = 'data/throughput/bluechannel_300_dang_counts.dat'
    if opt['spectrograph'] == 'red' :
        file = 'data/throughput/redchannel_throughput.dat'
    input = np.genfromtxt(file, comments='#', unpack=True)
    e_lam = input[0]
    e_frac1 = input[1]*(0.002658*e_lam)  #This second term is the expected count rate for a 1uJY source
                                                                            #So ths correction factor converts the throughput to an efficiency
    e_frac = interpol(lam, e_lam, e_frac1)

    skyspec = e_frac*skyspec #Observed phot/s/Ang/"2

    #Bin to final wavelength grid
    lam_out = get_lam(opt)
    skyspec_out = interpol(lam_out, lam, skyspec) * grating_param[1] * float(opt['binning_spectral']) # Photons/s/pixel (instead of Angstrom)


    return [lam_out, skyspec_out]

def get_pixel_scale():
    #For the given spectrograph, return the pixel scale

    #For now, return the same for both as they are approximately the same
    return 0.3

def get_gain(opt):

    if opt['spectrograph'] == 'blue' : return 1.1
    if opt['spectrograph'] == 'red' : return 1.3

def get_readnoise(opt):
    if opt['spectrograph'] == 'blue' : return 2.45
    if opt['spectrograph'] == 'red' : return 3.50

def slit_transmission(opt):

    trans = scipy.special.erf(float(opt['slit_width']) / (2*np.sqrt(2.)*float(opt['seeing'])/2.35482))
    return trans

def source_transmission(opt):

    lam = get_lam(opt)

    #For a source of constant magnitude
    if opt['spectype'] == 'fnu_flat':
        flux = np.arange(0, len(lam))*0.0 + float(opt['specmag'])
        flux = 10.0**((23.8-flux)/2.5) # This is the flux in micrjankies


    #Do the airmass correction
    input = np.genfromtxt('data/throughput/kpnoextinct.dat', comments='#', unpack=True)
    airmass_lam = input[0]
    airmass_extinct = input[1]
    airmass_interp = interpol(lam, airmass_lam, airmass_extinct)
    airmass_flux = 10.0**(-0.4*airmass_interp*float(opt['airmass']))
    flux = flux * airmass_flux


    #Correct for system efficiency
    if opt['spectrograph'] == 'blue' :
        file = 'data/throughput/bluechannel_300_dang_counts.dat'
    if opt['spectrograph'] == 'red' :
        file = 'data/throughput/redchannel_throughput.dat'
    input = np.genfromtxt(file, comments='#', unpack=True)
    e_lam = input[0]
    e_flux = input[1]
    eff_corr = interpol(lam, e_lam, e_flux)
    flux = flux * eff_corr*get_gain(opt) # photons (because of gain correction) / s / Angstrom


    #Correct for slit losses
    flux = flux * slit_transmission(opt)


    #Convolve to the proper resolution
    grating_param = grating_params(opt)
    resolution = get_resolution(opt)

    flux = gauss_convol(lam, flux, resolution) * grating_param[1] * float(opt['binning_spectral']) # Convolved spectrum in photons / s / pixel
    return [lam, flux]



def main(argv):
    # DP - 09/27/2016
    # Write out the arguments to a log file for debugging and general usage curiosity.
    log_file = open( "/var/www/html/expcalc/python/log.txt", "a" )
    if( log_file != None ):
        log_file.write( "Exposure Time Calculator submission\n------------------------------\n" )
        log_file.write( "{}: {}\n".format( str( datetime.datetime.now() ), argv ) )
    
    data =  {}

    #Initialize the options block and parse into easy to use format
    opt = json.loads(argv[1])
    if opt['calcType'] == 'Time':
        opt['calc_time'] = '1'
        opt['calc_snr'] = 0
        opt['snr_goal'] = opt['depth']

    if opt['calcType'] == 'SNR' :
        opt['calc_time'] = 0
        opt['calc_snr'] = '1'
        opt['time_goal'] = opt['depth']
    data['calcType'] = opt['calcType']

#set some defaults
    opt['spectype'] = 'fnu_flat'



    #Get the sky flux measurement
    [lam, skyflux] = calculate_sky(opt)

    opt['snr_minwave'] = opt['cenwave'] - 100
    opt['snr_maxwave'] = opt['cenwave'] + 100


    #Calculate the optimal extraction window
    npix = math.ceil(2*float(opt['seeing']) / (float(opt['binning_spatial'])*get_pixel_scale()))
    area_sky = npix * float(opt['slit_width']) * get_pixel_scale() * float(opt['binning_spectral'])
    skyflux = skyflux * area_sky

    #Get the object flux measurement
    [objlam, objflux] = source_transmission(opt)

    #Get the readnoise
    readnoise = get_readnoise(opt)

    #Get the correction due to the filter
    filterCorrection = getFilterCorrection(opt, lam)
    objflux = objflux * filterCorrection


    #Get the blaze function correction
    gratingCorrection = getBlazeCorrection(opt, lam)
    objflux = objflux * gratingCorrection




    #Now, Check to see if we were asked to calculate the signal to noise in a given time or the
    #time to reach a given exposure time and go for it.

    #Now, if you were asked to calculate the signal to noise
    if opt['calc_snr'] == '1':
        exptime = float(opt['time_goal'])

        snr = objflux * exptime / (objflux*exptime + skyflux*npix*exptime + readnoise**2*npix)**0.5
        min_snr = np.amin(snr)
        max_snr = np.amax(snr)

        min_lam = np.amin(lam)
        max_lam = np.amax(lam)


        #Now see if they have set a window
        if opt['snr_minwave'] != 'INDEF':
            minwave = float(opt['snr_minwave'])
        else:
            minwave = np.amin(lam)
        if opt['snr_maxwave'] != 'INDEF' :
            maxwave = float(opt['snr_maxwave'])
        else :
            maxwave = np.amax(lam)

        #Find the pixels that meet this
        snr_subarray = []
        lam_subarray = []
        exp_time_goal = []
        counter = 0

        data['minwave'] = minwave
        data['maxwave'] = maxwave

        for ii in range(0, len(lam)):
            if (lam[ii] > minwave) and (lam[ii] < maxwave) and (snr[ii] > 0):
                snr_subarray.append(snr[ii])
                lam_subarray.append(lam[ii])
                exp_time_goal.append(exptime)
                counter += 1

        data['snr_sub'] = zip(lam_subarray, snr_subarray);
        data['exp_sub'] = zip(lam_subarray, exp_time_goal);
        if counter > 1:
            snr_median = np.median(snr_subarray)
            data['snr_median'] = snr_median
            data['exptime_median'] = exptime
        exptime_median = exptime




    #If were were asked to calculate the time
    if opt['calc_time'] == '1':
        snr_goal = float(opt['snr_goal'])
        #Calculate the SNR versus lambda
        exptime = ( snr_goal**2 * (objflux + npix*skyflux) +
                    (snr_goal**4*(objflux+npix*skyflux)**2 +
                     4.0*objflux**2*snr_goal**2*npix*readnoise**2)**0.5) / (2*objflux**2)

        data['exptime'] = zip(lam.tolist(), exptime.tolist())

        min_time = np.amin(exptime)
        max_time = np.amax(exptime)

        min_lam = np.amin(lam)
        max_lam = np.amax(lam)


        #Now see if they set a window for where they want this calculation
        if opt['snr_minwave'] != 'INDEF':
            minwave =float(opt['snr_minwave'])
        else :
            minwave = np.amin(lam)
        if opt['snr_maxwave'] != 'INDEF' :
            maxwave = float(opt['snr_maxwave'])
        else:
            maxwave = np.amax(lam)

        #Identify pixels that meat that critera
        exptime_subarray = []
        lam_subarray = []
        snr_goal_sub = []
        counter = 0

        data['minwave'] = minwave
        data['maxwave'] = maxwave

        for ii in range(0, len(lam)):
            if (lam[ii] > minwave) and (lam[ii] < maxwave) and (exptime[ii] > 0):
                exptime_subarray.append(exptime[ii])
                lam_subarray.append(lam[ii])
                snr_goal_sub.append(snr_goal)
                counter +=1

        data['exp_sub'] = zip(lam_subarray,exptime_subarray)
        data['snr_sub'] = zip(lam_subarray, snr_goal_sub);
        if counter > 1:
            exptime_median = np.median(exptime_subarray)
            data['exptime_median'] = exptime_median
            snr = objflux * exptime_median / (objflux*exptime_median
                                              + skyflux*npix*exptime_median + readnoise**2*npix)**0.5
            data['snr_median'] = np.median(snr)

    #Make the spectral plots
    objcounts = objflux * exptime_median
    skycounts = skyflux * npix * exptime_median
    totalcounts = objcounts + skycounts

    data['lam'] = lam.tolist()
    data['skycounts'] = zip(lam.tolist(), skycounts.tolist())
    data['objcounts'] = zip(lam.tolist(), objcounts.tolist())
    data['totalcounts'] = zip(lam.tolist(), totalcounts.tolist())
    data['snr'] = zip(lam.tolist(), snr.tolist())
    
    # DP - close the log file.
    if( log_file != None ):
        log_file.close()
    
    print json.dumps(data)

if __name__=="__main__":
    main(sys.argv)
