import argparse
import os
import sys
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import scipy.optimize as opt
import urllib.request
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel
from PIL import Image
from scipy.stats import norm

class Plot(): 
    """
    Plot the spectrum of a galaxy who's data is stored in a FITS file. The name of the file should be the galaxy's AGC number.
    The script has to be in the same directory (as of now) as the FITS files.
    
    Parameters
    
    ----------
    
    filename : int
        AGC number of the galaxy, e.g, 104365s
    
    xmin : float
        Minumum x value on the plot
    
    xmax : float
        Maximum x value on the plot
    
    ymin : float
        Minumum y value on the plot
    
    ymax : float
        Maximum y value on the plot
        
    showimage : bool
        When true, the program will obtain the SDSS DR14 inverted image of the galaxy. Default is true.
    
    saveplot : bool
        When true, the program will save the plot as png image with the AGC number as the name of the file in the same directory the program is being run from.
        Default is false.
    
    smo : str
       Value for smoothing the spectrum, if nothing is passed smoothing will not occur; 'h' for hanning smoothing, 'bX' for boxcar where X is a postive integer.
        e.g, 'smo' = 'b7'
        
    h : bool
        When true, prints the doc string of this class. Default is false.
    
    """

    def __init__(self,filename, xmin= None, xmax= None, ymin= None, ymax= None, showimage= True, ned = None, saveplot = False, smo= None, h = False):
        
        self.filename ='A{:06}.fits'.format(filename) 
        self.run = False #boolean to check whether or not to run smoothing operation. If true, yes.
        
        if h:
            print(self.__doc__)
        
        if smo is not None:
            self.run = True
            if smo == 'h':
                self.smotype = 'h'
            else : self.smotype = smo[1:]  #Just sends the value for boxcar smoothing
    
        
        if showimage:
            self.displayimage(self.getheader(self.readdata()[1]))
        
        if ned is not None:
            self.open_ned(ned)
               
        self.plot(self.readdata()[0], xmin, xmax, ymin, ymax, saveplot)
        
    
    # Filename can be accessed by calling this method.
    def name(self):
        return self.filename
    # Smoothtype can be accessed by calling this method
    def smootype(self):
        return self.smotype
    # Boolean value whether to smooth the plot or not
    def runsmo(self):
        return self.run
    
    def displayimage(self,hdr):
        hdr = self.getheader(self.readdata()[1])
        url = 'http://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={ra}&dec={dec}&scale=0.2&width=500&height=500&opt=I'.format(ra = hdr[18], dec = hdr[19])
        imgfilename = 'grab.jpg'
        urllib.request.urlretrieve(url, imgfilename)
        
        img = Image.open('grab.jpg')
        img.show() #does not work in a jupyter notebook
        os.remove('grab.jpg')
        del img
        
    def open_ned(self, ned):
        hdr = self.getheader(self.readdata()[1])
        url = "http://ned.ipac.caltech.edu/cgi-bin/objsearch?in_csys=Equatorial&in_equinox=J2000.0&lon={ra}d&lat={dec}d&radius={arc}&hconst=73&omegam=0.27&omegav=0.73&corr_z=1&search_type=Near+Position+Search&z_constraint=Unconstrained&z_value1=&z_value2=&z_unit=z&ot_include=ANY&nmp_op=ANY&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=Distance+to+search+center&of=pre_text&zv_breaker=30000.0&list_limit=5&img_stamp=YES".format(
            ra=hdr[18], dec=hdr[19], arc=ned)
        webbrowser.open(url, new=0, autoraise=True)
    
    # Read in the data.
    # The 1 is because SDFits uses an extension on the normal data (which is in [0] and empty)
    def readdata(self):
        hdul = fits.open(self.name())
        fitsdata = hdul[1].data
        return fitsdata, hdul #returns a tuple where the index=0 contains the fitsdata and the index=1 contains the hdul
    
    #the method will return a tuple which will contain each numpy array
    def fillarrays(self,fitsdata):
        
        entries = len(fitsdata)      

        vel  = np.zeros(entries)
        freq = np.zeros(entries)
        spec = np.zeros(entries)
        base = np.zeros(entries)
        weight = np.zeros(entries)

        # Fill the arrays
        for i in range(len(fitsdata)):
            vel[i] = fitsdata[i][0]
            freq[i] = fitsdata[i][1]
            spec[i] = fitsdata[i][2]
            base[i] = fitsdata[i][3]
            weight[i] = fitsdata[i][4]
            
        return vel, freq, spec, base, weight  
    
    def smooth(self):
        
        data = self.fillarrays(self.readdata()[0])
        vel = data[0]
        freq = data[1]
        spec = data[2]
        base = data[3]
        weight = data[4]
        
        run = self.runsmo()
        
        if run is True: # Hanning Smoothing
            if (self.smootype() == 'h'):
                smoothed_signal = np.convolve(spec, [0.25,0.5,0.25], mode='same')
            else: #Boxcar Smoothing
                mag = int(self.smootype())
                box_kernel = Box1DKernel(mag)
                smoothed_signal = convolve(spec, box_kernel)
        return smoothed_signal
        
    def plot(self, fitsdata, xmin, xmax, ymin, ymax, saveplot): 
        
        data = self.fillarrays(self.readdata()[0]) 
        vel = data[0]
        freq = data[1]
        spec = data[2]
        base = data[3]
        weight = data[4]
        
        signal = spec
        run = self.runsmo()
        
        if run is True:
            signal = self.smooth()
        
        # Make a plot and show it
        
        fig, ax = plt.subplots()
        ax.plot(vel, signal, color = 'black', linewidth=1)
        ax.axhline(y=0, color = 'black',  dashes=[5,5])
        ax.set(xlabel="Velocity (km/s)",ylabel= "Flux (mJy)", title = 'AGC {}'.format(self.name()[1:-5]))
        ax.set(xlim=(xmin,xmax), ylim=(ymin,ymax))
        ax.grid(False)
        if saveplot is True:
            fig.savefig('{}_plot.png'.format(self.name()[:-5]))
        plt.show(block = True)
        # plt.pause(3)
        plt.close()
        
    # Returns  the FITS header  
    def getheader(self,hdul):
        hdr = hdul[1].header
        return hdr   # returning hdr[14] returns just the AGC number but does it like 'A 2532' instead of 'A002532'
    def __str__(self):
        return str(self.getheader(self.readdata()[1]))
    
    def __repr__(self):
        return str(self)

