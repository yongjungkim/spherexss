# SPHEREx Simple Simulator
# by Yongjung Kim
'''
Import Modules
'''
import numpy as np
from astropy.table import Table
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt
import pickle,sys
from scipy.interpolate import interp1d
# For smoothing?
# import pysynphot as S
from astropy.convolution import convolve_fft,Gaussian1DKernel
from scipy.integrate import simpson,trapezoid
from scipy.special import erfc

'''
Useful Functions
'''

def fnl_convert(w_in,f_in):
    '''
    Convert units of f_nu, f_lambda
    f_nu [Jy] or [erg/s/cm2/Hz] <-> f_lambda [erg/s/cm2/AA]

    Input

        w_in: wave input [angtrom]

        f_in: f_nu or f_lambda

    '''

    if f_in.unit == u.Jy: # if f_nu [Jy]
        f_out = f_in.to(u.erg/u.s/u.cm/u.cm/u.Hz) / w_in / w_in * const.c.to(u.cm/u.s)
    elif f_in.unit == u.erg /u.s/u.cm/u.cm/u.Hz: # if f_nu [erg/s/cm2/Hz]
        f_out = f_in / w_in/ w_in * const.c.to(u.cm/u.s)
    elif f_in.unit == u.erg /u.s/u.cm/u.cm/u.AA: # if f_lambda [erg/s/cm2/AA]
        f_out = f_in * w_in * w_in / const.c.to(u.cm/u.s)
        f_out = f_out.to(u.Jy)
    else:
        print('Inappropriate unit:%s'%f_in.unit)
        sys.exit()
    
    return f_out


'''
Basic Class?
'''
class sxss(object):
    


    '''
    Initial setting

    ex) sp = sxss(wave,flux)
    '''
    def __init__(self,wave,flux):
        '''
        Input

            wave: wavelength (should be given with astropy units)
            
            flux: f_nu or f_lambda (should be given with astropy units: Jy, erg/s/cm2/Hz, erg/s/cm2/AA)

        '''

        # Set wavelength
        self.wave = wave.to(u.Angstrom)
        
        # Set flux
        # f_nu [Jy], f_lambda [erg/s/cm2/AA], mAB [/]
        if (flux.unit == u.Jy) | (flux.unit == u.erg /u.s/u.cm/u.cm/u.Hz): # if f_nu [Jy] or [erg/s/cm2/Hz]
            self.f_nu = flux
            self.f_lambda = fnl_convert(self.wave,self.f_nu) 
        elif flux.unit == u.erg /u.s/u.cm/u.cm/u.AA: # if f_lambda [erg/s/cm2/AA]
            self.f_lambda = flux
            self.f_nu = fnl_convert(self.wave,self.f_lambda)
        
        # Set AB magnitude
        self.mAB = -2.5*np.log10(self.f_nu.value)+8.90

        # SPHEREx resolution
        self.spherex_resolution = [
            # [wave1, wave2, resolution]
            [7500*u.AA,  24200*u.AA, 41 ],
            [24200*u.AA, 38200*u.AA, 35 ],
            [38200*u.AA, 44200*u.AA, 110],
            [44200*u.AA, 50000*u.AA, 130]
        ]
         
        # SPHEREx sensitivity & wavelength info (CBE)
        tbl_sens = Table.read('Point_Source_Sensitivity_v28_base_cbe_detector.txt',format='ascii',names=['wave','wide','deep','detector'])
        self.sens_wave = np.array(tbl_sens['wave']*1e4)*u.Angstrom
        self.sens_mAB_wide = np.array(tbl_sens['wide'])
        self.sens_mAB_deep = np.array(tbl_sens['deep'])
        self.sens_f_nu_wide = 10 ** ( (8.90 - self.sens_mAB_wide)/2.5)
        self.sens_f_nu_deep = 10 ** ( (8.90 - self.sens_mAB_deep)/2.5)
        self.sens_detector = np.array(tbl_sens['detector'])

        # SPHEREx resolution
        sres = []
        for wave1,wave2,res in self.spherex_resolution:
            slice = (self.sens_wave>=wave1) & (self.sens_wave<wave2)
            sres+= np.full(np.sum(slice),res).tolist()
        self.sens_sres = np.array(sres)

    def __getitem__(self, key):
        return self.position[key]
        
    '''
    Smoothing
    '''
    def smooth(self):

        # Smoothing according to resolution
        def smoothR(wave,f_nu,R,w_sample=1):
            '''

            This code is sourced from https://github.com/spacetelescope/pysynphot/issues/78)
            
            Parameters
            ==========
            wave: wavelength [angstrom]
            
            f_nu: flux, f_nu [Jy]
            
            R: Resolution (integer)

            w_sample: oversampling factor for smoothing (integer)

            Returns
            =======
            flux_new
            '''
            w_grid = wave.value

            # Generate logarithmic wavelength grid for smoothing
            w_logmin = np.log10(np.nanmin(w_grid))
            w_logmax = np.log10(np.nanmax(w_grid))
            n_w = np.size(wave)*w_sample
            w_log = np.logspace(w_logmin, w_logmax, num=n_w)

            # Find stddev of Gaussian kernel for smoothing
            R_grid = (w_log[1:-1]+w_log[0:-2])/(w_log[1:-1]-w_log[0:-2])/2
            sigma = np.median(R_grid)/R
            if sigma < 1:
                sigma = 1
            
            # Interpolate on logarithmic grid
            f_log = np.interp(w_log, w_grid, f_nu)

            # Smooth convolving with Gaussian kernel
            gauss = Gaussian1DKernel(stddev=sigma)
            f_conv = convolve_fft(f_log, gauss)

            # Interpolate back on original wavelength grid
            f_sm = np.interp(w_grid, w_log, f_conv)

            return f_sm

        wave_smooth = []
        f_nu_smooth = []
        f_lambda_smooth = []
        mAB_smooth = []

        for wave1,wave2,res in self.spherex_resolution:
            f_nu_tmp = smoothR(self.wave,self.f_nu,res)
            f_lambda_tmp = smoothR(self.wave,self.f_lambda,res)
            mAB_tmp = smoothR(self.wave,self.mAB,res)

            slice = (self.wave > wave1) & (self.wave < wave2)
            wave_smooth += list(self.wave[slice].value)
            f_nu_smooth += list(f_nu_tmp[slice].value)
            f_lambda_smooth += list(f_lambda_tmp[slice].value)
            mAB_smooth += list(mAB_tmp[slice])

        self.wave_smooth = np.array(wave_smooth)*self.wave.unit
        self.f_nu_smooth = np.array(f_nu_smooth)*self.f_nu.unit
        self.f_lambda_smooth = np.array(f_lambda_smooth)*self.f_lambda.unit
        self.mAB_smooth = np.array(mAB_smooth)

    '''
    [Filtering]
        Available filters: tophat
    '''
    def filtering(self, method='tophat', params=[]):
        
        # Tophat transmission curve
        def tophat_trans(x, center=0, fwhm=1, smoothness=0.2):
            t_left  = erfc(+((2*(x-center)/fwhm)-1)/smoothness)/2
            t_right = erfc(-((2*(x-center)/fwhm)+1)/smoothness)/2

            return (t_left*t_right)

        # Mean flux with transmission curves
        def mean_flux(wave, f_in, wave_trs, trs, tol=1e-3):
            """
            Calculate mean flux from input spectrum & transmission curves

            Parameters
            ==========

            wave: wavelength [angstrom]
            
            f_in: f_nu or f_lambda
            
            trs: transmission curve as a function of wave

            Returns
            =======
            flux_new


            Reference
            ---------
            Lecture note by Dr. Yujin Yang:

            index_filt, = np.where(resp_lvf > resp_lvf.max()*tol)

            index_flux, = np.where(np.logical_and( wave > wave_lvf[index_filt].min(), 
                                                    wave < wave_lvf[index_filt].max() ))

            wave_resamp = np.concatenate( (wave[index_flux], wave_lvf[index_filt]) )
            wave_resamp.sort()
            wave_resamp = np.unique(wave_resamp)
            flux_resamp = np.interp(wave_resamp, wave, flux)
            resp_resamp = np.interp(wave_resamp, wave_lvf, resp_lvf)

            return trapezoid(resp_resamp / wave_resamp * flux_resamp, wave_resamp) \
                    / trapezoid(resp_resamp / wave_resamp, wave_resamp)

            """

            sel_trs = trs > trs.max()*tol
            sel_flux = (wave > wave_trs[sel_trs].min()) & (wave < wave_trs[sel_trs].max())

            wave1 = np.concatenate((wave[sel_flux],wave_trs[sel_trs]))
            wave1.sort()
            wave1 = np.unique(wave1)
            trs1  = np.interp(wave1,wave_trs,trs)
            f_in1 = np.interp(wave1,wave,f_in)

            # wave1, trs1, f_in1 = wave[sel_trs], trs[sel_trs], f_in[sel_trs]
            # print(len(trs1))
            if f_in.unit == self.f_nu.unit:
                f1 = trapezoid(trs1/wave1*f_in1,wave1)
                f2 = trapezoid(trs1/wave1,wave1)
                f_out = f1/f2
            elif f_in.unit == self.f_lambda.unit:
                f1 = trapezoid(wave1*trs1*f_in1,wave1)
                f2 = trapezoid(wave1*trs1,wave1)
                f_out = f1/f2
            
            return f_out


        
        # box filter (just using average flux)
        if method == 'box':

            f_nu_tmp = []
            f_lambda_tmp = []
            mAB_tmp = []

            for w,r in zip(self.sens_wave,self.sens_sres):
                wave1, wave2 = w*(1-0.5/r), w*(1+0.5/r)
                slice = (self.wave_smooth>wave1) & (self.wave_smooth<wave2)
                # print(np.sum(slice))
                f_nu_tmp.append(np.average(self.f_nu_smooth[slice].value))
                f_lambda_tmp.append(np.average(self.f_lambda_smooth[slice].value))
                mAB_tmp.append(np.average(self.mAB_smooth[slice]))

            self.f_nu_filtered = np.array(f_nu_tmp)*self.f_nu_smooth.unit
            self.f_lambda_filtered = np.array(f_lambda_tmp)*self.f_lambda_smooth.unit
            self.mAB_filtered = np.array(mAB_tmp)

        # Tophat filter
        elif method == 'tophat':

            # Black arrays
            f_nu_tmp = []
            f_lambda_tmp = []
            # mAB_tmp = []
            
            for w,r in zip(self.sens_wave,self.sens_sres):
                wave_trs = np.arange(w.value*(1-1.5/r),w.value*(1+1.5/r),1) * u.AA
                trs = tophat_trans(wave_trs,center=w,fwhm=w/r)
                f_nu_tmp.append(mean_flux(self.wave_smooth,self.f_nu_smooth,wave_trs,trs).value)
                f_lambda_tmp.append(mean_flux(self.wave_smooth,self.f_lambda_smooth,wave_trs,trs).value)
                # mAB_tmp.append(mean_flux(self.wave_smooth,self.f_nu_smooth,trs))
            
            self.f_nu_filtered = np.array(f_nu_tmp)*self.f_nu_smooth.unit
            self.f_lambda_filtered = np.array(f_lambda_tmp)*self.f_lambda_smooth.unit
            self.mAB_filtered = -2.5*np.log10(self.f_nu_filtered.value)+8.90

        # Else....
        else:
            print("%s is inappropriate method"%method)

    '''
    [Photometeric error]
        Calculate photometric errors based on Ivezic+19 & Ivezic+22
    '''
    def obssimul(self,delta_depth=0.0,sig_sys=0.01):
        # Photometric errors
        sig_sys2 = sig_sys*sig_sys #[mag^2]
        gamma = 0.039
        x = np.power(10.0, 0.4*(self.mAB_filtered-(np.array(self.sens_mAB_wide)-delta_depth)))
        # x = np.power(10.0, 0.4*(self.mAB_filtered-(np.array(self.sens_mAB_deep)-delta_depth)))
        sig_rand2 = (0.04-gamma)*x + gamma*x*x #[mag^2]
        sig1 = np.sqrt(sig_sys2 + sig_rand2) #[mag]

        # Random noises
        # np.random.seed(12345) # FIXED HERE!!!
        N_sig1 = [np.random.normal(0,s1) for s1 in sig1]
        self.mAB_obssimul = self.mAB_filtered + N_sig1
        self.mAB_e_obssimul = sig1

        # Conversion: AB mag to F_nu
        #  Uncertainty: |sigma_m| = (2.5/ln10) * sigma_f/f --> sigma_f ~ |sigma_m|*f/(2.5/ln10)
        self.f_nu_obssimul = np.power(10.0,((8.90-self.mAB_obssimul)/2.5)) * self.f_nu_smooth.unit
        self.f_nu_e_obssimul = sig1 * self.f_nu_obssimul/(2.5/np.log(10)) 



    '''
    Plot procedures
    '''
    def plot_f_nu_qso(self,z,mag,magtype='i_AB'):
        # Plot
        plt.figure(figsize=(10,4),dpi=200)
        pltmodel = plt.plot(self.wave,self.f_nu,label='Model')
        pltmodel_smooth = plt.plot(self.wave_smooth,self.f_nu_smooth,label='Model (smoothed)')
        pltmodel_sys = plt.plot(self.sens_wave,self.f_nu_filtered,marker='o',ls='',markersize=4,label='Average flux')
        pltmodel_obs = plt.errorbar(self.sens_wave,self.f_nu_obssimul,yerr=self.f_nu_e_obssimul,marker='o',ls='',markersize=4,label='Obs')
        # pltsys = plt.plot(swave,mag_sys,marker='o',ls='',markersize=4,label='Model')
        # pltobs = plt.errorbar(swave,mag_obs,yerr=sig1,marker='o',ls='',markersize=4,label='Obs')
        pltdepth = plt.plot(self.sens_wave,self.sens_f_nu_wide,marker='^',ls='',markersize=4,label=r'$5\sigma$ (CBE)')

        # Plot setting
        # plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength'+r' ($\AA$)')
        plt.ylabel(r'$f_{\nu}$ (Jy)')
        # plt.ylabel('AB magnitude')
        # yvals = list(mag_obs)+list(sens['wide'])
        # plt.ylim([np.max(yvals)+1,np.min(yvals)-1])
        yvals = list(self.f_nu_smooth.value) + list(self.sens_f_nu_wide) + list(self.f_nu_obssimul.value)
        plt.ylim([np.min(yvals)*0.5,np.max(yvals)*2])
        plt.xlim([6000,55000])
        plt.title('z=%.2f, '%z + magtype+'=%.2f'%mag)

        # Plot the locations of emission lines
        emwave = [6565, 4862.68, 4346.42,
                18735.5, 12821.3, 10941,
                10830, 2802.95, 1906.87, 1546.55]
        emname = [r'H$\alpha$',r'H$\beta$',r'H$\gamma$',
                r'P$\alpha$',r'P$\beta$',r'P$\gamma$',
                'HeI', 'MgII', 'CIII]', 'CIV']
        for emw,emn in zip(emwave,emname):
            emw2 = emw*(1+z)
            if (emw2 > 7500) & (emw2 < 50000):
                plt.plot([emw2,emw2],[np.min(yvals)*0.5,np.max(yvals)*2],ls=':',color='gray')
                if emw==10830:
                    plt.text(emw2*0.92,np.min(yvals)*0.55,emn,color='gray')
                else: plt.text(emw2,np.min(yvals)*0.55,emn,color='gray')

        plt.legend()
        # plt.legend(handles=((pltmodel[0],pltsys[0]),pltobs,pltdepth[0]),labels=['Model','Obs',r'$5\sigma$ (CBE)'])
        plt.tight_layout()
        plt.show()
    
    def plot_f_nu_btsettl(self,filename):
        # Plot
        plt.figure(figsize=(10,4))
        pltmodel = plt.plot(self.wave,self.f_nu,label='Model')
        pltmodel_smooth = plt.plot(self.wave_smooth,self.f_nu_smooth,label='Model (smoothed)')
        pltmodel_sys = plt.plot(self.sens_wave,self.f_nu_filtered,marker='o',ls='',markersize=4,label='Average flux')
        pltmodel_obs = plt.errorbar(self.sens_wave,self.f_nu_obssimul,yerr=self.f_nu_e_obssimul,marker='o',ls='',markersize=4,label='Obs')
        # pltsys = plt.plot(swave,mag_sys,marker='o',ls='',markersize=4,label='Model')
        # pltobs = plt.errorbar(swave,mag_obs,yerr=sig1,marker='o',ls='',markersize=4,label='Obs')
        pltdepth = plt.plot(self.sens_wave,self.sens_f_nu_wide,marker='^',ls='',markersize=4,label=r'$5\sigma$ (CBE)')

        # Plot setting
        # plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength'+r' ($\AA$)')
        plt.ylabel(r'$f_{\nu}$ (Jy)')
        # plt.ylabel('AB magnitude')
        # yvals = list(mag_obs)+list(sens['wide'])
        # plt.ylim([np.max(yvals)+1,np.min(yvals)-1])
        yvals = list(self.f_nu_smooth.value) + list(self.sens_f_nu_wide) + list(self.f_nu_obssimul.value)
        plt.ylim([np.min(yvals)*0.5,np.max(yvals)*2])
        plt.xlim([6000,55000])
        plt.title(filename)

        plt.legend()
        # plt.legend(handles=((pltmodel[0],pltsys[0]),pltobs,pltdepth[0]),labels=['Model','Obs',r'$5\sigma$ (CBE)'])
        plt.tight_layout()
        plt.show()
    
