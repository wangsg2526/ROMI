import numpy as np
import glob
import datetime
import re
import scipy.io
import netCDF4
import sys
from numba import jit


# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
@jit(nogil=True)
def runningMeanFast_conv(x, N):
    N2 = np.int_((N-1)/2)
    out = np.convolve(x, np.ones((N,))/N,mode='valid') 
    padbeg = np.zeros((N2,))
    padend = np.zeros((N2,))

    padbeg[0] = x[0]
    for i in np.arange(1,N2):
        padbeg[i] = x[0:2*i+1].mean()

    xrev = x[::-1]
    padend[0] = xrev[0]
    for i in np.arange(1,N2):
        padend[i] = xrev[:2*i+1].mean()
    padend = padend[::-1]

    out = np.concatenate(  ( padbeg, out, padend )  )

    return out
# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
@jit(nogil=True)
def runningMeanFast_conv_wrong(x, N):
    out = np.convolve(x, np.ones((N,))/N,mode='valid')[(N-1):]
    padbeg = np.zeros((N+1,))
    padend = np.zeros((N-3,))

    padbeg[0] = x[0]
    for i in np.arange(1,N+1):
        padbeg[i] = x[i-i:i+i].mean()

    xrev = x[::-1]
    padend[0] = xrev[0]
    for i in np.arange(1,N-3):
        padend[i] = xrev[i-i:i+i].mean()
    padend = padend[::-1]

    out = np.concatenate(  ( padbeg, out, padend )  )

    return out

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
def get_omi_eofs():
   mjo_mode = scipy.io.loadmat('./Kiladis-iso-index.mat')
   #mjo_mode = scipy.io.loadmat('/sw19/wangs/s2s_post/Kiladis-Index/Kiladis-iso-index.mat')
   nd, nlon, nlat = mjo_mode['eof1'].shape
   eof1_leap = np.zeros((366,nlon, nlat))
   eof1_leap[:60,:,:] = mjo_mode['eof1'][:60,:,:]
   eof1_leap[61:,:,:] = mjo_mode['eof1'][60:,:,:]
   eof1_leap[60,:,:] = (mjo_mode['eof1'][59,:,:] + mjo_mode['eof1'][60,:,:])*0.5

   eof2_leap = np.zeros((366,nlon, nlat))
   eof2_leap[:60,:,:] = mjo_mode['eof2'][:60,:,:]
   eof2_leap[61:,:,:] = mjo_mode['eof2'][60:,:,:]
   eof2_leap[60,:,:] = (mjo_mode['eof2'][59,:,:] + mjo_mode['eof2'][60,:,:])*0.5

   mjo_mode.update({'eof1_leap':eof1_leap, 'eof2_leap':eof2_leap})
   return mjo_mode

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
def get_olr4omi(olrDataFile):
    #specify NOAA interpolated OLR first from https://www.esrl.noaa.gov/psd/data/gridded/data.interp_OLR.html
    # This could read from noaa through opendap as:
    # ff = 'https://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/interp_OLR/olr.day.mean.nc'
    # The file is ~ 300 M. So this is slow. Instead we download it and save as interp_OLR.day.mean.nc
    # ff = '/sw19/wangs/s2s_post/noaa_olr/interp_OLR.day.mean.nc'
    fout = netCDF4.Dataset(olrDataFile,'r',mmap=False)
    lats = fout.variables['lat'][:].squeeze()
    lons = fout.variables['lon'][:].squeeze()
    time = fout.variables['time'][:].squeeze()
    olr = fout.variables['olr'][:].squeeze()
    fout.close()

    ilat_sel = (lats>=-20) & (lats<=20)
    lats = lats[ilat_sel]
    olr = olr[:,ilat_sel,:]
    ttime = [datetime.datetime(1800,1,1)+ datetime.timedelta(tt/24.0) for tt in time]
    tord = np.array([tt.toordinal() for tt in ttime])
    tsel = np.where( (tord >= datetime.datetime(1979,1,1).toordinal()) &  (tord <= datetime.datetime(2016, 12,31).toordinal()) ) [0]
    olr = olr[tsel,:,:]
    tord = tord[tsel]
    ttime = [ttime[it] for it in tsel]
    print(olr.shape, tord.shape, len(ttime))
    nt, nlat, nlon = olr.shape

    olr_clim = np.zeros((365, nlat, nlon))
    olr_clim_leap = np.zeros((366, nlat, nlon))
    ic = 0
    for iy in np.arange(1979, 2017):
            cc = np.where( (tord >= datetime.datetime(iy,1,1).toordinal()) &  (tord <= datetime.datetime(iy,12,31).toordinal()) ) [0]
            cc1 = tord[cc] - datetime.datetime(iy,1,1).toordinal()
            if np.mod(iy,4) != 0 :
                olr_clim = olr_clim + olr[cc,:,:]
                ic += 1
            else :
                ci = np.where(tord[cc] != datetime.datetime(iy,2,29).toordinal())[:]
                cc = cc[ci]
                olr_clim = olr_clim + olr[cc,:,:]
                ic += 1
    olr_clim = olr_clim/ic
    olr_clim.shape

    if 1 == 1:
        olr_clim_filt  = np.zeros(olr_clim.shape)
        ii = 75; jj = 13
        for ii in np.arange(nlon):
            for jj in np.arange(nlat):
                otmp = olr_clim[:,jj,ii]
                fft_coef = np.fft.fft(otmp)
                fft_coef[4:365-3] = 0.0 # remove mean and first 3 harmonic components
                otmp = np.real(np.fft.ifft(fft_coef))
                olr_clim_filt[:,jj,ii] = otmp
        olr_clim = olr_clim_filt

    olr_clim_leap[:60,:,:] = olr_clim[:60,:,:]
    olr_clim_leap[61:,:,:] = olr_clim[60:,:,:]
    olr_clim_leap[60,:,:] = (olr_clim[59,:,:] + olr_clim[60,:,:])*0.5

    olr_noac = np.zeros(olr.shape)
    for iy in np.arange(1979, 2017):
            cc = np.where( (tord >= datetime.datetime(iy,1,1).toordinal()) &  (tord <= datetime.datetime(iy,12,31).toordinal()) ) [0]
            cc1 = tord[cc] - datetime.datetime(iy,1,1).toordinal()
            if np.mod(iy,4) != 0 :
                olr_noac[cc,:,:] = olr[cc,:,:] - olr_clim[cc1,:,:]
            else:
                olr_noac[cc,:,:] = olr[cc,:,:] - olr_clim_leap[cc1,:,:]

    olr_rt_noac = np.zeros(olr.shape)
    for iy in np.arange(1979, 2017):
            cc = np.where( (tord >= datetime.datetime(iy,1,1).toordinal()) &  (tord <= datetime.datetime(iy,12,31).toordinal()) ) [0]
            cc1 = tord[cc] - datetime.datetime(iy,1,1).toordinal()
            if np.mod(iy,4) != 0 :
                olr_rt_noac[cc,:,:] = olr[cc,:,:] - olr_clim
            else: 
                olr_rt_noac[cc,:,:] = olr[cc,:,:] - olr_clim_leap            
    olr_rt_noac_sm = np.zeros(olr_rt_noac.shape)
    for it in np.arange(40, olr_rt_noac.shape[0]):
        olr_rt_noac_sm[it,:,:] = olr_rt_noac[it,:,:] - olr_rt_noac[it-40:it,:,:].mean(0)

    olr_rt_noac_sm_tapavg = np.zeros(olr_rt_noac_sm.shape)
    if 1 == 1:
        for ii in np.arange(olr_rt_noac_sm_tapavg.shape[1]):
          for jj in np.arange(olr_rt_noac_sm_tapavg.shape[2]):
            olr_rt_noac_sm_tapavg[:, ii, jj]  = runningMeanFast_conv(olr_rt_noac_sm[:, ii, jj], 9)

    return olr_rt_noac, olr_rt_noac_sm, olr_rt_noac_sm_tapavg, tord, ttime, lons, lats

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
def get_olr_clim():
    """ compute observed OLR climatology """
    ff = '/sw19/wangs/s2s_post/noaa_olr/interp_OLR.day.mean.nc'
    fout = netCDF4.Dataset(ff,'r',mmap=False)
    lats = fout.variables['lat'][:].squeeze()
    lons = fout.variables['lon'][:].squeeze()
    time = fout.variables['time'][:].squeeze()
    olr = fout.variables['olr'][:].squeeze()
    fout.close()

    ilat_sel = (lats>=-20) & (lats<=20)
    lats = lats[ilat_sel]
    olr = olr[:,ilat_sel,:]
    ttime = [datetime.datetime(1800,1,1)+ datetime.timedelta(int(tt/24.0)) for tt in time]
    tord = np.array([tt.toordinal() for tt in ttime])
    tsel = np.where( (tord >= datetime.datetime(1979,1,1).toordinal()) &  (tord <= datetime.datetime(2016, 12,31).toordinal()) ) [0]
    olr = olr[tsel,:,:]
    tord = tord[tsel]
    ttime = [ttime[it] for it in tsel]
    print(olr.shape, tord.shape, len(ttime))
    nt, nlat, nlon = olr.shape

    olr_clim = np.zeros((365, nlat, nlon))
    olr_clim_leap = np.zeros((366, nlat, nlon))
    ic = 0
    for iy in np.arange(1979, 2017):
            cc = np.where( (tord >= datetime.datetime(iy,1,1).toordinal()) &  (tord <= datetime.datetime(iy,12,31).toordinal()) ) [0]
            cc1 = tord[cc] - datetime.datetime(iy,1,1).toordinal()
            if np.mod(iy,4) != 0 :
                olr_clim = olr_clim + olr[cc,:,:]
                ic += 1
            else:
                ci = np.where(tord[cc] != datetime.datetime(iy,2,29).toordinal())[:]
                cc = cc[ci]
                olr_clim = olr_clim + olr[cc,:,:]
                ic += 1
    olr_clim = olr_clim/ic
    olr_clim.shape
    
    if 1 == 1:
        olr_clim_filt  = np.zeros(olr_clim.shape)
        ii = 75; jj = 13
        for ii in np.arange(nlon):
            for jj in np.arange(nlat):
                otmp = olr_clim[:,jj,ii]
                fft_coef = np.fft.fft(otmp)
                fft_coef[4:365-3] = 0.0 # remove mean and first 3 harmonic components
                otmp = np.real(np.fft.ifft(fft_coef))
                olr_clim_filt[:,jj,ii] = otmp
        olr_clim = olr_clim_filt

    olr_clim_leap[:60,:,:] = olr_clim[:60,:,:]
    olr_clim_leap[61:,:,:] = olr_clim[60:,:,:]
    olr_clim_leap[60,:,:] = (olr_clim[59,:,:] + olr_clim[60,:,:])*0.5

    
    return olr_clim, olr_clim_leap,  tord, ttime, lons, lats
# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
def combine_ttr_fcst_obs(ttr, stime, olr_noac, ttime):
    """ combine_ttr_fcst_obs: ttr and olr_noac are in the same grid, but different latidue orientation 
     ttr: forecast: olr starting from 20S to 20N (-20, -17,5, -15, ..., 20)
     olr_noac: observed ttr in previous days: olr from 20N to 20S (20, 17.5, ..., -20)
     output: olr anomaly 20S -> 20N (-20, -17,5, -15, ..., 20)
     """
    #smat = scipy.io.loadmat('/sw19/wangs/s2s_post/olr_20S20N/em_ana/_ecmf_em/ECMF_ref_ttr_em_20171120-20111120.mat')
    #stime = [datetime.datetime.fromordinal(smat['time'][0,i]/24 + datetime.datetime(1900,1,1).toordinal()) \
    #          for i in np.arange(smat['time'].shape[1])]
    stime_ord = [ss.toordinal() for ss in stime]
    #it0 = ttime.index(stime[0])
    it0 = ttime.index(stime[0]-datetime.timedelta(1)) + 1
    
    olr_comb = np.concatenate((olr_noac[it0-140:it0, ::-1, :], ttr), axis=0)

    olr_comb_sm = np.zeros(olr_comb.shape)
    for it in np.arange(0, olr_comb.shape[0]):
        olr_comb_sm[it,:,:] = olr_comb[it,:,:] - olr_comb[it-40:it,:,:].mean(0)
    if 1 == 1:
        for ii in np.arange(olr_comb_sm.shape[1]):
            for jj in np.arange(olr_comb_sm.shape[2]):
                olr_comb_sm[:,ii,jj] = runningMeanFast_conv(olr_comb_sm[:,ii,jj], N=9)
        
    olr_comb_sm = olr_comb_sm[40:,...] 
    tout = [ttime[it0-i] for i in np.arange(100,0, -1)]
    #print(len(stime), len(tout))
    tout = tout +  stime
    
    return olr_comb_sm, tout

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
def calc_mjo_ind(olr_in, tin, mjo_mode):
    """ 
     olr_in: olr anomaly starting from 20N to 20S (20, 17.5, ..., -20)
     mjo_mode: EOFs from 20N to 20S (20, 17.5, ..., -20)
    """
    mjo_rt = np.zeros((olr_in.shape[0], 2))    
    for it in np.arange(olr_in.shape[0]):
        a1 = olr_in[it,:,:]
        iday = tin[it].timetuple().tm_yday - 1
        
        if (tin[it].year % 4 == 0):
            a2 = mjo_mode['eof1_leap'][iday,:,:].T      
            mjo_rt[it, 0] = np.sum(a1*a2 )

            a2 = mjo_mode['eof2_leap'][iday,:,:].T      
            mjo_rt[it, 1] = np.sum(a1*a2 )
        else:
            a2 = mjo_mode['eof1'][iday,:,:].T      
            mjo_rt[it, 0] = np.sum(a1*a2 )

            a2 = mjo_mode['eof2'][iday, :,:].T      
            mjo_rt[it, 1] = np.sum(a1*a2 )
    return mjo_rt

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
def calc_romi(olr_rt_noac_sm_tapavg, ttime, mjo_mode):
    """ 
     olr_rt_noac_sm_tapavg: 9-pt averaged from obs, starting from 20N to 20S (20, 17.5, ..., -20) 
     mjo_mode: EOFs from 20S to 20N (-20, -17.5, ..., 20)
    """ 
    mjo_rt2 = np.zeros((olr_rt_noac_sm_tapavg.shape[0], 2))    
    for it in np.arange(olr_rt_noac_sm_tapavg.shape[0]):
        iday = ttime[it].timetuple().tm_yday - 1
        if (ttime[it].year % 4 == 0):
            a1 = olr_rt_noac_sm_tapavg[it,::-1,:]
        
            a2 = mjo_mode['eof1_leap'][iday,:,:].T      
            mjo_rt2[it, 0] = np.sum(a1*a2 )
        
            a2 = mjo_mode['eof2_leap'][iday,:,:].T      
            mjo_rt2[it, 1] = np.sum(a1*a2 )
        else:
            a1 = olr_rt_noac_sm_tapavg[it,::-1,:]
        
            a2 = mjo_mode['eof1'][iday,:,:].T      
            mjo_rt2[it, 0] = np.sum(a1*a2 )
        
            a2 = mjo_mode['eof2'][iday, :,:].T      
            mjo_rt2[it, 1] = np.sum(a1*a2 )
    return mjo_rt2
# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
