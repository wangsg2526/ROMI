#!/sw1/wangs/anaconda3.6_new/bin//python
###!/sw21/wangs/anaconda2/bin/python

import pandas as pd
import numpy as np
import sys, imp
from datetime import datetime
import omi_utils 

print(sys.version)
print(sys.executable)

imp.reload(omi_utils)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        olrfile = sys.argv[1]
    else:
        olrfile = './interp_OLR.day.mean.nc'

    print('NOAA OLR data here:', olrfile)
    print('-------------------------------------------------')
    print()

    olr_rt_noac, olr_rt_noac_sm, olr_rt_noac_sm_tapavg, tord, ttime, olons, olats = omi_utils.get_olr4omi(olrDataFile=olrfile)
    mjo_mode = omi_utils.get_omi_eofs()
    mjo_rt2 = omi_utils.calc_romi(olr_rt_noac_sm_tapavg, ttime, mjo_mode)
    
    ofac = np.std(mjo_rt2[:,0])
    print(ofac)
    momi_1 = mjo_rt2[:,0]/ofac
    momi_2 = mjo_rt2[:,1]/ofac
    pha = np.arctan2(-momi_1, momi_2);
    momi_time = ttime
    momi_rt_phase = np.floor((pha+np.pi)/(np.pi/4)).astype(int)
    momi_rt_phase += 1
    momi_rt_amp = np.sqrt(momi_1**2 + momi_2**2)
    momi_rt_df = pd.DataFrame({'omi1':momi_1, 'omi2':momi_2, 'rmma':momi_rt_amp, 'phase':momi_rt_phase, 'date':ttime, })
   
    # form the arrays and save it to a text file
 
    dyear = np.array([t.year for t in ttime])[:,np.newaxis]
    dmonth = np.array([t.month for t in ttime])[:,np.newaxis]
    dday = np.array([t.day for t in ttime])[:,np.newaxis]
    date_arr = np.concatenate((dyear, dmonth, dday), axis=1 )
    X = np.concatenate((date_arr, momi_1[:,None], momi_2[:,None], momi_rt_phase[:,None]), axis=1)
    np.savetxt('romi_1979-2016.txt', X, fmt='%4d   %2d   %2d %10.2f %10.2f   %1d')
    #np.savetxt('romi_1979-2016.txt', X, fmt='%4d %2d %2d %10.7f %10.7f %1d')
    # ------------------------------------------------------------------------------
    
    i1 = momi_time.index(datetime(2002,1,1))
    
    romi_dat = np.loadtxt('./romi.1x.txt')
    romi_time = [datetime(int(romi_dat[i,0]), int(romi_dat[i,1]), int(romi_dat[i,2])) for i in np.arange(romi_dat.shape[0]) ]
    romi_tord = np.array([tt.toordinal() for tt in romi_time])
    romi_1 = romi_dat[:,4]
    romi_2 = romi_dat[:,5]
    pha = np.arctan2(-romi_1, romi_2);
    romi_phase = np.floor((pha+np.pi)/(np.pi/4)).astype(int)
    romi_phase += 1
    romi_amp = np.sqrt(romi_1**2 + romi_2**2)
    romi_all_df = pd.DataFrame({'omi1':romi_1, 'omi2':romi_2, 'rmma':romi_amp, 'phase':romi_phase, 'date':romi_time, })
    
    nn = 5400
    cor = np.corrcoef(momi_1[i1:i1+nn], romi_1[:nn])[0,1], np.corrcoef(momi_2[i1:i1+nn], romi_2[:nn])[0,1], 
    print('correlation with ROMI from Kiladis et al')
    print(cor)
    
    if 1 == 1:
        X11 = np.loadtxt('./romi_1979-2016.txt')
        xd = X[:,-1] - X11[:,-1]
        print(np.max(xd))
        
        xd = X[:,-2] - X11[:,-2]
        print(np.max(xd))
        
        xd = X[:,-3] - X11[:,-3]
        print(np.max(xd))
