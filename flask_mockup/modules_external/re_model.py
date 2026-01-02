# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:07:04 2025

@author: Damien
"""


import numpy as np
import matplotlib.pyplot as plt
import auralys_utils as au

sref = 105  # level at 1 dref
dref = 1  # reference distance (m)
c_sound = 343

def compute_re_model(seat, speakers, speaker_axis, directivity, spat_gains, spat_delays):
    '''

    Parameters
    ----------
    seat : 2D array of listening positions coordinates.
    speakers : 2D array of speakers positions coordinates.
    speaker_axis : 2D array of axis coordinate.
    directivity : angle of aperture.
    spat_gains : in dB.
    spat_delays : in ms.

    Returns
    -------
    re : array of re vectors for each seat
    w: estimated width in degree

    '''
    
    nb_spk = speakers.shape[0]
    nb_seat = seat.shape[0]
    #--------------------------------------------------------------------------
    

    spat_gains = au.a2dB(spat_gains)

    #--------------------------------------------------------------------------
    # compute gains and delays for acoustic propagation
    st_levels = np.zeros([nb_seat, nb_spk])

    for st in np.arange(nb_seat):
        for ls in np.arange(nb_spk):
            angle = au.getAngle(speakers[ls], seat[st], speaker_axis[ls])
            angle_att_dB = au.a2dB(au.custom_cardioid(angle, np.deg2rad(directivity)))
            dist = au.getDist(speakers[ls], seat[st])
            dist_att_dB = au.inv_square_att_law_dB(dist, dref, sref) - sref
            st_levels[st, ls] = angle_att_dB + dist_att_dB

    seat_delays = np.zeros([nb_seat, nb_spk])

    for s in np.arange(nb_seat):
        for i in range(nb_spk):
            seat_delays[s, i] = au.getDist(seat[s], speakers[i])/c_sound


    #--------------------------------------------------------------------------
    #compute total gains and delays
    total_delays = np.zeros([nb_seat, nb_spk])
    

    for se in np.arange(nb_seat):
        for sp in np.arange(nb_spk):
            total_delays[se, sp] = seat_delays[se,sp] + spat_delays[sp]

    total_gains = np.zeros([nb_seat, nb_spk])

    for se in np.arange(nb_seat):
        for sp in np.arange(nb_spk):
            total_gains[se, sp] = spat_gains[sp] + st_levels[se,sp]

    total_gains_norm = (total_gains - np.min(total_gains))/(np.max(total_gains)-np.min(total_gains))


    #--------------------------------------------------------------------------
    # energy vector extended


    delta_t = total_delays-np.min(total_delays,axis=1)[:,np.newaxis]
    tau = -2.5/4 #-1/8 for stationary sound, -5/4 for impulsive sounds in dB/ms
    re = np.zeros([nb_seat, 2])
    w = np.zeros(nb_seat)
    
    for st in np.arange(nb_seat):
        
        # creation of unity vector from each seat toward each speaker
        theta = (speakers-seat[st,:])
        theta = theta/((theta[:,0]**2+theta[:,1]**2)**0.5)[:,np.newaxis] # normalization

        wt = au.dB2a(tau*delta_t[st,:])

        re[st,:] = np.sum(((wt*au.dB2a(total_gains[st,:]))**2)[:,np.newaxis]*theta, axis=0)/np.sum((wt*au.dB2a(total_gains[st,:]))**2)
        re_norm = (re[st,0]**2+re[st,1]**2)**0.5
        # w[st] =186.4*(1-re_norm)+10.7
        w[st] = 5/8*180/np.pi*2*np.arccos(re_norm)
        
        
        
    return re, w
