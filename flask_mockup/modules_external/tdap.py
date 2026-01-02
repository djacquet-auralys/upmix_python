# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:18:51 2025

@author: Damien

Amplitude panning based on triangular gain distribution.

"""

import numpy as np
import matplotlib.pyplot as plt

def normalize_angle_signed(angle_deg):
    """
    Normalise un angle en degrés pour qu'il soit dans l'intervalle [-180°, 180°].

    Paramètre :
        angle_deg (float) : angle en degrés (peut être n'importe quel réel)

    Retour :
        float : angle normalisé dans [-180°, 180°]
    """
    angle = angle_deg % 360
    if angle > 180:
        angle -= 360
    return angle

def normalize_angle_custom(angle_deg, min_angle):
    """
    Normalise un angle en degrés dans l'intervalle [min_angle, min_angle + 360].

    Paramètres :
        angle_deg (float) : angle à normaliser (en degrés)
        min_angle (float) : borne inférieure de l'intervalle (ex: -180, -90, 0)

    Retour :
        float : angle normalisé dans [min_angle, min_angle + 360]
    """
    return ((angle_deg - min_angle) % 360) + min_angle


def tdap(pan, width, spk, min_angle=None):
    """
    amplitude panning. the gains are generated from a triangular distribution centered on the  pan angle and accross the speakers

    Parameters
    ----------
    pan : panning angle (azimtuh) in degrees.
    
    width : width angle in degrees between 0 and 180°

    spk : 2D array of speakers coordinates x and y

    Returns
    -------
    gains : 1D array of gains (linear)
    
    re : vector indicating the perceived position

    """
    
    width = np.minimum(np.maximum(width, 0), 300)
    
    spk_azimuth = np.degrees(np.arctan2(spk[:,0],spk[:,1]))    
    nb_spk = len(spk_azimuth)
    gains = np.zeros(nb_spk)
    anglediff = 360
    mu = 0.8
    n_essais = 0
    g_nom = 1
    
    src_angle = np.deg2rad((90 - pan) % 360)


    while abs(anglediff)>1 and n_essais<100:
        
        if n_essais>0:
            pan = normalize_angle_signed(pan + anglediff*mu)
        
        if min_angle is None:
            spk_azimuth = normalize_angle_custom(spk_azimuth, pan-180)
        else:
            spk_azimuth = normalize_angle_custom(spk_azimuth, min_angle)
            
        width = np.maximum(width,np.sort(np.abs(spk_azimuth-pan))[1])
      
            
        for s in range(nb_spk):
            deltapan = np.abs(pan-spk_azimuth[s])
            gains[s] = g_nom*(width-deltapan)/(width)
            
        gains = np.maximum(gains,0)
        
        
        if gains.any():
            gains=gains/np.sum(gains**2)**0.5
        
        # estimation de la direction d'arrivée par somme de vecteur d'énergie
        re = np.sum(gains.reshape(nb_spk,1)**2*spk,axis=0)    
        
        anglediff = np.degrees(np.arctan2(re[1],re[0])-src_angle)
        anglediff = normalize_angle_signed(anglediff)
        n_essais = n_essais +1
        
        
    return gains, re
    