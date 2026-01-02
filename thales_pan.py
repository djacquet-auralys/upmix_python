# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 17:27:21 2025

@author: Damien
"""

import numpy as np
import matplotlib.pyplot as plt

import multichannel_layouts as mc
import tdap as tdap
import random

plt.close('all')

# def normalize_angle_signed(angle_deg):
#     """
#     Normalise un angle en degrés pour qu'il soit dans l'intervalle [-180°, 180°].

#     Paramètre :
#         angle_deg (float) : angle en degrés (peut être n'importe quel réel)

#     Retour :
#         float : angle normalisé dans [-180°, 180°]
#     """
#     angle = angle_deg % 360
#     if angle > 180:
#         angle -= 360
#     return angle

# def normalize_angle_custom(angle_deg, min_angle):
#     """
#     Normalise un angle en degrés dans l'intervalle [min_angle, min_angle + 360].

#     Paramètres :
#         angle_deg (float) : angle à normaliser (en degrés)
#         min_angle (float) : borne inférieure de l'intervalle (ex: -180, -90, 0)

#     Retour :
#         float : angle normalisé dans [min_angle, min_angle + 360]
#     """
#     return ((angle_deg - min_angle) % 360) + min_angle


#%% on affiche un système de haut parleurs

layout_ids = list(mc._LAYOUTS.keys())
layout = random.choice(layout_ids)
# layout = '11.1'

print(layout)

radius = 10

speakers = mc.get_spk_coordinates(layout, radius=radius)
nb_spk = speakers['nb_spk']
nb_spk_tot = speakers['nb_spk']+speakers['nb_lfe']

#on trie les speakers non LFE

labels_arr = np.array(speakers['labels'])                              # array de str
mask       = ~np.char.startswith(labels_arr, "LFE")        # inversion du start-with
spk = speakers['positions_xy'][mask]
spk_orientations = speakers['orient_deg'][mask]
spk_azimuth = speakers['azimuth_deg'][mask]

fig, ax1 = plt.subplots()

for s in range(speakers['nb_spk']):
    ax1.plot(spk[s, 0], spk[s, 1], 'r', marker=(2, 1, -spk_orientations[s]), markersize=10, label="Speakers")
    
    
#%%

pan = np.random.randint(-180,180)
width = np.random.randint(0,180)





src_angle = np.deg2rad((90 - pan) % 360)
source = np.array([np.cos(src_angle), np.sin(src_angle)])*radius
seats = np.array([[0,0]])


# speaker_ax_unit = np.column_stack((np.cos(np.deg2rad(-1*spk_orientations+90)),np.sin(np.deg2rad(-1*spk_orientations+(90)))))
# speaker_axis = spk + speaker_ax_unit



#%%

# g_nom = 1
# anglediff = 360
# n_essais = 0
# mu = 0.6

# gains = np.zeros(speakers['nb_spk'])

# while abs(anglediff)>3 and n_essais<300:
    
#     pan = normalize_angle_signed(pan + anglediff*mu)        
#     spk_azimuth = normalize_angle_custom(spk_azimuth, pan-180)
    
#     width = np.maximum(width,np.sort(np.abs(spk_azimuth-pan))[1])
  
        
#     for s in range(speakers['nb_spk']):
#         deltapan = np.abs(pan-spk_azimuth[s])
#         gains[s] = g_nom*(width-deltapan)/(width)
        
#     gains = np.maximum(gains,0)
#     gains=gains/np.sum(gains**2)**0.5
    
#     # estimation de la direction d'arrivée par somme de vecteur d'énergie
#     re = np.sum(gains.reshape(speakers['nb_spk'],1)**2*spk,axis=0)    
    
#     anglediff = np.degrees(np.arctan2(re[1],re[0])-src_angle)
#     anglediff = normalize_angle_signed(anglediff)
#     print("erreur : ", anglediff)
#     n_essais = n_essais+1
    
#%%    
    
gains, re = tdap.tdap(pan, width, spk)

# print('n_essais : ', n_essais)
print('gains : ', gains)

ax1.plot(source[0], source[1],'om')

fig, ax2 = plt.subplots()

for s in range(speakers['nb_spk']):
    ax2.plot(spk_azimuth,np.zeros(len(spk_azimuth)), '^r')

for s in range(speakers['nb_spk']):  
    ax2.arrow(spk_azimuth[s],0,0,gains[s], linewidth=2, color='k')
ax2.arrow(pan % 360,0,0,1,linewidth=2, color='m')

re_angle = np.degrees(np.arctan2(re[0],re[1]))
ax2.arrow(re_angle % 360,0,0,1,linewidth=2, color='b', linestyle = 'dashed')
    
for s in range(nb_spk):
    ax1.arrow(spk[s, 0],
              spk[s,1],
              0,gains[s]*radius/2, linewidth=2, color='k')
ax1.arrow(0,0,re[0],re[1], linewidth=2, color='b')

ax1.set_title('pan ' + str(pan) + ', width ' + str(width))


