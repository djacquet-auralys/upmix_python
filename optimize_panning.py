# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:23:07 2025

@author: Damien

optimisation du KNN par Re pour un nombre donné de places
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

import multichannel_layouts as mc
import auralys_utils as au
import re_model as re_model
from vbap import vbap_2d
from dbap import dbap_2d
import random
import matplotlib.patches as patches
from scipy.optimize import minimize, Bounds

plt.close('all')

directivity = 110 # aperture of the speakers
sref = 105  # level at 1 dref
dref = 1  # reference distance (m)
c_sound = 343



#%% on affiche un système de haut parleurs

layout_ids = list(mc._LAYOUTS.keys())
# layout = random.choice(layout_ids)
layout = "11.1"

radius = 10

speakers = mc.get_spk_coordinates(layout, radius=radius)
nb_spk = speakers['nb_spk']
nb_spk_tot = speakers['nb_spk']+speakers['nb_lfe']

#on trie les speakers non LFE

labels_arr = np.array(speakers['labels'])                              # array de str
mask       = ~np.char.startswith(labels_arr, "LFE")        # inversion du start-with
spk = speakers['positions_xy'][mask]
spk_orientations = speakers['orient_deg'][mask]

#%% affichage
fig, ax = plt.subplots()

for s in range(speakers['nb_spk']):
    ax.plot(spk[s, 0], spk[s, 1], 'r', marker=(2, 1, -spk_orientations[s]), markersize=10, label="Speakers")
    
    # par convention les angles d'orientations sont données dans le sens horaire, et non dans le sens direct mathématique

# Définition des limites des axes
ax.set_xlim(-1.5*radius, 1.5*radius)
ax.set_ylim(-1.5*radius, 1.5*radius)

# Ajout d'une grille pour une meilleure visibilité
ax.grid(True)
ax.set_aspect('equal')

# Affichage de la figure
plt.show()

# %% get audience location

plt.title("draw the contour of the audience \n middle button to exit \n")
listening_zone = np.asarray(plt.ginput(-1, timeout=-1))


xa, ya = zip(*np.vstack((listening_zone, listening_zone[0, :])))
plt_audience_contour, = ax.plot(xa, ya, 'cyan')
plt.draw()

'''------------------------------'''

# sampling audience area

# density of samples in m2
density = 0.5 # audience density
nb_max_seats = 200
nb_seats = 201

x_audience_min = np.min(listening_zone, axis=0)[0]
x_audience_max = np.max(listening_zone, axis=0)[0]
y_audience_min = np.min(listening_zone, axis=0)[1]
y_audience_max = np.max(listening_zone, axis=0)[1]


while nb_seats>nb_max_seats:
    x_pts = np.arange(x_audience_min, x_audience_max, density)
    y_pts = np.arange(y_audience_min, y_audience_max, density)

    x_pts, y_pts = np.meshgrid(x_pts, y_pts)

    audience_path = mpltPath.Path(listening_zone)

    seats = np.hstack((x_pts.reshape((np.size(x_pts), 1)),
                      y_pts.reshape((np.size(y_pts), 1))))
    inside = audience_path.contains_points(seats)

    seats = seats[np.where(inside)]

    nb_seats = np.shape(seats)[0]
    density = density+0.5

plt_seats, = ax.plot(seats[:,0],seats[:,1],'.b')
plt.draw()

#%% set source and gains
source = np.array([au.pol2cart(radius+1,np.random.rand()*7)])
# source = np.array([[-11,5]])
ax.plot(source[:,0],source[:,1],'om')

spk_src_dist = np.zeros(nb_spk)
for i in range(nb_spk):
    spk_src_dist[i] = au.getDist(source, spk[i])

nb_playing_speakers=int(np.round(nb_spk+1.1/2))
nb_playing_speakers = np.minimum(nb_playing_speakers, nb_spk)

closest_idx = np.argsort(spk_src_dist)[:nb_playing_speakers]

gains_closets_spk = np.ones(nb_playing_speakers)

gains = np.zeros(nb_spk)
gains[closest_idx] = np.ones(len(closest_idx))
spat_delays = np.zeros(nb_spk)



speaker_ax_unit = np.column_stack((np.cos(np.deg2rad(-1*spk_orientations+90)),np.sin(np.deg2rad(-1*spk_orientations+(90)))))
speaker_axis = spk + speaker_ax_unit

#%% remodel

print('\n\n-------------')
print('optimize gains')

def compute_angle_diff(gains,spat_delays = spat_delays[closest_idx], seats=seats,speaker_pts=spk[closest_idx],sources=source, speaker_axis = speaker_axis[closest_idx], directivity=directivity, plot=False, print_flag = True):
        
    nb_spk = len(speaker_pts)
    
    re, w = re_model.compute_re_model(seats, speaker_pts, speaker_axis, 100, gains, spat_delays)
    doa = np.degrees(np.arctan2(re[:,0],re[:,1]))
    
    angle_diff = np.zeros(nb_seats)
    for s in range(nb_seats):
        angle_diff[s] = au.getAngle(seats[s,:], source[0], seats[s,:]+re[s,:])
        
    angle_diff =(( np.degrees(angle_diff) + 180) % 360) - 180

        
    angle_diff_score = np.mean(np.abs(angle_diff))
    # angle_diff_score = np.sum(angle_diff**2)*np.sum(w)
    if print_flag:
        print('gains : ',np.round(gains,2))
        # print('angle_diffs : ', np.floor(angle_diff))
        print('angle_diff_score : ', angle_diff_score)
    
    if plot:
        for s in range(nb_spk):
            ax.arrow(spk[s, 0],
                     spk[s,1],
                     0,(au.a2dB(gains[s])+60)/60*radius/2, linewidth=2)
    
        for st in range(len(seats)):
            ax.arrow(seats[st,0],seats[st,1],re[st,0]*radius/3, re[st,1]*radius/3,
                     head_width=0.2, head_length=0.2,
                     length_includes_head=True,
                     color='blue')
            
            # # calcul du centre de l'arc
            # arc_cx = seats[st,0]
            # arc_cy = seats[st,1]
        
            # # rayon visuel de l'arc (ajuste selon ton échelle)
            # arc_radius = (re[st,0]**2+re[st,1]**2)**0.5*radius/3
            
        
            # # orientation de l'arc = angle de la flèche en degrés
            # arrow_angle = np.degrees(np.arctan2(re[st,1], re[st,0]))
        
            # # création et ajout de l'arc
            # arc = patches.Arc(
            #     (arc_cx, arc_cy),
            #     width=2*arc_radius, height=2*arc_radius,
            #     angle=arrow_angle,
            #     theta1=-w[st]/2, theta2=w[st]/2,
            #     color='blue', linewidth=2
            # )
            # ax.add_patch(arc)
   
    
    return angle_diff_score


bounds = Bounds(np.zeros(nb_playing_speakers), np.ones(nb_playing_speakers))
result = minimize(compute_angle_diff,gains_closets_spk, bounds = bounds, method='L-BFGS-B')

optimal_gains = result.x
optimal_gains = optimal_gains/np.sum(optimal_gains**2)**0.5

gains[closest_idx] = optimal_gains
gains[gains<0.0001] = 0.

compute_angle_diff(gains, speaker_pts=spk,spat_delays=spat_delays, speaker_axis = speaker_axis,  plot=True)


print('\n')
print(au.a2dB(gains))

#%% optimize delays

print('\n\n-------------')
print('optimize delays')

def compute_angle_diff2(spat_delays, gains = gains[closest_idx], seats=seats,speaker_pts=spk[closest_idx],sources=source, speaker_axis = speaker_axis[closest_idx], directivity=directivity, plot=False, print_flag=False):
    print('delays : ',np.round(spat_delays*1000,2))
    return compute_angle_diff(gains, spat_delays = spat_delays, seats=seats,speaker_pts=speaker_pts,sources=source, speaker_axis = speaker_axis, directivity=directivity, plot=False, print_flag=print_flag)

bounds = Bounds(np.zeros(nb_playing_speakers), np.ones(nb_playing_speakers)*radius/c_sound*2)

# delays_closest_spk = spk_src_dist[closest_idx]/c_sound
delays_closest_spk = np.zeros(nb_playing_speakers)

result = minimize(compute_angle_diff2,delays_closest_spk, bounds = bounds, method='L-BFGS-B')
optimal_delays = result.x
optimal_delays = optimal_delays - np.min(optimal_delays)

# spat_delays[closest_idx] = optimal_delays
spat_delays[closest_idx] = optimal_delays

compute_angle_diff(gains, speaker_pts=spk,spat_delays=spat_delays, speaker_axis = speaker_axis,  plot=True, print_flag=True)

print('\n')
print('delays : ', 1000*spat_delays)
print('gains : ',au.a2dB(gains))
