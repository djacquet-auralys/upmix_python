# -*- coding: utf-8 -*-
"""
Created on Thu May 22 17:59:50 2025

@author: Damien
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

import multichannel_layouts as mc
import auralys_utils as au
import re_model as re_model
from tdap import tdap

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
layout = random.choice(layout_ids)
# layout = "11.1"

radius = 15

speakers = mc.get_spk_coordinates(layout, radius=radius)
nb_spk = speakers['nb_spk']
nb_spk_tot = speakers['nb_spk']+speakers['nb_lfe']

#on trie les speakers non LFE

labels_arr = np.array(speakers['labels'])                              # array de str
mask       = ~np.char.startswith(labels_arr, "LFE")        # inversion du start-with
spk = speakers['positions_xy'][mask]
spk_orientations = speakers['orient_deg'][mask]
spk_azimuth = speakers['azimuth_deg'][mask]

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
# listening_zone = np.asarray(plt.ginput(-1, timeout=-1))

listening_zone = spk[np.argsort(spk_azimuth)]

xa, ya = zip(*np.vstack((listening_zone, listening_zone[0, :])))
plt_audience_contour, = ax.plot(xa, ya, 'cyan')
plt.draw()

'''------------------------------'''

# sampling audience area

# density of samples in m2
density = 0.5 # audience density
nb_max_seats = 50
nb_seats = 51

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
source = np.array([[-1,1]])*radius
ax.plot(source[:,0],source[:,1],'om')

spk_src_dist = np.zeros(nb_spk)
for i in range(nb_spk):
    spk_src_dist[i] = au.getDist(source, spk[i])


gains = np.zeros(nb_spk)

spat_delays = np.zeros(nb_spk)

# spat_delays = spk_src_dist/c_sound*1000
# spat_delays = spat_delays - np.min(spat_delays)

speaker_ax_unit = np.column_stack((np.cos(np.deg2rad(-1*spk_orientations+90)),np.sin(np.deg2rad(-1*spk_orientations+(90)))))
speaker_axis = spk + speaker_ax_unit

#%% remodel

print('\n\n-------------')
print('optimize gains')

def compute_spatialization_score(width ,spat_delays = spat_delays, target_width = 30, seats=seats,speaker_pts=spk,sources=source, speaker_axis = speaker_axis, directivity=directivity, plot=False, print_flag = True, color = 'b'):
        
    nb_spk = len(speaker_pts)
    
    central_seat = np.mean(seats, axis=0)
    
    pan = np.degrees(np.arctan2(source[:,0],source[:,1]))
    
    gains, dumb  = tdap(pan, width, speaker_pts)
        

    
    re, w = re_model.compute_re_model(seats, speaker_pts, speaker_axis, 100, gains, spat_delays)

    angle_diff = np.zeros(nb_seats)
    for s in range(nb_seats):
        angle_diff[s] = au.getAngle(seats[s,:], source[0], seats[s,:]+re[s,:])
        
    angle_diff =(( np.degrees(angle_diff) + 180) % 360) - 180
    
    
    
    
    jnd_theta = 3 #just noticeable difference of localization
    jnd_w = 10 #just noticeable difference of width
    λ = jnd_theta/jnd_w
    λ = 30

    dw = (w - target_width)
    eθ = angle_diff / jnd_theta
    ew = dw / jnd_w
    Eθ = np.hypot(eθ.mean(),  eθ.std(ddof=0))
    Ew = np.hypot(ew.mean(),  ew.std(ddof=0))
    angle_diff_score =  np.hypot(Eθ, np.sqrt(λ)*Ew)
        
    # angle_diff_score = np.mean(np.abs(angle_diff)) + np.std(angle_diff) + np.mean((w-target_width)) + np.std((w-target_width)) 
    # angle_diff_score = 1*np.mean(angle_diff**2) + 1*np.std(angle_diff) + 1*np.mean((w-target_width)**2)  + 1*np.std((w-target_width))
    
    
    if print_flag:
        print('width : ', w)
        #print('gains : ',np.round(gains,2))
        print('angle_diffs : ', (np.mean(np.abs(angle_diff))))
        print('re_width : ', np.mean(w))
        #print('angle_diff_score : ', angle_diff_score)
        print('\n')
        
    if plot:
        fig, ax = plt.subplots()
        for s in range(nb_spk):
            ax.arrow(spk[s, 0],
                     spk[s,1],
                     0,gains[s]*radius/2, linewidth=2, color='k')
    
        for st in range(len(seats)):
            ax.arrow(seats[st,0],seats[st,1],re[st,0]*radius/3, re[st,1]*radius/3,
                     head_width=0.2, head_length=0.2,
                     length_includes_head=True,
                     color=color)
            
            # calcul du centre de l'arc
            arc_cx = seats[st,0]
            arc_cy = seats[st,1]
        
            # rayon visuel de l'arc (ajuste selon ton échelle)
            arc_radius = (re[st,0]**2+re[st,1]**2)**0.5*radius/3
            
        
            # orientation de l'arc = angle de la flèche en degrés
            arrow_angle = np.degrees(np.arctan2(re[st,1], re[st,0]))
        
            # création et ajout de l'arc
            arc = patches.Arc(
                (arc_cx, arc_cy),
                width=2*arc_radius, height=2*arc_radius,
                angle=arrow_angle,
                theta1=-w[st]/2, theta2=w[st]/2,
                color='blue', linewidth=2
            )
            ax.add_patch(arc)
            
        plt.title('target: ' + str(target_width) + ', score:' + str(angle_diff_score) +', width:' + str(np.mean(w)))
   
    
    return angle_diff_score


# compute_spatialization_score(optimal_width, plot=True, print_flag = True)

#%%
width_to_test = np.arange(20., 300., 1)
target_width = np.arange(10,180,10)

angle_dev = np.zeros([len(target_width),len(width_to_test)])
optimal_width = np.zeros(len(target_width))

plt.figure()

for k in range(len(target_width)):
    
    print('target : ' , target_width[k])

    # for i in range(len(width_to_test)):
    #     angle_dev[i] =  compute_spatialization_score(width_to_test[i],target_width = target_width, plot=False, print_flag = True)
    
    # optimal_width = width_to_test[np.argmin(angle_dev)]
    # compute_spatialization_score(optimal_width,spat_delays = np.zeros(nb_spk), plot=True, print_flag = False)
    
    for i in range(len(width_to_test)):
        angle_dev[k,i] =  compute_spatialization_score(width_to_test[i],target_width = target_width[k], spat_delays = spat_delays, plot=False, print_flag = False)
    
    optimal_width[k] = width_to_test[np.argmin(angle_dev[k,:])]
    # compute_spatialization_score(optimal_width,spat_delays = spat_delays, plot=True, print_flag = False, color = 'r')
    
    # plt.figure()
    plt.plot(width_to_test, angle_dev[k,:]-np.min(angle_dev[k,:]))

# print('optimal_width : ', optimal_width)

# gains  = mdap_2d(spk,np.mean(seats, axis=0), source, width = optimal_width)
# print('gains : ', gains)

# compute_spatialization_score(optimal_width ,target_width = target_width[k], plot=True, print_flag = True)

plt.figure()
plt.plot(target_width, optimal_width)
# plt.title('target = ' + str(target_width))
