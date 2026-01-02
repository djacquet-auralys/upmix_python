# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 12:28:55 2025

@author: Damien

Upmix Auralys POC

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.patches as patches
import matplotlib.tri as tri
import auralys_utils as au
import re_model as re_model

import json
import  easygui as eg
from tdap import tdap
from pythonosc import udp_client

plt.close('all')


directivity = 110 # aperture of the speakers
sref = 105  # level at 1 dref
dref = 1  # reference distance (m)
c_sound = 343

attack = 1.
release = 50.
release_min = 50. # release for the center extract
release_max = 200. # release for the side extract
slope = 500 # default slope
min_gain = -40  # minimum gain in dB

max_sources = 11


# get width of the panorama in °
panorama_width = 200

# rejection target at the middle of the adjacent band
rej_gain = -20

# maximum number of speaker the algorithm can handle
max_spk = 16


'''---------------------------
Configure Venue
---------------------------'''

venue_img_file = eg.fileopenbox()

# display venue plan
scale_fig, ax = plt.figure(), plt.subplot()
venue_img = plt.imread(venue_img_file)
plt.imshow(venue_img)

# get scale
plt.title("select 2 points of known distance \n")
pixels_scale_pts = np.asarray(plt.ginput(2, timeout=-1))
plt.plot([pixels_scale_pts[0, 0], pixels_scale_pts[1, 0]],
         [pixels_scale_pts[0, 1], pixels_scale_pts[1, 1]],
         'r', linestyle='dashed', linewidth=3)

pixels_distance = int(eg.integerbox(msg='enter the distance',
                                 title='scaling distance'))

px_segment = pixels_scale_pts[1, :]-pixels_scale_pts[0, :]
scale_factor = pixels_distance/np.sqrt(px_segment.T @ px_segment)

print('\n\n')

# %% redraw in true scale
venue_fig = plt.figure(figsize=scale_fig.get_size_inches(), dpi=scale_fig.dpi)
venue_ax = plt.subplot()
venue_dim = np.array(np.shape(venue_img)[:2])*scale_factor
venue_dim = np.flip(venue_dim)
extent = [0, venue_dim[0], 0, venue_dim[1]]

plt.close(fig=scale_fig)

alphas = np.ones(venue_img.shape[0:2])*0.5

plt_venue_bck = venue_ax.imshow(venue_img, extent=extent, alpha = alphas)

x_max = venue_dim[0]
y_max = venue_dim[1]

venue_ax.set_xlim([-x_max*0.2, x_max*1.2])
venue_ax.set_ylim([-y_max*0.2, y_max*1.2])
'''------------------------------'''
# %% get audience location

plt.title("draw the contour of the audience \n middle button to exit \n")
audience_pts = np.asarray(plt.ginput(-1, timeout=-1))

# %%create lists of x and y values
xa, ya = zip(*np.vstack((audience_pts, audience_pts[0, :])))
plt_audience_contour, = venue_ax.plot(xa, ya, 'm')
plt.draw()

'''------------------------------'''

# sampling audience area

# density of samples in m2
density = 1 # audience density
nb_max_seats = 200
nb_seats = 201

x_audience_min = np.min(audience_pts, axis=0)[0]
x_audience_max = np.max(audience_pts, axis=0)[0]
y_audience_min = np.min(audience_pts, axis=0)[1]
y_audience_max = np.max(audience_pts, axis=0)[1]


while nb_seats>nb_max_seats:
    x_pts = np.arange(x_audience_min, x_audience_max, density)
    y_pts = np.arange(y_audience_min, y_audience_max, density)

    x_pts, y_pts = np.meshgrid(x_pts, y_pts)

    audience_path = mpltPath.Path(audience_pts)

    seats = np.hstack((x_pts.reshape((np.size(x_pts), 1)),
                      y_pts.reshape((np.size(y_pts), 1))))
    inside = audience_path.contains_points(seats)

    seats = seats[np.where(inside)]

    nb_seats = np.shape(seats)[0]
    density = density+0.5

plt_seats, = venue_ax.plot(seats[:,0],seats[:,1],'om')
plt.draw()





'''------------------------------'''
# %% get speakers location
plt.title("for each speaker, click first on its position and then on its axis direction. reapet for every speaker \n middle button to exit \n")
input_spk_pts = np.asarray(plt.ginput(-1, timeout=-1))
speaker_pts = input_spk_pts[0:-1:2,:]
nb_spk = speaker_pts.shape[0]
speaker_axis = input_spk_pts[1::2,:]

directivity = int(eg.integerbox(msg='enter the speaker aperture',
                                 title='loudspeaker aperture', default = directivity ,lowerbound=0, upperbound=360))

# %% create lists of x and y values
xspk, yspk = zip(*np.vstack((speaker_pts, speaker_pts[0, :])))

spk_orientations = np.zeros(nb_spk)
for i in range(nb_spk):
    spk_orientations[i] = au.getAngle(speaker_pts[i,:], np.array([speaker_pts[i,0]+1,speaker_pts[i,1]]), speaker_axis[i,:])
    venue_ax.plot(xspk[i], yspk[i], 'r',marker=(2, 1, np.degrees(spk_orientations[i])-90), markersize=10)


'''------------------------------'''
# %% place upmix sources
# les distances ne sont pas utiles

nb_source = np.min([nb_spk + (1- (nb_spk % 2)),max_sources])
colors = [plt.cm.hsv(i / nb_source) for i in range(nb_source)]

# get center of the audience
plt.title("click on the center of the audience")
plt.draw()
audience_bary = np.asarray(plt.ginput(1, timeout=-1))[0]
venue_ax.plot(audience_bary[0],audience_bary[1],'*r')

# get center of the stage
plt.title("click on the center position of the panorama")
plt.draw()
panorama_center = np.asarray(plt.ginput(1, timeout=-1))


spk_angles = np.degrees(au.getAngle(audience_bary, speaker_pts))



panorama_width = np.max(spk_angles)-np.min(spk_angles)
panorama_width = int(eg.integerbox(msg='enter panorama width in °',
                                 title='panorama width', default = panorama_width*0.8 ,lowerbound=60, upperbound=panorama_width))
src_angles = np.linspace(0, panorama_width, nb_source) + np.degrees(au.getAngle(audience_bary, panorama_center))-panorama_width/2
src_angles = au.normalize_angle_custom(src_angles,-180)

spk_dist = au.getDist(speaker_pts, audience_bary)

sorted_indices = np.argsort(spk_angles)
spk_angles_sorted = spk_angles[sorted_indices]
spk_dist_sorted = spk_dist[sorted_indices]

src_dist = np.interp(src_angles, spk_angles_sorted, spk_dist_sorted)*1.2
sources = au.pol2cart(src_dist, np.deg2rad(src_angles)) + audience_bary

#%%
for i in range(nb_source):
    plt.plot(sources[i,0], sources[i,1],'o', color = colors[i])
    
# %% compute gains for sources

spread = 0

width = panorama_width/nb_source*2*au.map_affine(spread, 0, 1, 0.7, 3)
slope = au.map_affine(spread, 0, 1, 500, 50)


gains = np.zeros([nb_source, nb_spk])
spat_delays = np.zeros([nb_source, nb_spk])

pan = np.zeros(nb_source)

plt.figure()
for k in range(nb_source):
    src = (sources-audience_bary)[k,:]
    pan[k] = np.degrees(np.arctan2(src[0], src[1]))
    gains[k,:], dump = tdap(pan[k], width, speaker_pts-audience_bary, -180)
    for j in range(nb_spk):
        spat_delays[k,j] = au.getDist(sources[k], speaker_pts[j])/340*1000*1
    plt.plot(gains[k,:],color = colors[k])

# %% compute re model

w = np.zeros([nb_source, nb_seats])
re = np.zeros([nb_source, nb_seats, 2])
perceived_source = np.zeros([nb_source, nb_seats, 2])
angle_deviation = np.zeros([nb_source, nb_seats])
angle_seat_source = np.zeros([nb_source, nb_seats])
angle_perceived_source =  np.zeros([nb_source, nb_seats])


for k in range(nb_source):
    re[k,:,:], w[k,:] = re_model.compute_re_model(seats, speaker_pts, speaker_axis, directivity, gains[k,:], spat_delays[k,:])
    
    
    for s in range(nb_seats):
        perceived_source[k,s,:] = seats[s]+re[k,s,:]*0.5
        angle_perceived_source[k,s] = au.getAngle(seats[s], perceived_source[k,s,:])
        angle_seat_source[k, s] = au.getAngle(seats[s],sources[k])
        angle_deviation[k,s] = (angle_seat_source[k, s]-angle_perceived_source[k,s] + np.pi)%(2*np.pi) - np.pi

        # x_src = perceived_source[k,s,0]
        # y_src = perceived_source[k,s,1]
        # venue_ax.plot(x_src,y_src,'o', color= colors[k], markersize=5)

# déviation moyenne      
mean_deviation = np.degrees(np.mean(np.abs(angle_deviation),axis=0))

# fidélité directionnelle : ramené entre 0 et 1
JNDdir = 10 # just noticiable difference in °
Qdir = np.maximum(0,1-mean_deviation/JNDdir)
# Qdir est haut si la déviation est basse


# régularité des espacements
AA = np.zeros([nb_source, nb_seats])

for s in range(nb_seats):
    AA[:,s] = au.normalize_angle_custom( np.degrees(angle_perceived_source[:,s]), np.degrees(au.getAngle(audience_bary, panorama_center))-180)

std_perceived_sources = np.degrees(np.std(np.diff(AA,axis=0),axis=0))
mean_angle_between_sources = np.degrees(np.mean(np.diff(AA,axis=0),axis=0))
# coefficient de variation = variation autour de la moyenne
cv_angle_between_sources = std_perceived_sources/mean_angle_between_sources

CV0 = 0.6
Qspace = np.maximum(0, 1-cv_angle_between_sources/CV0)
# Qspace bas s'il y a des zones compréssées / étirées, haut si c'est régulier

# régularité des largeurs
mean_width = np.mean(w,axis=0)
std_width = np.std(w,axis=0)
cv_width = std_width/mean_width
Qwidth = np.maximum(0,1-cv_width/CV0)
#Qwidth bas s'il y a des largeurs très différentes, haut si c'est régulier

# recouvrement
Qblur = 1-np.abs(np.sum(w,axis=0)/(np.max(AA,axis=0)-np.min(AA,axis=0))-1)
Qblur = np.maximum(Qblur, 0)

# perceived_aperture = np.max(AA, axis =0)-np.min(AA, axis =0)

a1 = 0.7
a2 = 0.1
a3 = 0.1
a4 = 0.1

# a1 = 0
# a2 = 0
# a3 = 0
# a4 = 1

seat_score = (a1*Qdir + a2*Qspace + a3*Qwidth +a4*Qblur)


triang = tri.Triangulation(seats[:,0], seats[:,1])
score_map = venue_ax.tripcolor(triang, seat_score, shading='gouraud', cmap='RdYlGn', vmin=-0.1, vmax=0.9)
# score_map.set_clim(0, 100)

 
print('mean score : ', np.mean(seat_score))

for k in range(nb_source):
    for s in range(nb_seats): 
        arc_radius = density*0.8
        angle = np.degrees((angle_perceived_source[k,s]))
        theta1 = -w[k,s]/2
        theta2 = w[k,s]/2
        
        arc = patches.Arc((seats[s]), arc_radius, arc_radius,angle = angle, theta1=theta1, theta2 = theta2, color= colors[k], linewidth = 2)
        venue_ax.add_patch(arc)
        
#%% write json dict
eps = 0.0001

extract_width = 2/(nb_source)
extract_pan = np.zeros(max_sources)
extract_pan[0:nb_source] =  np.flip(np.arange(-1+extract_width/2,1-extract_width/2+eps,extract_width))

release = np.abs(extract_pan)*(release_max-release_min) + release_min

filename = "C:/Users/Damien/Documents/Auralys/Python/auralys_upmix/upmixparams.json"
# filename = eg.filesavebox(title = 'save upmix parameters', default = filename)

# Convert int32 values to native Python types (int or float)
nb_source = int(nb_source)
slope = float(slope)
gains = [[float(g) for g in row] for row in gains]
delays = [[float(d) for d in row] for row in spat_delays]

upmix_params = {
    "width": float(extract_width),  # Convert to float
    "slope": float(slope), 
    "min_gain": float(min_gain)
}

for s in range(max_sources):
    source_data = {
        "pan"+str(s+1): float(extract_pan[s]),
        "gains"+str(s+1): gains[s],
        "delays"+str(s+1): [d for d in spat_delays[s]],
        "release"+str(s+1): float(release[s]),
        "attack": float(attack),   # Convert to float
    }
    upmix_params.update(source_data)


try:
    with open(filename, 'w') as file:
        json.dump(upmix_params, file, indent=4)
except IOError:
    print("An error occurred while writing to the file.")
    
client = udp_client.SimpleUDPClient("127.0.0.1", 6789)
json_str = json.dumps(upmix_params)
client.send_message("/upmix", json_str)

        
