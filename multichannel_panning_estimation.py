# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:36:03 2025

@author: Damien
"""

import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import auralys_utils as au
import multichannel_layouts as mc
import re_model as re

plt.close("all")

layout_ids = list(mc._LAYOUTS.keys())
layout = random.choice(layout_ids)
layout = "13.1"

radius = 5

speakers = mc.get_spk_coordinates(layout, radius=radius)
nb_spk_tot = speakers["nb_spk"] + speakers["nb_lfe"]

# %% affichage
fig, ax = plt.subplots()

for spk in range(nb_spk_tot):
    ax.plot(
        speakers["positions_xy"][spk, 0],
        speakers["positions_xy"][spk, 1],
        "r",
        marker=(2, 1, -speakers["orient_deg"][spk]),
        markersize=10,
        label="Speakers",
    )

    # par convention les angles d'orientations sont données dans le sens horaire, et non dans le sens direct mathématique


# %% gains

from dbap import dbap_2d
from vbap import mdap_2d, vbap_2d

listen_pos = np.array([[0, 0]])
source = np.random.rand(1, 2) * (np.random.randint(0, 1, 2) * 2 - 1) * radius
source = np.array([[-1.01429807, -3.93939624]])

spat_delays = np.zeros(
    nb_spk_tot
)  # pas de délai dans les algorithmes de panning classiques

# gains = vbap_2d( speakers['positions_xy'], listen_pos , source)
# gains, spat_delays = dbap_2d(speakers['positions_xy'], source, rolloff=12)
gains = mdap_2d(speakers["positions_xy"], listen_pos, source, width=0.15)


speaker_ax_unit = np.column_stack(
    (
        np.cos(np.deg2rad(-1 * speakers["orient_deg"] + 90)),
        np.sin(np.deg2rad(-1 * speakers["orient_deg"] + (90))),
    )
)
speaker_axis = speakers["positions_xy"] + speaker_ax_unit

# %% remodel

re_vectors, w = re.compute_re_model(
    listen_pos, speakers["positions_xy"], speaker_axis, 100, gains, spat_delays
)
doa = np.degrees(np.arctan2(re_vectors[:, 0], re_vectors[:, 1]))

for spk in range(nb_spk_tot):
    ax.arrow(
        speakers["positions_xy"][spk, 0],
        speakers["positions_xy"][spk, 1],
        0,
        gains[spk] * radius / 2,
        linewidth=2,
    )

for st in range(len(listen_pos)):
    ax.arrow(
        listen_pos[st, 0],
        listen_pos[st, 1],
        re_vectors[st, 0] * radius / 2,
        re_vectors[st, 1] * radius / 2,
        head_width=0.2,
        head_length=0.2,
        length_includes_head=True,
        color="blue",
    )

    # calcul du centre de l'arc
    arc_cx = listen_pos[st, 0]
    arc_cy = listen_pos[st, 1]

    # rayon visuel de l'arc (ajuste selon ton échelle)
    arc_radius = (re_vectors[st, 0] ** 2 + re_vectors[st, 1] ** 2) ** 0.5 * radius / 2

    # orientation de l'arc = angle de la flèche en degrés
    arrow_angle = np.degrees(np.arctan2(re_vectors[st, 1], re_vectors[st, 0]))

    # création et ajout de l'arc
    arc = patches.Arc(
        (arc_cx, arc_cy),
        width=2 * arc_radius,
        height=2 * arc_radius,
        angle=arrow_angle,
        theta1=-w[st] / 2,
        theta2=w[st] / 2,
        color="blue",
        linewidth=2,
    )
    ax.add_patch(arc)

ax.plot(listen_pos[:, 0], listen_pos[:, 1], "ok")
ax.plot(source[:, 0], source[:, 1], "om")


ax.set_aspect("equal")
ax.grid(which="both", linestyle="--", alpha=0.4)  # type: ignore
ax.set_title("layout: " + layout + " - w: " + str(w) + "°")

# %%
att_db = -14.25
g = au.a2dB(gains) + att_db
g_lin = au.dB2a(g)
estimate_att = np.sum(g_lin**2) ** 0.5

print(au.a2dB(estimate_att))  # estimation du gain de mixage (à conserver)

comp_gain = 1 / (np.max(g_lin) / estimate_att)
comp_gain_db = au.a2dB(comp_gain)
print(comp_gain_db)
print(au.a2dB(gains))
