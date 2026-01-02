# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:47:51 2025

@author: Damien
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import re_model as re


plt.close('all')

#%% ——————————————————————————————
# 1) Configuration Dolby 7.1
radius = 2.0                                   # distance HP-centre (m)
speaker_order = ["L", "R", "C", "LFE", "LS", "RS", "LB", "RB"]
azimuth_deg = {                                # 0 ° = avant ; angles + vers la droite
    "L":  -30,  "R":   30,  "C":   0,  "LFE":   0,
    "LS": -90,  "RS":  90,  "LB": -150, "RB": 150,
}

# positions cartésiennes (x droite +, y avant +)
hp_positions = np.zeros([len(speaker_order),2])
for sp in range(len(speaker_order)):
    az = azimuth_deg[speaker_order[sp]]
    hp_positions[sp,:] = np.array([
        (radius * np.sin(np.deg2rad(az)),
         radius * np.cos(np.deg2rad(az)))])

# hp_positions = np.array([
#     (radius * np.sin(np.deg2rad(az)),
#      radius * np.cos(np.deg2rad(az)))
#     for az in [azimuth_deg[sp] for sp in speaker_order]
# ])

# orientation des axes (flèches) : chaque HP regarde le centre
hp_orientations_deg = np.array(
    [(az + 180) % 360 for az in azimuth_deg.values()]
)

# ——————————————————————————————
# 2) Carré de positions d’écoute 2 × 2 m (pas 0,5 m)
grid = np.linspace(-1, 1, 5)
xx, yy = np.meshgrid(grid, grid)
listen_pos = np.column_stack([xx.ravel(), yy.ravel()])

listen_pos = np.array([[0,0]])

# ——————————————————————————————
# 3) Gains simulés (source azimut +20 °)
gains = np.array([0.64278761, 0.98480775, 0.93969262, 0., 0., 0.34202014, 0., 0.])
# gains = np.array([0., 1, 1, 0., 0., 0., 0., 0.])
# gains = np.array([1., 0, 0, 0., 0., 0., 0., 0.])

# ——————————————————————————————
# 4) Tracé
fig, ax = plt.subplots(figsize=(7, 7))

# 4-a  positions d’écoute
ax.scatter(listen_pos[:, 0], listen_pos[:, 1],
           s=15, c='k', marker='x', label="Positions d'écoute")

# 4-b  haut-parleurs
norm = Normalize(0, 1)                               # pour la couleur de gain
cmap = plt.get_cmap('Blues')

for sp in range(len(speaker_order)):
    [x, y] = hp_positions[sp,:]
    orient_deg = hp_orientations_deg[sp]
    orient_rad = np.deg2rad(orient_deg)
    g = gains[sp]

    # carré orienté : orientation des côtés = flèche ⟂
    [x, y] = hp_positions[sp,:]
    square = RegularPolygon((x, y), numVertices=4, radius=0.25,
                            orientation=orient_rad + np.pi/4,
                            facecolor=cmap(norm(g)), edgecolor='black')
    ax.add_patch(square)

    # flèche orange : longueur ∝ gain
    ax.arrow(x, y,
             0.5 * g * np.sin(orient_rad),
             0.5 * g * np.cos(orient_rad),
             head_width=0.07, head_length=0.12,
             length_includes_head=True,
             color='orange')

    # étiquette (« L », « R », …) centrée
    ax.text(x, y, speaker_order[sp], ha='center', va='center',
            fontsize=9, weight='bold', color='white' if g > 0.3 else 'black')

# 4-c  colorbar des gains
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label('Gain normalisé', rotation=90)

# 4-d  habillage
ax.set_aspect('equal')
ax.set_xlabel('x (m, droite +)')
ax.set_ylabel('y (m, avant +)')
ax.set_title("Dolby 7.1 : orientation des haut-parleurs\n"
             "et carré de positions d’écoute 2 × 2 m")
ax.grid(True)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

#%% re model

spat_delays = np.zeros(len(speaker_order))
speaker_axis = np.zeros([len(speaker_order),2])

# gains = 20*np.log10(gains)

re_vectors,w = re.compute_re_model(listen_pos, hp_positions, speaker_axis, 100, gains, spat_delays)

for st in range(len(listen_pos)):
    ax.arrow(listen_pos[st,0],listen_pos[st,1],re_vectors[st,0], re_vectors[st,1],
             head_width=0.07, head_length=0.12,
             length_includes_head=True,
             color='red')