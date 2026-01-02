# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:33:31 2025

@author: Damien
"""

import numpy as np
import matplotlib.pyplot as plt


def vbap_2d(
    spk_xy: np.ndarray,
    listener_xy: np.ndarray,
    source_xy: np.ndarray,
    tol: float = 1e-9
) -> np.ndarray:
    """
    Vector Base Amplitude Panning (VBAP) de Ville Pulkki en 2-D.

    (…code identique à votre dernière version…)
    """
    # --- 1. Vecteurs relatifs ---------------------------------------------
    spk_rel = spk_xy - listener_xy
    src_rel = source_xy - listener_xy
    src_vec = src_rel.ravel()

    if np.linalg.norm(src_vec) < tol:                       # source au centre
        gains = np.zeros(spk_xy.shape[0])
        gains[np.argmax(np.linalg.norm(spk_rel, axis=1))] = 1.0
        return gains

    # --- 2. Directions unitaires ------------------------------------------
    spk_dir = spk_rel / np.linalg.norm(spk_rel, axis=1, keepdims=True)
    src_dir = src_vec / np.linalg.norm(src_vec)

    # --- 3. Paires adjacentes ---------------------------------------------
    angles = (np.arctan2(spk_rel[:, 1], spk_rel[:, 0]) + 2*np.pi) % (2*np.pi)
    order = np.argsort(angles)
    angles_sorted = angles[order]
    src_angle = (np.arctan2(src_vec[1], src_vec[0]) + 2*np.pi) % (2*np.pi)

    N = spk_xy.shape[0]
    gains = np.zeros(N)

    for k in range(N):
        a1, a2 = angles_sorted[k], angles_sorted[(k + 1) % N]
        if k == N - 1:                                      # 2π ➜ 0
            a2 += 2*np.pi
        if a1 - tol <= src_angle <= a2 + tol:
            i, j = order[k], order[(k + 1) % N]
            L = np.column_stack((spk_dir[i], spk_dir[j]))
            if abs(np.linalg.det(L)) < tol:
                break                                       # colinéaire → fallback

            g_pair = np.linalg.solve(L, src_dir)
            if np.all(g_pair >= -tol):
                gains[i], gains[j] = g_pair
                gains[gains < 0] = 0.0
                gains /= np.linalg.norm(gains)
                return gains

    # --- 4. Fallback -------------------------------------------------------
    idx = np.argmax(spk_dir @ src_dir)
    gains[idx] = 1.0
    return gains


# --------------------------------------------------------------------------- #
#                     Multiple Direction Amplitude Panning                   #
# --------------------------------------------------------------------------- #

def mdap_2d(
    spk_xy: np.ndarray,
    listener_xy: np.ndarray,
    source_xy: np.ndarray,
    width: float = 0.0,
    tol: float = 1e-9
) -> np.ndarray:
    """
    MDAP (Multiple-Direction Amplitude Panning) 2-D reposant sur VBAP.

    La *largeur* (0 ≤ width ≤ 1) élargit la source en lissant
    l’énergie `VBAP` sur le cercle des haut-parleurs à l’aide d’un noyau
    gaussien ; width = 0 reproduit exactement VBAP, width = 1 fournit une
    source couvrant tout le tour.

    Paramètres
    ----------
    spk_xy      : (N, 2) ndarray – positions des haut-parleurs
    listener_xy : (1, 2) ndarray – position d’écoute
    source_xy   : (1, 2) ndarray – position de la source virtuelle
    width       : float          – 0 (ponctuelle) … 1 (omnidirectionnelle)
    tol         : float          – tolérance numérique

    Retour
    ------
    gains : (N,) ndarray – gains linéaires normalisés (∑ g² = 1)
    """
    # 1) gains VBAP de base
    g_vbap = vbap_2d(spk_xy, listener_xy, source_xy, tol=tol)

    # 2) si width ~ 0, rendre directement les gains VBAP
    if width <= tol:
        return g_vbap.copy()

    # 3) distribution d’énergie VBAP (e = g²)
    e = g_vbap ** 2
    N = e.size

    # 4) noyau gaussien circulaire (σ proportionnel à width)
    #    σ = width * (N / 4)  → width=1 ≃ 90° (= quart du cercle)
    sigma = max(width * N / 4, 1e-3)
    K = int(np.ceil(3 * sigma))               # ±3σ → noyau quasi-complet
    k = np.arange(-K, K + 1)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()

    # 5) convolution circulaire (np.roll) pour lisser l’énergie
    e_blur = np.zeros_like(e)
    for idx in range(N):
        e_blur[idx] = np.sum(e[(idx + k) % N] * kernel)

    # 6) retour aux gains + normalisation d’énergie
    g_blur = np.sqrt(e_blur)
    g_blur /= np.linalg.norm(g_blur) + 1e-12

    return g_blur


# --------------------------------------------------------------------------- #
#                              Exemple d’usage                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # # carré de 4 HP
    # spk_xy = np.array([[ 1,  1],
    #                    [-1,  1],
    #                    [-1, -1],
    #                    [ 1, -1]], dtype=float)

    # listener_xy = np.array([[0., 0.]])
    # source_xy   = np.array([[0.7, 0.2]])

    # print("VBAP :", np.round(vbap_2d(spk_xy, listener_xy, source_xy), 3))
    # for w in (0.25, 0.50, 0.75, 1.0):
    #     print(f"MDAP width={w:.2f} :", np.round(mdap_2d(spk_xy, listener_xy,
                                                        # source_xy, width=w), 3))
                                                        
                                                        
    # ---------------------------------------------------------------------------
    # 2) Configuration : cercle de 10 haut-parleurs (rayon = 1)
    # ---------------------------------------------------------------------------
    nb_spk = 20
    angles = 2 * np.pi * np.arange(nb_spk) / nb_spk          # 0 … 2π
    spk_xy = np.column_stack((np.cos(angles), np.sin(angles)))  # (N, 2)
    
    listener_xy = np.array([[0., 0.]])
    source_xy   = np.array([[0.7, 0.2]])                     # même source
    
    # ---------------------------------------------------------------------------
    # 3) Calcul des gains VBAP
    # ---------------------------------------------------------------------------
    wdthtotest=np.arange(0.0,0.5,0.05)
    
    for w in wdthtotest:
        gains_vbap = mdap_2d(spk_xy, listener_xy, source_xy, width=w)
        
        # ---------------------------------------------------------------------------
        # 4) Affichage
        # ---------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('VBAP – cercle de 10 HP')
        
        # HP (triangles)
        ax.scatter(spk_xy[:, 0], spk_xy[:, 1], marker='^', label='Haut-parleurs')
        
        # Source (rond)
        ax.scatter(source_xy[0, 0], source_xy[0, 1], marker='o', label='Source')
        
        # Barres verticales = gains (échelle 0->0.5)
        bar_scale = 0.5
        for (x, y), g in zip(spk_xy, gains_vbap):
            ax.plot([x, x], [y, y + g * bar_scale], linewidth=2, color='k')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.legend(loc='upper right')
        plt.show()
