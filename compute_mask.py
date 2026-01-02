# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def compute_att(p, w, x, slope, plot=False):
    """
    p      : centre (scalaire)
    w      : largeur (scalaire)
    slope  : pente en dB / 1 de panning (scalaire)
    x      : vecteur de panning
    """
    # dB
    att_dB = np.minimum(slope * (w/2 - np.abs(x - p)) + 1, 1) - 1
    # lin
    att = 10**(att_dB / 20.0)

    if plot:
        plt.plot(x, att_dB)
        plt.grid('minor')

    return att_dB, att


def compute_global_gain(p_extract, w, slope, x, E_target=1.0):
    """
    p_extract : tableau des centres (nb_sources,)
    w         : scalaire OU tableau (nb_sources,)
    slope     : scalaire OU tableau (nb_sources,)
    x         : vecteur de panning
    """
    p_extract = np.asarray(p_extract)

    # force w et slope en np.array pour gérer le cas scalaire / vecteur
    w_arr = np.asarray(w)
    slope_arr = np.asarray(slope)

    A_list = []
    for i, p in enumerate(p_extract):
        # sélection w_i et slope_i en fonction de si w/slope sont scalaires ou vecteurs
        if w_arr.size > 1:
            w_i = w_arr[i]
        else:
            w_i = float(w_arr)

        if slope_arr.size > 1:
            slope_i = slope_arr[i]
        else:
            slope_i = float(slope_arr)

        _, att = compute_att(p, w_i, x, slope_i, plot=False)
        A_list.append(att)

    A = np.vstack(A_list)  # shape: (nb_sources, len(x))

    # Énergie totale ~ intégrale de la somme des |a_i(x)|^2
    energy_density = np.sum(A**2, axis=0)         # len(x)
    E_total = np.trapz(energy_density, x)         # scalaire

    gain_global = np.sqrt(E_target / E_total)
    return gain_global, A, E_total


# =========================
# Exemple d'utilisation
# =========================
plt.close('all')

# Exemple simple : un filtre
p = 0.3
w_single = 0.12
slope_single = 50
x = np.linspace(-1, 1, num=2000)

plt.figure()
compute_att(p, w_single, x, slope_single, plot=True)
plt.title("Un filtre (dB)")

# Banc de filtres
nb_sources = 7
p_extract = np.arange(-1, 1, 2/nb_sources) + 1/nb_sources

# ICI : w et slope peuvent être des tableaux OU des scalaires
# Exemple 1 : w et slope constants
# w = 0.5
# slope = 200

# Exemple 2 : w et slope différents par source
w = np.linspace(0.3, 0.7, nb_sources)         # largeur variable
slope = np.linspace(100, 300, nb_sources)     # pente variable

gain_global, A, E_total = compute_global_gain(p_extract, w, slope, x, E_target=1.0)

print("Énergie totale non normalisée :", E_total)
print("Gain global appliqué (lin)    :", gain_global)
print("Gain global (dB)              :", 20*np.log10(gain_global))

# Tracé du banc normalisé
plt.figure()
plt.ylim([-60, 3])

for i, p in enumerate(p_extract):
    att_norm = gain_global * A[i, :]                     # amplitude linéaire normalisée
    att_norm_dB = 20 * np.log10(np.maximum(att_norm, 1e-6))
    plt.plot(x, att_norm_dB, label=f"p={p:.2f}")

plt.grid('both')
plt.legend()
plt.title("Banc de filtres normalisé en énergie (w/slope par source)")
plt.show()
