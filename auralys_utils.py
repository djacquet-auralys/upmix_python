# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:11:45 2023

@author: DamienJacquet
"""
import numpy as np

def a2dB(x):
    """Conversion linéaire -> dB"""
    return 20 * np.log10(x)


def dB2a(x):
    """Conversion dB -> linéaire"""
    return 10 ** (x / 20)


def getAngle(b, a, c=None):
    """
    Compute the angle at point(s) b between vectors ba and bc using atan2.

    Parameters
    ----------
    b : np.ndarray, shape (2,) or (N, 2)
        Vertex point(s) of the angle.
    a : np.ndarray, shape compatible with b
        First point(s) defining direction ba.
    c : np.ndarray or None, shape compatible with b
        Second point(s) defining direction bc.
        If None, defaults to b + [1, 0] (i.e., pointing along +X).

    Returns
    -------
    theta : np.ndarray
        Angle(s) in radians between ba and bc.
    """
    b = np.atleast_2d(b)
    a = np.atleast_2d(a)
    
    if c is None:
        c = b + np.array([1.0, 0.0])
    else:
        c = np.atleast_2d(c)

    ba = a - b
    bc = c - b

    theta = np.arctan2(ba[:, 1], ba[:, 0]) - np.arctan2(bc[:, 1], bc[:, 0])
    return theta.squeeze()


def getDist(a, b):
    """
    Compute Euclidean distance between points a and b.
    Supports broadcasting for arrays of shape (N, 2).
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return np.linalg.norm(b - a, axis=1 if a.shape[0] > 1 else None)


def cardioid(theta):
    """Linear cardioid law normalized"""
    return (1 + np.cos(theta)) / 2


def custom_cardioid(theta, aperture):
    """
    Cardioid law for a loudspeaker with given aperture at -6dB.
    Both aperture and theta are in radians.
    """
    card_exp = -6 / (20 * np.log10((1 + np.cos(aperture / 2)) / 2))
    return cardioid(theta) ** card_exp


def inv_square_att_law_dB(new_dist, dref, sref):
    """
    Inverse square law to compute attenuation in air.
    
    sref = reference level at distance dref
    new_dist = target distance
    """
    return sref + 20 * np.log10(dref / new_dist)


def coherent_sig_sum(A, phi):
    """
    Returns the amplitude and phase of a sum of sinusoids of same frequency
    with different amplitudes and phases.

    Parameters
    ----------
    A : np.ndarray
        Amplitudes
    phi : np.ndarray
        Phases in radians

    Returns
    -------
    G : float
        Resulting amplitude
    psy : float
        Resulting phase in radians
    """
    G = np.sqrt((np.sum(A * np.cos(phi)))**2 + (np.sum(A * np.sin(phi)))**2)
    psy = np.arctan2(np.sum(A * np.sin(phi)), np.sum(A * np.cos(phi)))
    return G, psy


def phase_f_delay(f, delay):
    """
    Calculate phase in radians for a given time delay and frequency.

    Parameters
    ----------
    f : float or np.ndarray
        Frequency in Hz
    delay : float or np.ndarray
        Time delay in seconds

    Returns
    -------
    phi : float or np.ndarray
        Phase in radians
    """
    return (2 * np.pi * f * delay) % (2 * np.pi)


def cart2pol(point):
    """
    Convert Cartesian coordinates to polar.

    Parameters
    ----------
    point : np.ndarray, shape (2,) or (N, 2)

    Returns
    -------
    rho_phi : np.ndarray, shape (N, 2)
        [[rho0, phi0], ..., [rhoN, phiN]]
    """
    point = np.atleast_2d(point)
    x = point[:, 0]
    y = point[:, 1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.stack((rho, phi), axis=-1).squeeze()


def pol2cart(rho, phi):
    """
    Convert polar coordinates to Cartesian.

    Parameters
    ----------
    rho : float or np.ndarray of shape (N,)
        Radii
    phi : float or np.ndarray of shape (N,)
        Angles in radians

    Returns
    -------
    coords : np.ndarray, shape (N, 2)
        [[x0, y0], ..., [xN, yN]]
    """
    rho = np.asarray(rho)
    phi = np.asarray(phi)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.stack((x, y), axis=-1).squeeze()

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



def map_affine(x, in_min, in_max, out_min, out_max):
    """
    Effectue un mappage affine de x depuis l'intervalle [in_min, in_max]
    vers l'intervalle [out_min, out_max].

    Paramètres :
        x : valeur ou tableau de valeurs à mapper
        in_min, in_max : bornes de l'intervalle d'entrée
        out_min, out_max : bornes de l'intervalle de sortie

    Retour :
        Valeur(s) mappée(s) dans l'intervalle de sortie
    """
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

