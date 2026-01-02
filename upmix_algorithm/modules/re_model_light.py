# -*- coding: utf-8 -*-
"""
re_model_light.py - Module léger pour le calcul de vecteur d'énergie (RE)

Version simplifiée de re_model.py optimisée pour l'estimation de panning :
- Position d'écoute à l'origine (0, 0)
- Pas d'atténuation angulaire (directivité ignorée)
- Pas d'atténuation par distance
- Pas de délais
- Calcul en linéaire (pas de conversion dB)
- Rayon unitaire (seule la direction compte)

@author: Damien
"""

import os

# Import des layouts
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ajouter le chemin parent pour importer multichannel_layouts
_parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from multichannel_layouts import _LAYOUTS, get_spk_coordinates


def get_speaker_unit_vectors(layout: str) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Calcule les vecteurs unitaires des haut-parleurs d'un layout.

    Parameters
    ----------
    layout : str
        Nom du layout ('stereo', '5.1', '7.1', etc.)

    Returns
    -------
    unit_vectors : np.ndarray, shape (n_channels, 2)
        Vecteurs unitaires (x, y) pointant vers chaque haut-parleur.
        Coordonnées: +x = droite, +y = avant.
    labels : List[str]
        Labels des canaux.
    lfe_indices : List[int]
        Indices des canaux LFE dans le layout.

    Notes
    -----
    Convention d'angle:
    - 0° = avant (+y)
    - Angles positifs = droite (sens horaire vu de dessus)
    - sin(azimut) = x, cos(azimut) = y
    """
    # Récupérer les informations du layout (rayon=1 car on veut des vecteurs unitaires)
    layout_data = get_spk_coordinates(layout, radius=1.0)

    labels: List[str] = layout_data["labels"]
    azimuth_deg: np.ndarray = layout_data["azimuth_deg"]

    # Convertir azimuts en vecteurs unitaires
    # Convention: 0° = avant (+y), positif = droite
    # x = sin(az), y = cos(az)
    az_rad = np.deg2rad(azimuth_deg)
    unit_vectors = np.column_stack((np.sin(az_rad), np.cos(az_rad)))

    # Identifier les indices LFE
    lfe_indices = [i for i, lbl in enumerate(labels) if "LFE" in lbl.upper()]

    return unit_vectors, labels, lfe_indices


def compute_energy_vector(
    gains: np.ndarray,
    unit_vectors: np.ndarray,
    lfe_indices: Optional[List[int]] = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Calcule le vecteur d'énergie à partir des gains linéaires.

    Version optimisée sans conversion dB, position centrale, pas de délais.

    Parameters
    ----------
    gains : np.ndarray
        Gains linéaires. Peut être:
        - 1D: shape (n_channels,) pour un seul calcul
        - 2D: shape (n_points, n_channels) pour calcul vectorisé
        - 3D: shape (n_frames, n_freq, n_channels) pour STFT
    unit_vectors : np.ndarray, shape (n_channels, 2)
        Vecteurs unitaires des haut-parleurs.
    lfe_indices : List[int], optional
        Indices des canaux LFE à exclure du calcul.
    epsilon : float
        Valeur minimale pour éviter division par zéro.

    Returns
    -------
    re : np.ndarray
        Vecteur d'énergie normalisé (x, y).
        - Pour gains 1D: shape (2,)
        - Pour gains 2D: shape (n_points, 2)
        - Pour gains 3D: shape (n_frames, n_freq, 2)

    Notes
    -----
    Formule: RE = sum(g_i² * v_i) / sum(g_i²)
    où g_i sont les gains linéaires et v_i les vecteurs unitaires.
    """
    gains = np.asarray(gains, dtype=np.float32)

    # Masquer les canaux LFE si spécifiés
    if lfe_indices:
        mask = np.ones(gains.shape[-1], dtype=bool)
        mask[lfe_indices] = False
        gains = gains[..., mask]
        unit_vectors = unit_vectors[mask]

    # Gains au carré
    gains_squared: np.ndarray = gains**2

    # Calcul du vecteur d'énergie
    # Gère les différentes dimensions d'entrée
    if gains.ndim == 1:
        # (n_channels,) -> (2,)
        total_energy = np.sum(gains_squared) + epsilon
        re = np.sum(gains_squared[:, np.newaxis] * unit_vectors, axis=0) / total_energy

    elif gains.ndim == 2:
        # (n_points, n_channels) -> (n_points, 2)
        total_energy = np.sum(gains_squared, axis=-1, keepdims=True) + epsilon
        # gains_squared: (n_points, n_channels)
        # unit_vectors: (n_channels, 2)
        # Résultat: (n_points, 2)
        weighted = gains_squared[..., :, np.newaxis] * unit_vectors[np.newaxis, :, :]
        re = np.sum(weighted, axis=1) / total_energy

    elif gains.ndim == 3:
        # (n_frames, n_freq, n_channels) -> (n_frames, n_freq, 2)
        total_energy = np.sum(gains_squared, axis=-1, keepdims=True) + epsilon
        # gains_squared: (n_frames, n_freq, n_channels)
        # unit_vectors: (n_channels, 2)
        # Résultat: (n_frames, n_freq, 2)
        weighted = gains_squared[..., np.newaxis] * unit_vectors  # broadcasting
        re = np.sum(weighted, axis=2) / total_energy

    else:
        raise ValueError(f"gains doit avoir 1, 2 ou 3 dimensions, reçu {gains.ndim}")

    return re.astype(np.float32)


def energy_vector_to_angle(
    re: np.ndarray, normalize_range: float = 360.0
) -> np.ndarray:
    """
    Convertit un vecteur d'énergie en angle normalisé.

    Parameters
    ----------
    re : np.ndarray
        Vecteur d'énergie (x, y). Dernière dimension = 2.
    normalize_range : float
        Plage de normalisation en degrés:
        - 60.0 pour stéréo (±30°)
        - 360.0 pour multicanal (cercle complet)

    Returns
    -------
    pan : np.ndarray
        Angle normalisé dans [-1, 1].
        - -1 = gauche (pour stéréo) ou arrière-gauche (pour multicanal)
        - 0 = centre/avant
        - +1 = droite (pour stéréo) ou arrière-droite (pour multicanal)

    Notes
    -----
    L'angle est calculé avec atan2(x, y) pour avoir 0° vers l'avant (+y).
    Convention: angles positifs = droite.
    """
    # atan2(x, y) donne l'angle avec 0° vers +y (avant)
    # Angles positifs = droite
    angle_rad = np.arctan2(re[..., 0], re[..., 1])
    angle_deg = np.rad2deg(angle_rad)

    # Normalisation dans [-1, 1]
    # Pour stéréo (60°): ±30° -> ±1
    # Pour multicanal (360°): ±180° -> ±0.5 (on utilise [-1, 1] pour ±180°)
    pan = angle_deg / (normalize_range / 2.0)

    # Clipping pour garantir [-1, 1]
    pan = np.clip(pan, -1.0, 1.0)

    return pan.astype(np.float32)


def estimate_panning(
    stft_magnitudes: np.ndarray, layout: str, epsilon: float = 1e-12
) -> np.ndarray:
    """
    Estime le panning à partir des magnitudes STFT.

    C'est la fonction principale de ce module, combinant toutes les étapes.

    Parameters
    ----------
    stft_magnitudes : np.ndarray
        Magnitudes |STFT| en linéaire.
        Shape: (n_frames, n_freq, n_channels)
    layout : str
        Nom du layout d'entrée ('stereo', '5.1', etc.)
    epsilon : float
        Valeur minimale pour éviter division par zéro.

    Returns
    -------
    panning : np.ndarray, shape (n_frames, n_freq)
        Estimation de panning normalisée dans [-1, 1].
        - -1 = gauche extrême
        - 0 = centre
        - +1 = droite extrême

    Notes
    -----
    - Les canaux LFE sont automatiquement exclus du calcul.
    - Pour stéréo (layout "stereo"), normalisation par 60° (±30°).
    - Pour multicanal, normalisation par 360° (cercle complet).

    Examples
    --------
    >>> stft_mag = np.abs(stft_complex)  # (100, 65, 2) pour stéréo
    >>> pan = estimate_panning(stft_mag, "stereo")
    >>> print(pan.shape)  # (100, 65)
    >>> print(pan.min(), pan.max())  # entre -1 et 1
    """
    # Récupérer les vecteurs unitaires et identifier les LFE
    unit_vectors, _labels, lfe_indices = get_speaker_unit_vectors(layout)

    # Calculer le vecteur d'énergie
    re = compute_energy_vector(
        gains=stft_magnitudes,
        unit_vectors=unit_vectors,
        lfe_indices=lfe_indices,
        epsilon=epsilon,
    )

    # Déterminer la plage de normalisation
    # Stéréo = 60° (±30°), multicanal = 360°
    is_stereo = layout.lower().replace(" ", "").replace("-", "") == "stereo"
    normalize_range = 60.0 if is_stereo else 360.0

    # Convertir en angle normalisé
    panning = energy_vector_to_angle(re, normalize_range)

    return panning


def get_energy_vector_magnitude(re: np.ndarray) -> np.ndarray:
    """
    Calcule la magnitude (norme) du vecteur d'énergie.

    Utilisé pour estimer la largeur de la source.

    Parameters
    ----------
    re : np.ndarray
        Vecteur d'énergie (x, y). Dernière dimension = 2.

    Returns
    -------
    magnitude : np.ndarray
        Norme du vecteur d'énergie, entre 0 et 1.
        - 1 = source ponctuelle (toute l'énergie sur un seul HP)
        - 0 = source diffuse (énergie égale sur tous les HP)
    """
    return np.sqrt(re[..., 0] ** 2 + re[..., 1] ** 2).astype(np.float32)


def estimate_source_width(
    stft_magnitudes: np.ndarray, layout: str, epsilon: float = 1e-12
) -> np.ndarray:
    """
    Estime la largeur de source à partir des magnitudes STFT.

    Parameters
    ----------
    stft_magnitudes : np.ndarray
        Magnitudes |STFT| en linéaire.
        Shape: (n_frames, n_freq, n_channels)
    layout : str
        Nom du layout d'entrée.
    epsilon : float
        Valeur minimale pour éviter division par zéro.

    Returns
    -------
    width : np.ndarray, shape (n_frames, n_freq)
        Estimation de largeur en degrés.
        Formule: width = 5/8 * 180/π * 2 * arccos(|RE|)
    """
    unit_vectors, _labels, lfe_indices = get_speaker_unit_vectors(layout)

    re = compute_energy_vector(
        gains=stft_magnitudes,
        unit_vectors=unit_vectors,
        lfe_indices=lfe_indices,
        epsilon=epsilon,
    )

    re_magnitude = get_energy_vector_magnitude(re)

    # Clipping pour éviter erreurs numériques avec arccos
    re_magnitude = np.clip(re_magnitude, 0.0, 1.0)

    # Formule de largeur (comme dans re_model.py original)
    width = 5.0 / 8.0 * 180.0 / np.pi * 2.0 * np.arccos(re_magnitude)

    return width.astype(np.float32)


# --- Fonctions utilitaires pour les layouts ---


def get_available_layouts() -> List[str]:
    """Retourne la liste des layouts disponibles."""
    return list(_LAYOUTS.keys())


def get_layout_info(layout: str) -> Dict[str, Any]:
    """
    Retourne les informations d'un layout.

    Parameters
    ----------
    layout : str
        Nom du layout.

    Returns
    -------
    info : dict
        Dictionnaire avec 'labels', 'azimuth_deg', 'n_channels',
        'n_lfe', 'is_stereo'.
    """
    _unit_vectors, labels, lfe_indices = get_speaker_unit_vectors(layout)
    layout_data = get_spk_coordinates(layout, radius=1.0)

    return {
        "labels": labels,
        "azimuth_deg": layout_data["azimuth_deg"].tolist(),
        "n_channels": len(labels),
        "n_lfe": len(lfe_indices),
        "n_fullrange": len(labels) - len(lfe_indices),
        "is_stereo": layout.lower().replace(" ", "").replace("-", "") == "stereo",
        "lfe_indices": lfe_indices,
    }
