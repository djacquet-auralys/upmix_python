# -*- coding: utf-8 -*-
"""
respatializer.py - Respatialisation des sources vers le layout de destination

Ce module implémente :
1. Application des gains de spatialisation par source
2. Application des délais (ms -> samples)
3. Sommation des contributions vers chaque HP de destination
4. Routage direct du LFE

@author: Damien
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ajouter le chemin parent pour importer multichannel_layouts
_parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from multichannel_layouts import _LAYOUTS, get_spk_coordinates


def get_output_layout_info(layout: str) -> Dict:
    """
    Récupère les informations sur le layout de sortie.

    Parameters
    ----------
    layout : str
        Nom du layout ('5.1', '7.1', etc.)

    Returns
    -------
    info : dict
        Informations sur le layout :
        - n_channels : nombre total de canaux
        - n_speakers : nombre de HP (sans LFE)
        - lfe_indices : indices des canaux LFE
        - speaker_indices : indices des canaux non-LFE
        - labels : labels des canaux
    """
    layout_data = get_spk_coordinates(layout, radius=1.0)

    labels: List[str] = layout_data["labels"]
    n_channels = len(labels)

    # Identifier les indices LFE et non-LFE
    lfe_indices = [i for i, lbl in enumerate(labels) if "LFE" in lbl.upper()]
    speaker_indices = [i for i in range(n_channels) if i not in lfe_indices]

    return {
        "n_channels": n_channels,
        "n_speakers": len(speaker_indices),
        "n_lfe": len(lfe_indices),
        "lfe_indices": lfe_indices,
        "speaker_indices": speaker_indices,
        "labels": labels,
    }


def ms_to_samples(delay_ms: float, sample_rate: float) -> int:
    """
    Convertit un délai en millisecondes vers un nombre de samples.

    Parameters
    ----------
    delay_ms : float
        Délai en millisecondes.
    sample_rate : float
        Fréquence d'échantillonnage en Hz.

    Returns
    -------
    delay_samples : int
        Délai en samples (arrondi à l'entier le plus proche).
    """
    return int(round(delay_ms * sample_rate / 1000.0))


def apply_delay(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """
    Applique un délai à un signal.

    Parameters
    ----------
    signal : np.ndarray
        Signal d'entrée, shape (n_samples,).
    delay_samples : int
        Délai en samples (entier positif).

    Returns
    -------
    delayed : np.ndarray
        Signal retardé, même shape que l'entrée.
        Les premiers échantillons sont à zéro.
    """
    if delay_samples <= 0:
        return signal.copy()

    delayed = np.zeros_like(signal)
    if delay_samples < len(signal):
        delayed[delay_samples:] = signal[:-delay_samples]

    return delayed


def apply_gain_and_delay(
    source: np.ndarray,
    gains: np.ndarray,
    delays_ms: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    """
    Applique gains et délais d'une source vers les canaux de sortie.

    Parameters
    ----------
    source : np.ndarray
        Signal source, shape (n_samples,).
    gains : np.ndarray
        Gains vers chaque canal de sortie, shape (n_output_channels,).
        Gains linéaires (pas en dB).
    delays_ms : np.ndarray
        Délais vers chaque canal en ms, shape (n_output_channels,).
    sample_rate : float
        Fréquence d'échantillonnage.

    Returns
    -------
    output : np.ndarray
        Signaux de sortie, shape (n_samples, n_output_channels).
    """
    n_samples = len(source)
    n_outputs = len(gains)

    output = np.zeros((n_samples, n_outputs), dtype=np.float32)

    for i in range(n_outputs):
        if gains[i] > 0:
            delay_samples = ms_to_samples(float(delays_ms[i]), sample_rate)
            delayed = apply_delay(source, delay_samples)
            output[:, i] = delayed * gains[i]

    return output


def spatialize_sources(
    sources: List[np.ndarray],
    gains_list: List[np.ndarray],
    delays_list: List[np.ndarray],
    sample_rate: float,
    n_output_channels: int,
) -> np.ndarray:
    """
    Spatialise plusieurs sources vers les canaux de sortie.

    Parameters
    ----------
    sources : List[np.ndarray]
        Liste des signaux sources, chacun de shape (n_samples,).
    gains_list : List[np.ndarray]
        Gains par source, chaque array de shape (n_output_channels,).
    delays_list : List[np.ndarray]
        Délais par source (en ms), chaque array de shape (n_output_channels,).
    sample_rate : float
        Fréquence d'échantillonnage.
    n_output_channels : int
        Nombre de canaux de sortie.

    Returns
    -------
    output : np.ndarray
        Signal de sortie, shape (n_samples, n_output_channels).
    """
    if len(sources) == 0:
        raise ValueError("Aucune source à spatialiser")

    n_samples = len(sources[0])
    output = np.zeros((n_samples, n_output_channels), dtype=np.float32)

    for source, gains, delays in zip(sources, gains_list, delays_list):
        contribution = apply_gain_and_delay(source, gains, delays, sample_rate)
        output += contribution

    return output


def add_lfe_to_output(
    output: np.ndarray,
    lfe_signal: np.ndarray,
    lfe_output_index: int,
) -> np.ndarray:
    """
    Ajoute le signal LFE au canal de sortie approprié.

    Parameters
    ----------
    output : np.ndarray
        Signal de sortie multicanal, shape (n_samples, n_channels).
    lfe_signal : np.ndarray
        Signal LFE, shape (n_samples,).
    lfe_output_index : int
        Index du canal LFE dans la sortie.

    Returns
    -------
    output : np.ndarray
        Signal de sortie avec LFE ajouté.
    """
    output = output.copy()
    n_samples = min(len(lfe_signal), output.shape[0])
    output[:n_samples, lfe_output_index] = lfe_signal[:n_samples]
    return output


def add_lf_mono1_to_sources(
    sources: List[np.ndarray],
    lf_mono1: np.ndarray,
    lf_gains: List[float],
    latency_samples: int,
) -> List[np.ndarray]:
    """
    Ajoute le signal LF_mono1 retardé à chaque source.

    Parameters
    ----------
    sources : List[np.ndarray]
        Liste des signaux sources extraits.
    lf_mono1 : np.ndarray
        Signal basse fréquence mono.
    lf_gains : List[float]
        Gain LF pour chaque source (linéaire).
    latency_samples : int
        Latence STFT en samples (pour aligner les signaux).

    Returns
    -------
    sources_with_lf : List[np.ndarray]
        Sources avec LF_mono1 ajouté.
    """
    # Retarder LF_mono1 pour aligner avec la latence STFT
    lf_delayed = apply_delay(lf_mono1, latency_samples)

    sources_with_lf = []
    for source, lf_gain in zip(sources, lf_gains):
        # S'assurer que les longueurs correspondent
        n_samples = min(len(source), len(lf_delayed))
        result = source.copy()
        result[:n_samples] += lf_delayed[:n_samples] * lf_gain
        sources_with_lf.append(result)

    return sources_with_lf


def parse_source_params(
    upmix_params: Dict,
    n_sources: int,
    n_output_speakers: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[bool]]:
    """
    Parse les paramètres de spatialisation depuis le JSON.

    Parameters
    ----------
    upmix_params : Dict
        Paramètres upmix du JSON.
    n_sources : int
        Nombre de sources à parser.
    n_output_speakers : int
        Nombre de HP de sortie (sans LFE).

    Returns
    -------
    gains_list : List[np.ndarray]
        Gains par source, shape (n_output_speakers,) chacun.
    delays_list : List[np.ndarray]
        Délais par source en ms.
    lf_gains : List[float]
        Gains LF par source.
    mutes : List[bool]
        État mute par source.
    """
    gains_list = []
    delays_list = []
    lf_gains = []
    mutes = []

    for i in range(1, n_sources + 1):
        # Gains vers chaque HP de sortie
        gains_key = f"gains{i}"
        if gains_key in upmix_params:
            gains = np.array(upmix_params[gains_key], dtype=np.float32)
        else:
            # Gains par défaut (unitaires)
            gains = np.ones(n_output_speakers, dtype=np.float32)

        # Délais vers chaque HP
        delays_key = f"delays{i}"
        if delays_key in upmix_params:
            delays = np.array(upmix_params[delays_key], dtype=np.float32)
        else:
            # Pas de délai par défaut
            delays = np.zeros(n_output_speakers, dtype=np.float32)

        # Gain LF
        lf_gain_key = f"LF_gain{i}"
        lf_gain = upmix_params.get(lf_gain_key, 1.0)

        # Mute
        mute_key = f"mute{i}"
        mute = upmix_params.get(mute_key, 0) == 1

        gains_list.append(gains)
        delays_list.append(delays)
        lf_gains.append(float(lf_gain))
        mutes.append(mute)

    return gains_list, delays_list, lf_gains, mutes


class Respatializer:
    """
    Classe pour la respatialisation des sources.

    Attributes
    ----------
    output_layout : str
        Layout de sortie.
    sample_rate : float
        Fréquence d'échantillonnage.
    layout_info : dict
        Informations sur le layout de sortie.
    """

    def __init__(
        self,
        output_layout: str,
        sample_rate: float = 48000.0,
    ) -> None:
        """
        Initialise le respatialisateur.

        Parameters
        ----------
        output_layout : str
            Layout de sortie ('5.1', '7.1', etc.)
        sample_rate : float
            Fréquence d'échantillonnage.
        """
        self.output_layout = output_layout
        self.sample_rate = sample_rate
        self.layout_info = get_output_layout_info(output_layout)

    @property
    def n_output_channels(self) -> int:
        """Nombre total de canaux de sortie."""
        return self.layout_info["n_channels"]

    @property
    def n_speakers(self) -> int:
        """Nombre de HP (sans LFE)."""
        return self.layout_info["n_speakers"]

    @property
    def lfe_indices(self) -> List[int]:
        """Indices des canaux LFE."""
        return self.layout_info["lfe_indices"]

    def spatialize(
        self,
        sources: List[np.ndarray],
        gains_list: List[np.ndarray],
        delays_list: List[np.ndarray],
        lfe_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Spatialise les sources vers le layout de sortie.

        Parameters
        ----------
        sources : List[np.ndarray]
            Signaux sources, chacun de shape (n_samples,).
        gains_list : List[np.ndarray]
            Gains par source vers chaque HP.
        delays_list : List[np.ndarray]
            Délais par source vers chaque HP (en ms).
        lfe_signal : np.ndarray, optional
            Signal LFE à router directement.

        Returns
        -------
        output : np.ndarray
            Signal de sortie, shape (n_samples, n_output_channels).
        """
        if len(sources) == 0:
            # Pas de sources, retourner silence avec éventuellement le LFE
            if lfe_signal is not None:
                n_samples = len(lfe_signal)
                output = np.zeros((n_samples, self.n_output_channels), dtype=np.float32)
                if self.lfe_indices:
                    output = add_lfe_to_output(output, lfe_signal, self.lfe_indices[0])
                return output
            else:
                raise ValueError("Aucune source et pas de LFE")

        # Spatialiser les sources vers les HP (sans LFE)
        output = spatialize_sources(
            sources=sources,
            gains_list=gains_list,
            delays_list=delays_list,
            sample_rate=self.sample_rate,
            n_output_channels=self.n_output_channels,
        )

        # Ajouter le LFE si fourni
        if lfe_signal is not None and self.lfe_indices:
            output = add_lfe_to_output(output, lfe_signal, self.lfe_indices[0])

        return output

    def spatialize_from_params(
        self,
        sources: List[np.ndarray],
        upmix_params: Dict,
        lfe_signal: Optional[np.ndarray] = None,
        lf_mono1: Optional[np.ndarray] = None,
        latency_samples: int = 0,
    ) -> np.ndarray:
        """
        Spatialise les sources en utilisant les paramètres JSON.

        Parameters
        ----------
        sources : List[np.ndarray]
            Signaux sources extraits.
        upmix_params : Dict
            Paramètres upmix du JSON.
        lfe_signal : np.ndarray, optional
            Signal LFE.
        lf_mono1 : np.ndarray, optional
            Signal basse fréquence mono à ajouter.
        latency_samples : int
            Latence STFT en samples.

        Returns
        -------
        output : np.ndarray
            Signal de sortie.
        """
        n_sources = len(sources)

        # Parser les paramètres
        gains_list, delays_list, lf_gains, mutes = parse_source_params(
            upmix_params, n_sources, self.n_speakers
        )

        # Filtrer les sources non-mutées
        active_sources = []
        active_gains = []
        active_delays = []
        active_lf_gains = []

        for src, gain, delay, lf_g, mute in zip(
            sources, gains_list, delays_list, lf_gains, mutes
        ):
            if not mute:
                active_sources.append(src)
                active_gains.append(gain)
                active_delays.append(delay)
                active_lf_gains.append(lf_g)

        # Ajouter LF_mono1 si fourni
        if lf_mono1 is not None and len(active_sources) > 0:
            active_sources = add_lf_mono1_to_sources(
                active_sources, lf_mono1, active_lf_gains, latency_samples
            )

        # Spatialiser
        return self.spatialize(
            sources=active_sources,
            gains_list=active_gains,
            delays_list=active_delays,
            lfe_signal=lfe_signal,
        )


def compute_default_gains(
    source_pan: float,
    output_layout: str,
) -> np.ndarray:
    """
    Calcule des gains par défaut basés sur le panning de la source.

    Version simplifiée (pas TDAP complet) - distribue le signal
    entre les deux HP les plus proches.

    Parameters
    ----------
    source_pan : float
        Position de la source (-1 à 1).
    output_layout : str
        Layout de sortie.

    Returns
    -------
    gains : np.ndarray
        Gains vers chaque HP (sans LFE).
    """
    layout_info = get_output_layout_info(output_layout)
    layout_data = get_spk_coordinates(output_layout, radius=1.0)

    speaker_indices = layout_info["speaker_indices"]
    n_speakers = len(speaker_indices)

    # Récupérer les angles des HP (sans LFE)
    azimuth_deg = layout_data["azimuth_deg"]
    speaker_angles = azimuth_deg[speaker_indices]

    # Convertir le pan en angle (approximatif)
    # Pour stéréo: -1 = -30°, 0 = 0°, 1 = 30°
    # Pour multicanal: -1 = -180°, 1 = 180°
    if output_layout.lower() == "stereo":
        source_angle = source_pan * 30.0
    else:
        source_angle = source_pan * 180.0

    # Calculer les distances angulaires
    distances = np.abs(speaker_angles - source_angle)
    # Gérer le wrap-around
    distances = np.minimum(distances, 360.0 - distances)

    # Gains inversement proportionnels à la distance
    # Utiliser une loi de puissance constante
    max_dist = 180.0  # Distance maximale
    weights = np.maximum(0, 1.0 - distances / max_dist)
    weights = weights**2  # Loi de puissance

    # Normaliser pour préserver l'énergie
    if np.sum(weights) > 0:
        weights = weights / np.sqrt(np.sum(weights**2))
    else:
        weights = np.zeros(n_speakers, dtype=np.float32)
        weights[0] = 1.0  # Fallback

    return weights.astype(np.float32)


def get_available_output_layouts() -> List[str]:
    """
    Retourne la liste des layouts de sortie disponibles.

    Returns
    -------
    layouts : List[str]
        Noms des layouts disponibles.
    """
    return list(_LAYOUTS.keys())
