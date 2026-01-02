# -*- coding: utf-8 -*-
"""
extractor.py - Extraction de sources fréquentielles

Ce module implémente l'extraction de sources à partir de la STFT :
1. Sélection du canal d'entrée le plus proche de l'angle estimé
2. Application du masque d'extraction (multiplication complexe)
3. Extraction de plusieurs sources indépendantes

@author: Damien
"""

from typing import List, Optional, Tuple

import numpy as np

from .mask_generator import generate_extraction_mask
from .re_model_light import get_speaker_unit_vectors


def get_channel_angles(layout: str) -> Tuple[np.ndarray, List[int]]:
    """
    Récupère les angles normalisés des canaux d'un layout.

    Parameters
    ----------
    layout : str
        Nom du layout ('stereo', '5.1', '7.1', etc.)

    Returns
    -------
    angles : np.ndarray, shape (n_channels,)
        Angles normalisés entre -1 et 1.
        Pour stéréo: L=-1, R=+1
        Pour multicanal: angle / 180 (donc -1 à +1 pour 360°)
    lfe_indices : List[int]
        Indices des canaux LFE.

    Notes
    -----
    Les angles sont normalisés selon la convention:
    - Stéréo: ±30° -> ±1 (60° span)
    - Multicanal: ±180° -> ±1 (360° span)
    """
    unit_vectors, _labels, lfe_indices = get_speaker_unit_vectors(layout)

    # Calculer les angles depuis les vecteurs unitaires
    # atan2(x, y) donne l'angle avec y=avant comme référence
    angles_rad = np.arctan2(unit_vectors[:, 0], unit_vectors[:, 1])
    angles_deg = np.degrees(angles_rad)

    # Normaliser selon le type de layout
    is_stereo = layout.lower().replace(" ", "").replace("-", "") == "stereo"

    if is_stereo:
        # Stéréo: normaliser par 60° (±30° -> ±1)
        # Mais on clip pour éviter les valeurs hors range
        angles_normalized = np.clip(angles_deg / 60.0, -1.0, 1.0)
    else:
        # Multicanal: normaliser par 180° (±180° -> ±1)
        angles_normalized = angles_deg / 180.0

    return angles_normalized.astype(np.float32), lfe_indices


def select_closest_channel(
    panning: np.ndarray,
    channel_angles: np.ndarray,
    lfe_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Sélectionne l'indice du canal le plus proche de l'angle estimé.

    Parameters
    ----------
    panning : np.ndarray
        Estimation de panning, shape (n_frames, n_freq).
        Valeurs entre -1 et 1.
    channel_angles : np.ndarray
        Angles normalisés des canaux, shape (n_channels,).
    lfe_indices : List[int], optional
        Indices des canaux LFE à exclure de la sélection.

    Returns
    -------
    selected_indices : np.ndarray, shape (n_frames, n_freq)
        Indices des canaux sélectionnés pour chaque bin.
    """
    n_channels = len(channel_angles)

    # Créer un masque pour exclure les canaux LFE
    valid_mask = np.ones(n_channels, dtype=bool)
    if lfe_indices:
        for idx in lfe_indices:
            valid_mask[idx] = False

    valid_indices = np.where(valid_mask)[0]
    valid_angles = channel_angles[valid_mask]

    if len(valid_indices) == 0:
        raise ValueError("Aucun canal valide (tous sont LFE)")

    # Calcul des distances pour chaque canal valide
    # Shape: (n_frames, n_freq, n_valid_channels)
    panning_expanded = panning[..., np.newaxis]  # (n_frames, n_freq, 1)
    distances = np.abs(panning_expanded - valid_angles)

    # Pour les angles proches de ±1 (wrap-around pour multicanal)
    # On prend le minimum avec la distance via le wrap
    distances_wrapped = 2.0 - distances
    distances = np.minimum(distances, distances_wrapped)

    # Trouver l'indice du canal le plus proche parmi les valides
    closest_valid_idx = np.argmin(distances, axis=-1)

    # Convertir en indices originaux
    selected_indices = valid_indices[closest_valid_idx]

    return selected_indices.astype(np.int32)


def apply_mask_to_stft(
    stft: np.ndarray,
    mask: np.ndarray,
    channel_indices: np.ndarray,
) -> np.ndarray:
    """
    Applique le masque à la STFT en sélectionnant le canal approprié.

    Parameters
    ----------
    stft : np.ndarray
        STFT multicanal, shape (n_frames, n_freq, n_channels).
    mask : np.ndarray
        Masque d'extraction, shape (n_frames, n_freq).
        Gains linéaires entre 0 et 1.
    channel_indices : np.ndarray
        Indices des canaux sélectionnés, shape (n_frames, n_freq).

    Returns
    -------
    extracted : np.ndarray
        STFT extraite, shape (n_frames, n_freq).
        Signal complexe.
    """
    n_frames, n_freq = mask.shape

    # Sélectionner le canal approprié pour chaque bin
    # Utiliser advanced indexing
    frame_idx = np.arange(n_frames)[:, np.newaxis]
    freq_idx = np.arange(n_freq)[np.newaxis, :]

    selected_stft = stft[frame_idx, freq_idx, channel_indices]

    # Appliquer le masque (multiplication complexe)
    extracted = selected_stft * mask

    return extracted.astype(np.complex64)


def extract_source(
    stft: np.ndarray,
    panning: np.ndarray,
    source_pan: float,
    width: float,
    slope: float,
    min_gain_db: float,
    attack_frames: float,
    release_frames: float,
    input_layout: str,
    apply_blur: bool = True,
    apply_smoothing: bool = True,
    power: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extrait une source à partir de la STFT multicanal.

    Combine toutes les étapes :
    1. Obtention des angles des canaux
    2. Sélection du canal le plus proche
    3. Génération du masque
    4. Application du masque

    Parameters
    ----------
    stft : np.ndarray
        STFT multicanal, shape (n_frames, n_freq, n_channels).
    panning : np.ndarray
        Estimation de panning, shape (n_frames, n_freq).
    source_pan : float
        Position de la source à extraire (-1 à 1).
    width : float
        Largeur du masque.
    slope : float
        Pente de la transition.
    min_gain_db : float
        Gain minimum en dB.
    attack_frames : float
        Temps d'attaque en frames.
    release_frames : float
        Temps de release en frames.
    input_layout : str
        Layout d'entrée pour la sélection de canal.
    apply_blur : bool
        Si True, appliquer le blur fréquentiel.
    apply_smoothing : bool
        Si True, appliquer le lissage temporel.
    power : np.ndarray, optional
        Puissance du signal pour le freeze.

    Returns
    -------
    extracted : np.ndarray
        STFT extraite, shape (n_frames, n_freq).
    """
    # 1. Obtenir les angles des canaux d'entrée
    channel_angles, lfe_indices = get_channel_angles(input_layout)

    # 2. Sélectionner le canal le plus proche du panning estimé
    channel_indices = select_closest_channel(panning, channel_angles, lfe_indices)

    # 3. Générer le masque d'extraction
    mask = generate_extraction_mask(
        panning=panning,
        source_pan=source_pan,
        width=width,
        slope=slope,
        min_gain_db=min_gain_db,
        attack_frames=attack_frames,
        release_frames=release_frames,
        apply_blur=apply_blur,
        apply_smoothing=apply_smoothing,
        power=power,
    )

    # 4. Appliquer le masque
    extracted = apply_mask_to_stft(stft, mask, channel_indices)

    return extracted


def extract_multiple_sources(
    stft: np.ndarray,
    panning: np.ndarray,
    source_params: List[dict],
    input_layout: str,
    apply_blur: bool = True,
    apply_smoothing: bool = True,
    power: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Extrait plusieurs sources indépendamment.

    Parameters
    ----------
    stft : np.ndarray
        STFT multicanal, shape (n_frames, n_freq, n_channels).
    panning : np.ndarray
        Estimation de panning, shape (n_frames, n_freq).
    source_params : List[dict]
        Liste de paramètres pour chaque source.
        Chaque dict contient : pan, width, slope, min_gain_db, attack_frames, release_frames.
        Si 'mute' est présent et égal à 1, la source est ignorée.
    input_layout : str
        Layout d'entrée.
    apply_blur : bool
        Si True, appliquer le blur fréquentiel.
    apply_smoothing : bool
        Si True, appliquer le lissage temporel.
    power : np.ndarray, optional
        Puissance du signal pour le freeze.

    Returns
    -------
    extracted_sources : List[np.ndarray]
        Liste des STFT extraites, une par source non-mutée.
        Shape de chaque élément : (n_frames, n_freq).
    """
    extracted_sources = []

    for params in source_params:
        # Vérifier si la source est mutée
        if params.get("mute", 0) == 1:
            continue

        extracted = extract_source(
            stft=stft,
            panning=panning,
            source_pan=params["pan"],
            width=params["width"],
            slope=params["slope"],
            min_gain_db=params["min_gain_db"],
            attack_frames=params["attack_frames"],
            release_frames=params["release_frames"],
            input_layout=input_layout,
            apply_blur=apply_blur,
            apply_smoothing=apply_smoothing,
            power=power,
        )

        extracted_sources.append(extracted)

    return extracted_sources


class SourceExtractor:
    """
    Classe pour l'extraction de sources avec configuration persistante.

    Attributes
    ----------
    input_layout : str
        Layout d'entrée.
    channel_angles : np.ndarray
        Angles normalisés des canaux.
    lfe_indices : List[int]
        Indices des canaux LFE.
    """

    def __init__(self, input_layout: str) -> None:
        """
        Initialise l'extracteur.

        Parameters
        ----------
        input_layout : str
            Layout d'entrée ('stereo', '5.1', etc.)
        """
        self.input_layout = input_layout
        self.channel_angles, self.lfe_indices = get_channel_angles(input_layout)

    def extract(
        self,
        stft: np.ndarray,
        panning: np.ndarray,
        source_pan: float,
        width: float,
        slope: float,
        min_gain_db: float,
        attack_frames: float,
        release_frames: float,
        apply_blur: bool = True,
        apply_smoothing: bool = True,
        power: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extrait une source.

        Parameters
        ----------
        stft : np.ndarray
            STFT multicanal.
        panning : np.ndarray
            Estimation de panning.
        source_pan : float
            Position de la source.
        width, slope, min_gain_db, attack_frames, release_frames : float
            Paramètres du masque.
        apply_blur, apply_smoothing : bool
            Options de lissage.
        power : np.ndarray, optional
            Puissance pour le freeze.

        Returns
        -------
        extracted : np.ndarray
            STFT extraite.
        """
        return extract_source(
            stft=stft,
            panning=panning,
            source_pan=source_pan,
            width=width,
            slope=slope,
            min_gain_db=min_gain_db,
            attack_frames=attack_frames,
            release_frames=release_frames,
            input_layout=self.input_layout,
            apply_blur=apply_blur,
            apply_smoothing=apply_smoothing,
            power=power,
        )

    def extract_batch(
        self,
        stft: np.ndarray,
        panning: np.ndarray,
        source_params: List[dict],
        apply_blur: bool = True,
        apply_smoothing: bool = True,
        power: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Extrait plusieurs sources.

        Parameters
        ----------
        stft : np.ndarray
            STFT multicanal.
        panning : np.ndarray
            Estimation de panning.
        source_params : List[dict]
            Paramètres de chaque source.
        apply_blur, apply_smoothing : bool
            Options de lissage.
        power : np.ndarray, optional
            Puissance pour le freeze.

        Returns
        -------
        extracted_sources : List[np.ndarray]
            Liste des STFT extraites.
        """
        return extract_multiple_sources(
            stft=stft,
            panning=panning,
            source_params=source_params,
            input_layout=self.input_layout,
            apply_blur=apply_blur,
            apply_smoothing=apply_smoothing,
            power=power,
        )

    def get_channel_info(self) -> dict:
        """
        Retourne les informations sur les canaux.

        Returns
        -------
        info : dict
            Informations sur le layout.
        """
        return {
            "layout": self.input_layout,
            "channel_angles": self.channel_angles,
            "lfe_indices": self.lfe_indices,
            "n_channels": len(self.channel_angles),
            "n_valid_channels": len(self.channel_angles) - len(self.lfe_indices),
        }
