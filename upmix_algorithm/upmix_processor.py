# -*- coding: utf-8 -*-
"""
upmix_processor.py - Processeur principal d'intégration pour l'algorithme upmix

Ce module orchestre toutes les étapes de l'upmix :
1. Crossovers : Séparation HF/BF et création de LF_mono1
2. LFE : Détection/création du canal LFE
3. Upmix fréquentiel : STFT, panning, extraction de sources
4. Ajout LF_mono1 : Sommation des basses aux sources extraites
5. Respatialisation : Application gains/délais vers layout de sortie

@author: Damien
"""

import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.io import wavfile

# Ajouter le chemin pour multichannel_layouts
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from multichannel_layouts import get_spk_coordinates

from .modules import (
    Crossover,
    LFEProcessor,
    Respatializer,
    SourceExtractor,
    STFTProcessor,
    add_lf_mono1_to_sources,
    compute_latency,
    detect_lfe_channels,
    estimate_panning,
)


def get_layout_labels(layout: str) -> List[str]:
    """Récupère les labels des canaux pour un layout donné."""
    layout_data = get_spk_coordinates(layout, radius=1.0)
    return layout_data["labels"]


class UpmixProcessor:
    """
    Classe principale pour le traitement upmix.

    Orchestre toutes les étapes de conversion d'un signal audio
    depuis un layout source vers un layout de destination.

    Attributes
    ----------
    input_layout : str
        Layout d'entrée ('stereo', '5.1', etc.)
    output_layout : str
        Layout de sortie ('5.1', '7.1', etc.)
    sample_rate : float
        Fréquence d'échantillonnage.
    params : dict
        Paramètres de configuration.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        input_layout: str = "stereo",
        output_layout: str = "5.1",
        sample_rate: float = 48000.0,
    ) -> None:
        """
        Initialise le processeur upmix.

        Parameters
        ----------
        params : dict
            Paramètres de configuration (structure JSON).
        input_layout : str
            Layout d'entrée.
        output_layout : str
            Layout de sortie.
        sample_rate : float
            Fréquence d'échantillonnage.
        """
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.sample_rate = sample_rate
        self.params = params

        # Labels des canaux d'entrée
        self.input_labels = get_layout_labels(input_layout)
        self.n_input_channels = len(self.input_labels)
        self.input_lfe_indices = detect_lfe_channels(self.input_labels)

        # Paramètres globaux
        self.f_xover1 = params.get("F_xover1", 150.0)
        self.f_lfe = params.get("F_LFE", 120.0)
        self.nfft = params.get("nfft", 128)
        self.overlap = params.get("overlap", 0.25)
        self.max_sources = params.get("max_sources", 11)

        # Paramètres upmix
        self.upmix_params = params.get("upmix_params", {})

        # Initialisation des sous-modules
        self._init_modules()

    def _init_modules(self) -> None:
        """Initialise les sous-modules de traitement."""
        # Crossover
        self.crossover = Crossover(
            freq=self.f_xover1,
            fs=self.sample_rate,
            n_channels=self.n_input_channels,
            lfe_channel_indices=self.input_lfe_indices,
            q=0.707,
        )

        # LFE Processor
        self.lfe_processor = LFEProcessor(
            labels=self.input_labels,
            f_lfe=self.f_lfe,
            fs=self.sample_rate,
        )

        # STFT Processor
        self.stft_processor = STFTProcessor(
            nfft=self.nfft,
            overlap=self.overlap,
        )

        # Extractor
        self.extractor = SourceExtractor(self.input_layout)

        # Respatializer
        self.respatializer = Respatializer(
            output_layout=self.output_layout,
            sample_rate=self.sample_rate,
        )

        # Latence STFT
        self.latency_samples = compute_latency(self.nfft, self.overlap)

    def process(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Traite le signal audio complet.

        Parameters
        ----------
        input_signal : np.ndarray
            Signal d'entrée, shape (n_samples, n_channels).

        Returns
        -------
        output_signal : np.ndarray
            Signal de sortie, shape (n_samples, n_output_channels).
        """
        # Validation de l'entrée
        if input_signal.ndim == 1:
            input_signal = input_signal[:, np.newaxis]

        input_signal = input_signal.astype(np.float32)

        # Calcul du RMS d'entrée pour normalisation (Spec ligne 610)
        input_rms = np.sqrt(np.mean(input_signal**2))

        # === ÉTAPE 1 : CROSSOVERS ===
        hf_signals, lf_mono1 = self._step1_crossovers(input_signal)

        # === ÉTAPE 2 : CRÉATION/DÉTECTION LFE ===
        lfe_signal = self._step2_lfe(input_signal)

        # === ÉTAPE 3 : UPMIX FRÉQUENTIEL ===
        extracted_sources = self._step3_upmix_frequentiel(hf_signals)

        # === ÉTAPE 4 : AJOUT LF_MONO1 ===
        sources_with_lf = self._step4_add_lf_mono1(extracted_sources, lf_mono1)

        # === ÉTAPE 5 : RESPATIALISATION ===
        output_signal = self._step5_respatialization(sources_with_lf, lfe_signal)

        # Normalisation RMS pour préserver l'énergie globale
        output_rms = np.sqrt(np.mean(output_signal**2))
        if output_rms > 1e-9:
            output_signal *= input_rms / output_rms

        return output_signal

    def _step1_crossovers(
        self, input_signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Étape 1 : Application des crossovers.

        Sépare les signaux en HF et BF, calcule LF_mono1.

        Returns
        -------
        hf_signals : np.ndarray
            Signaux haute fréquence (n_samples, n_channels).
        lf_mono1 : np.ndarray
            Signal mono basse fréquence (n_samples,).
        """
        # Crossover.process retourne (lf, hf, lf_mono1)
        _lf_signals, hf_signals, lf_mono1 = self.crossover.process(input_signal)

        return hf_signals, lf_mono1

    def _step2_lfe(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Étape 2 : Création/détection du LFE.

        Returns
        -------
        lfe_signal : np.ndarray
            Signal LFE (n_samples,).
        """
        lfe_signal = self.lfe_processor.process(input_signal)
        return lfe_signal

    def _step3_upmix_frequentiel(self, hf_signals: np.ndarray) -> List[np.ndarray]:
        """
        Étape 3 : Upmix fréquentiel.

        Applique STFT, estime le panning, extrait les sources.

        Returns
        -------
        extracted_sources : List[np.ndarray]
            Liste des signaux sources extraits (domaine temporel).
        """
        n_samples = hf_signals.shape[0]

        # STFT multicanal
        stft = self.stft_processor.forward_multichannel(hf_signals)

        # Estimation du panning
        stft_magnitudes = np.abs(stft)
        panning = estimate_panning(stft_magnitudes, layout=self.input_layout)

        # Puissance pour le freeze (Spec ligne 561)
        # On utilise la puissance totale sur tous les canaux pour décider du freeze
        power = np.sum(stft_magnitudes**2, axis=-1)

        # Construire les paramètres des sources
        source_params = self._build_source_params()

        # Extraction des sources (domaine STFT)
        extracted_stft = self.extractor.extract_batch(
            stft=stft,
            panning=panning,
            source_params=source_params,
            apply_blur=True,
            apply_smoothing=True,
            power=power,
        )

        # ISTFT pour chaque source
        extracted_sources = []
        for stft_source in extracted_stft:
            signal = self.stft_processor.inverse(stft_source, n_samples)
            extracted_sources.append(signal)

        return extracted_sources

    def _step4_add_lf_mono1(
        self,
        extracted_sources: List[np.ndarray],
        lf_mono1: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Étape 4 : Ajout de LF_mono1 aux sources.

        Le signal LF_mono1 est retardé pour compenser la latence STFT.

        Returns
        -------
        sources_with_lf : List[np.ndarray]
            Sources avec LF_mono1 ajouté.
        """
        if len(extracted_sources) == 0:
            return extracted_sources

        # Récupérer les gains LF pour chaque source
        n_sources = len(extracted_sources)
        lf_gains = []

        for i in range(1, n_sources + 1):
            lf_gain_key = f"LF_gain{i}"
            lf_gain = self.upmix_params.get(lf_gain_key, 1.0)
            lf_gains.append(float(lf_gain))

        # Ajouter LF_mono1 retardé
        sources_with_lf = add_lf_mono1_to_sources(
            sources=extracted_sources,
            lf_mono1=lf_mono1,
            lf_gains=lf_gains,
            latency_samples=self.latency_samples,
        )

        return sources_with_lf

    def _step5_respatialization(
        self,
        sources: List[np.ndarray],
        lfe_signal: np.ndarray,
    ) -> np.ndarray:
        """
        Étape 5 : Respatialisation.

        Applique les gains/délais et somme vers le layout de sortie.

        Returns
        -------
        output_signal : np.ndarray
            Signal de sortie multicanal.
        """
        return self.respatializer.spatialize_from_params(
            sources=sources,
            upmix_params=self.upmix_params,
            lfe_signal=lfe_signal,
            lf_mono1=None,  # Déjà ajouté à l'étape 4
            latency_samples=0,
        )

    def _build_source_params(self) -> List[Dict]:
        """
        Construit la liste des paramètres pour chaque source.

        Returns
        -------
        source_params : List[dict]
            Paramètres pour chaque source.
        """
        # Paramètres communs
        width = self.upmix_params.get("width", 0.18)
        slope = self.upmix_params.get("slope", 500.0)
        min_gain_db = self.upmix_params.get("min_gain", -40.0)
        attack = self.upmix_params.get("attack", 1.0)

        source_params = []

        # Calculer le nombre de sources actives
        n_speakers = self.respatializer.n_speakers
        n_sources = min(self.max_sources, n_speakers)

        for i in range(1, n_sources + 1):
            pan_key = f"pan{i}"
            release_key = f"release{i}"
            mute_key = f"mute{i}"

            pan = self.upmix_params.get(pan_key, 0.0)
            release = self.upmix_params.get(release_key, 50.0)
            mute = self.upmix_params.get(mute_key, 0)

            source_params.append(
                {
                    "pan": pan,
                    "width": width,
                    "slope": slope,
                    "min_gain_db": min_gain_db,
                    "attack_frames": attack,
                    "release_frames": release,
                    "mute": mute,
                }
            )

        return source_params

    def process_file(
        self,
        input_path: str,
        output_path: str,
    ) -> None:
        """
        Traite un fichier WAV complet.

        Parameters
        ----------
        input_path : str
            Chemin du fichier d'entrée.
        output_path : str
            Chemin du fichier de sortie.
        """
        # Lecture du fichier
        sample_rate, input_signal = wavfile.read(input_path)
        original_dtype = input_signal.dtype

        # Mise à jour du sample_rate si différent
        if sample_rate != self.sample_rate:
            self.sample_rate = float(sample_rate)
            self._init_modules()

        # Normalisation en float32 [-1, 1]
        if original_dtype == np.int16:
            input_signal_float = input_signal.astype(np.float32) / 32768.0
        elif original_dtype == np.int32:
            input_signal_float = input_signal.astype(np.float32) / 2147483648.0
        else:
            input_signal_float = input_signal.astype(np.float32)

        # Traitement
        output_signal = self.process(input_signal_float)

        # Écriture du fichier de sortie en préservant le format d'entrée (Spec ligne 467)
        if original_dtype == np.int16:
            output_final = np.clip(output_signal * 32768, -32768, 32767).astype(
                np.int16
            )
        elif original_dtype == np.int32:
            output_final = np.clip(
                output_signal * 2147483648, -2147483648, 2147483647
            ).astype(np.int32)
        else:
            output_final = output_signal.astype(original_dtype)

        wavfile.write(output_path, int(self.sample_rate), output_final)

    def get_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur la configuration.

        Returns
        -------
        info : dict
            Informations de configuration.
        """
        return {
            "input_layout": self.input_layout,
            "output_layout": self.output_layout,
            "sample_rate": self.sample_rate,
            "f_xover1": self.f_xover1,
            "f_lfe": self.f_lfe,
            "nfft": self.nfft,
            "overlap": self.overlap,
            "latency_samples": self.latency_samples,
            "latency_ms": self.latency_samples / self.sample_rate * 1000,
            "n_output_channels": self.respatializer.n_output_channels,
            "n_speakers": self.respatializer.n_speakers,
        }


def create_default_params(
    input_layout: str = "stereo",
    output_layout: str = "5.1",
    n_sources: int = 5,
) -> Dict[str, Any]:
    """
    Crée des paramètres par défaut pour l'upmix.

    Parameters
    ----------
    input_layout : str
        Layout d'entrée.
    output_layout : str
        Layout de sortie.
    n_sources : int
        Nombre de sources à extraire.

    Returns
    -------
    params : dict
        Paramètres de configuration.
    """
    upmix_params: Dict[str, Any] = {
        "width": 0.18,
        "slope": 500.0,
        "min_gain": -40.0,
        "attack": 1.0,
    }

    # Générer des paramètres par source
    for i in range(1, n_sources + 1):
        # Pan uniformément réparti
        pan = -1.0 + 2.0 * (i - 1) / max(n_sources - 1, 1)
        upmix_params[f"pan{i}"] = pan
        upmix_params[f"release{i}"] = 50.0
        upmix_params[f"mute{i}"] = 0
        upmix_params[f"LF_gain{i}"] = 1.0

        # Gains par défaut (à personnaliser)
        # Pour 5.1 : L, R, C, LFE, LS, RS (6 canaux)
        if output_layout == "5.1":
            upmix_params[f"gains{i}"] = [0.5, 0.5, 0.0, 0.0, 0.3, 0.3]
            upmix_params[f"delays{i}"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif output_layout == "7.1":
            upmix_params[f"gains{i}"] = [0.5, 0.5, 0.0, 0.0, 0.3, 0.3, 0.2, 0.2]
            upmix_params[f"delays{i}"] = [0.0] * 8
        else:
            # Gains par défaut pour stéréo
            upmix_params[f"gains{i}"] = [0.7, 0.7]
            upmix_params[f"delays{i}"] = [0.0, 0.0]

    return {
        "input_layout": input_layout,
        "output_layout": output_layout,
        "F_xover1": 150.0,
        "F_LFE": 120.0,
        "max_sources": n_sources,
        "nfft": 128,
        "overlap": 0.25,
        "upmix_params": upmix_params,
    }
