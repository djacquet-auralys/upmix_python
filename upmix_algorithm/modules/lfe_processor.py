"""
Module de traitement du canal LFE (Low-Frequency Effects).

Ce module gère la détection des canaux LFE existants et la création
d'un nouveau canal LFE à partir d'une somme filtrée des canaux audio.
"""

from typing import List, Optional, Tuple

import numpy as np

from .biquad_filter import CascadeBiquadFilter, FilterType
from .crossover import sum_power_constant


def detect_lfe_channels(labels: List[str]) -> List[int]:
    """
    Détecte les indices des canaux LFE dans un layout.

    Les canaux LFE sont identifiés par la présence de "LFE" dans leur label.

    Args:
        labels: Liste des labels des canaux (ex: ["L", "R", "C", "LFE", "LS", "RS"]).

    Returns:
        Liste des indices des canaux LFE.
    """
    lfe_indices = []
    for i, label in enumerate(labels):
        if "LFE" in label.upper():
            lfe_indices.append(i)
    return lfe_indices


def get_non_lfe_channels(labels: List[str]) -> List[int]:
    """
    Retourne les indices des canaux non-LFE.

    Args:
        labels: Liste des labels des canaux.

    Returns:
        Liste des indices des canaux non-LFE.
    """
    lfe_indices = detect_lfe_channels(labels)
    return [i for i in range(len(labels)) if i not in lfe_indices]


def create_lfe_from_sum(
    signal: np.ndarray,
    labels: List[str],
    f_lfe: float,
    fs: float,
    q: float = 0.707,
    n_stages: int = 2,
) -> np.ndarray:
    """
    Crée un canal LFE à partir de la somme des canaux non-LFE.

    Formule : LFE = LP(sqrt(sum(all_channels²))) à F_LFE

    Args:
        signal: Signal multicanal (n_samples, n_channels).
        labels: Labels des canaux.
        f_lfe: Fréquence de coupure du filtre LFE en Hz (défaut: 120 Hz).
        fs: Fréquence d'échantillonnage en Hz.
        q: Facteur de qualité du filtre (défaut: 0.707).
        n_stages: Nombre de biquads en cascade (défaut: 2 pour ordre 4).

    Returns:
        Signal LFE (1D array).
    """
    if signal.ndim != 2:
        raise ValueError("Le signal doit être un tableau 2D (n_samples, n_channels)")

    _n_samples, n_channels = signal.shape

    if len(labels) != n_channels:
        raise ValueError(
            f"Le nombre de labels ({len(labels)}) ne correspond pas "
            f"au nombre de canaux ({n_channels})"
        )

    # Récupérer les canaux non-LFE
    non_lfe_indices = get_non_lfe_channels(labels)

    if not non_lfe_indices:
        raise ValueError("Aucun canal non-LFE trouvé")

    # Somme à puissance constante des canaux non-LFE
    non_lfe_signals: List[np.ndarray] = [
        signal[:, i].astype(np.float32) for i in non_lfe_indices
    ]
    mono_sum = sum_power_constant(non_lfe_signals)

    # Appliquer le filtre passe-bas pour LFE
    lfe_filter = CascadeBiquadFilter(f_lfe, q, fs, FilterType.LPF, n_stages=n_stages)
    lfe_signal = lfe_filter.process(mono_sum)

    return lfe_signal


def extract_existing_lfe(signal: np.ndarray, labels: List[str]) -> Optional[np.ndarray]:
    """
    Extrait le canal LFE existant d'un signal multicanal.

    Si plusieurs canaux LFE existent, ils sont sommés en mono.

    Args:
        signal: Signal multicanal (n_samples, n_channels).
        labels: Labels des canaux.

    Returns:
        Signal LFE (1D array) ou None si aucun LFE n'existe.
    """
    if signal.ndim != 2:
        raise ValueError("Le signal doit être un tableau 2D (n_samples, n_channels)")

    lfe_indices = detect_lfe_channels(labels)

    if not lfe_indices:
        return None

    if len(lfe_indices) == 1:
        # Un seul LFE
        return signal[:, lfe_indices[0]].astype(np.float32)
    else:
        # Plusieurs LFE : somme en mono
        lfe_signals: List[np.ndarray] = [
            signal[:, i].astype(np.float32) for i in lfe_indices
        ]
        # Somme simple (pas à puissance constante pour des LFE existants)
        return np.sum(lfe_signals, axis=0)


def process_lfe(
    signal: np.ndarray,
    labels: List[str],
    f_lfe: float,
    fs: float,
    q: float = 0.707,
    n_stages: int = 2,
) -> Tuple[np.ndarray, bool]:
    """
    Traite le canal LFE : extrait l'existant ou en crée un nouveau.

    Args:
        signal: Signal multicanal (n_samples, n_channels).
        labels: Labels des canaux.
        f_lfe: Fréquence de coupure du filtre LFE en Hz.
        fs: Fréquence d'échantillonnage en Hz.
        q: Facteur de qualité du filtre.
        n_stages: Nombre de biquads en cascade.

    Returns:
        Tuple (lfe_signal, was_existing) où was_existing indique si le LFE
        existait déjà dans le signal d'entrée.
    """
    existing_lfe = extract_existing_lfe(signal, labels)

    if existing_lfe is not None:
        return existing_lfe, True
    else:
        created_lfe = create_lfe_from_sum(signal, labels, f_lfe, fs, q, n_stages)
        return created_lfe, False


class LFEProcessor:
    """
    Classe pour le traitement continu du canal LFE.

    Gère la détection et la création de LFE avec état persistant
    pour le traitement en streaming.
    """

    _lfe_filter: Optional[CascadeBiquadFilter]

    def __init__(
        self,
        labels: List[str],
        f_lfe: float,
        fs: float,
        q: float = 0.707,
        n_stages: int = 2,
    ) -> None:
        """
        Initialise le processeur LFE.

        Args:
            labels: Labels des canaux.
            f_lfe: Fréquence de coupure du filtre LFE en Hz.
            fs: Fréquence d'échantillonnage en Hz.
            q: Facteur de qualité du filtre.
            n_stages: Nombre de biquads en cascade.
        """
        self.labels = labels
        self.f_lfe = f_lfe
        self.fs = fs
        self.q = q
        self.n_stages = n_stages

        self.lfe_indices = detect_lfe_channels(labels)
        self.non_lfe_indices = get_non_lfe_channels(labels)
        self.has_existing_lfe = len(self.lfe_indices) > 0

        # Créer le filtre LP pour LFE si nécessaire
        if not self.has_existing_lfe:
            self._lfe_filter = CascadeBiquadFilter(
                f_lfe, q, fs, FilterType.LPF, n_stages
            )
        else:
            self._lfe_filter = None

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Traite le signal et retourne le canal LFE.

        Args:
            signal: Signal multicanal (n_samples, n_channels).

        Returns:
            Signal LFE (1D array).
        """
        if signal.ndim != 2:
            raise ValueError(
                "Le signal doit être un tableau 2D (n_samples, n_channels)"
            )

        n_samples, n_channels = signal.shape

        if n_channels != len(self.labels):
            raise ValueError(
                f"Le nombre de canaux ({n_channels}) ne correspond pas "
                f"aux labels ({len(self.labels)})"
            )

        if self.has_existing_lfe:
            # Extraire le LFE existant
            if len(self.lfe_indices) == 1:
                return signal[:, self.lfe_indices[0]].astype(np.float32)
            else:
                # Somme des LFE multiples
                lfe_sum = np.zeros(n_samples, dtype=np.float32)
                for idx in self.lfe_indices:
                    lfe_sum += signal[:, idx].astype(np.float32)
                return lfe_sum
        else:
            # Créer LFE depuis somme
            assert (
                self._lfe_filter is not None
            )  # Guaranteed when has_existing_lfe=False
            non_lfe_signals: List[np.ndarray] = [
                signal[:, i].astype(np.float32) for i in self.non_lfe_indices
            ]
            mono_sum = sum_power_constant(non_lfe_signals)
            return self._lfe_filter.process(mono_sum)

    def reset(self) -> None:
        """Réinitialise l'état du filtre LFE."""
        if self._lfe_filter is not None:
            self._lfe_filter.reset()
