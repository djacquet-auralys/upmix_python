# core/upmix.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Adapte ces chemins d'import à ton placement réel des modules :
from modules_external import auralys_utils as au
from modules_external import re_model as re_m
from modules_external import tdap as tdap_mod

from .config import C_SOUND, A1_QDIR, A2_QSPACE, A3_QWIDTH, A4_QBLUR

@dataclass
class UpmixResults:
    sources: np.ndarray        # (S,2)
    gains: np.ndarray          # (S,M) linéaires
    spat_delays: np.ndarray    # (S,M) secondes


def place_upmix_sources(audience_bary: np.ndarray,
                        panorama_center: np.ndarray,
                        speaker_pts: np.ndarray) -> np.ndarray:
    """
    Retour: sources : (S, 2) en mètres (coord. XY absolues).
    TODO: Reprendre la logique 'place upmix sources' de ton script (lignes >= 167):
      - calcul des azimuts HP vus depuis audience_bary
      - estimation/usage de la fenêtre de panorama autour de panorama_center
      - répartition des angles de sources dans la fenêtre
      - distances via interpolation sur les HP (tri/angle) si c’est ce que faisait ton script
      - conversion pol->cart puis translation par audience_bary
    """
    # --- EXEMPLE MINIMAL / STUB (à remplacer par ta vraie logique) ---
    # On place 11 sources par défaut, centrées sur panorama_center, espacées régulièrement en angle,
    # avec une distance égale à la distance moyenne audience_bary->HP.
    M = speaker_pts.shape[0]
    S = 11
    # azimut du centre (rad) depuis bary vers panorama_center
    vec_center = panorama_center - audience_bary
    pan0_deg = np.degrees(np.arctan2(vec_center[0], vec_center[1]))  # convention tdap: 0° sur +Y
    width_window_deg = 60.0  # fenêtre par défaut si la tienne n’est pas dispo ici
    angles = np.linspace(pan0_deg - width_window_deg/2, pan0_deg + width_window_deg/2, S)

    # distance moyenne audience->HP
    dists = np.linalg.norm(speaker_pts - audience_bary[None, :], axis=1)
    r = float(np.mean(dists)) if M > 0 else 1.0

    sources = []
    for ang in angles:
        # tdap conv.: x=sin, y=cos pour un azimut en degrés
        rad = np.radians(ang)
        direction = np.array([np.sin(rad), np.cos(rad)])  # unitaire
        src = audience_bary + r * direction
        sources.append(src)
    return np.asarray(sources, dtype=float)


def compute_gains_delays(spread: float,
                         panorama_width: float,           # degrés
                         sources: np.ndarray,             # (S,2)
                         audience_bary: np.ndarray,       # (2,)
                         speaker_pts: np.ndarray          # (M,2)
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Retour:
      gains: (S, M) linéaires
      spat_delays: (S, M) en SECONDES (cohérent avec re_model)
    Méthode:
      - pour chaque source s:
        pan_s = angle_tdap(s - audience_bary); width_s = f(panorama_width, spread)
        g_s = tdap.tdap(pan_s_deg, width_s_deg, hp_rel) -> (M,)
        delay_sj = dist(source, hp_j)/C_SOUND (s)
    """
    S = sources.shape[0]
    M = speaker_pts.shape[0]
    gains = np.zeros((S, M), dtype=float)
    delays = np.zeros((S, M), dtype=float)

    # width "locale" par source — à ajuster avec ta formule exacte
    # (dans ton script tu faisais varier width selon nb_sources et spread)
    width_per_src = max(1e-3, panorama_width / max(S, 1)) * (2.0 * max(spread, 1e-6))

    # HP en coords relatives au barycentre
    hp_rel = speaker_pts - audience_bary[None, :]

    for s_idx, s in enumerate(sources):
        rel = s - audience_bary
        # pan_s en degrés (conv. tdap: 0° sur +Y)
        pan_s_deg = np.degrees(np.arctan2(rel[0], rel[1]))
        # TDAP gains linéaires
        # tdap.tdap(pan_deg, width_deg, hp_rel_xy) -> (M,)  (selon ton tdap.py)
        g_lin = tdap_mod.tdap(pan_s_deg, width_per_src, hp_rel)
        gains[s_idx, :] = g_lin

        # délais spatiaux en secondes pour re_model
        dist_sm = np.linalg.norm(speaker_pts - s[None, :], axis=1)
        delays[s_idx, :] = dist_sm / C_SOUND

    return gains, delays


def compute_re_model(seats: np.ndarray,              # (N,2)
                     speakers: np.ndarray,           # (M,2)
                     speaker_axis: np.ndarray,       # (M,2) points "axe" (orientations)
                     directivity: float,             # degrés d’ouverture
                     spat_gains: np.ndarray,         # (M,) linéaires pour UNE source
                     spat_delays: np.ndarray         # (M,) secondes pour UNE source
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Façade vers re_model.compute_re_model(...)
    Retour: re: (N,2), w: (N,)
    """
    # IMPORTANT: re_model attend des gains LINÉAIRES (il convertit lui-même en dB)
    # et des délais en SECONDES (il additionne avec le délai acoustique).
    re, w = re_m.compute_re_model(
        seats, speakers, speaker_axis, directivity,
        spat_gains, spat_delays
    )
    return re, w


def compute_seat_score(seats: np.ndarray,            # (N,2) donné par l’UI
                       speaker_pts: np.ndarray,      # (M,2)
                       spk_orientations: np.ndarray, # (M,) en degrés
                       directivity: float,           # degrés
                       gains: np.ndarray,            # (S,M) linéaires
                       spat_delays: np.ndarray       # (S,M) secondes
                       ) -> np.ndarray:
    """
    Retour: seat_score: (N,)
    - Pour chaque source s: compute_re_model(...) => re_s, w_s
    - Combiner re_s/w_s en indicateurs psychoacoustiques puis score global par siège.
    - Aucune visualisation ici.
    """
    N = seats.shape[0]
    S, M = gains.shape

    # Construire speaker_axis à partir des orientations (un point “devant” chaque HP)
    # Convention: 0° sur +Y (comme tdap) => point à 1 m devant
    rad = np.radians(spk_orientations)
    forward = np.stack([np.sin(rad), np.cos(rad)], axis=1)  # (M,2)
    speaker_axis = speaker_pts + forward  # (M,2)

    # Accumulateurs psychoacoustiques (exemples — ajuste selon ta formule exacte)
    # On peut stocker la somme des vecteurs énergie et largeurs perçues
    re_sum = np.zeros((N, 2), dtype=float)
    w_sum = np.zeros(N, dtype=float)

    for s in range(S):
        re_s, w_s = compute_re_model(
            seats, speaker_pts, speaker_axis, directivity,
            gains[s, :], spat_delays[s, :]
        )
        re_sum += re_s
        w_sum += w_s

    # Indicateurs simples (placeholders, à remplacer par ta métrique exacte)
    # - Qdir: alignement directionnel (norme du vecteur / somme des normes)
    # - Qspace: par ex. pénalité d’écart angulaire par rapport à une cible (0 si pas de cible)
    # - Qwidth: normalisation de w_sum dans [0,1] (selon bornes désirées)
    # - Qblur: pénalité liée à la variance temporelle spatiale (si applicable)
    eps = 1e-9
    re_norm = np.linalg.norm(re_sum, axis=1)  # (N,)
    Qdir = re_norm / (np.max(re_norm) + eps)

    # Placeholders basiques :
    Qspace = np.ones(N, dtype=float)       # à définir selon ton modèle
    Qwidth = w_sum / (np.max(w_sum) + eps) # normalisation simple
    Qblur  = np.ones(N, dtype=float)       # à définir si tu as un indicateur

    seat_score = (
        A1_QDIR   * Qdir +
        A2_QSPACE * Qspace +
        A3_QWIDTH * Qwidth +
        A4_QBLUR  * Qblur
    )
    # Clip dans [0,1] par sécurité
    seat_score = np.clip(seat_score, 0.0, 1.0)
    return seat_score


def send_upmix_params(sources: np.ndarray,           # (S,2)
                      max_sources: int,
                      release_max: float,
                      release_min: float,
                      slope: float,
                      gains: np.ndarray,             # (S,M)
                      spat_delays: np.ndarray,       # (S,M) secondes
                      attack: float,
                      *,
                      osc_host: str = "127.0.0.1",
                      osc_port: int = 4000,
                      write_file: bool = False,
                      filepath: str | None = None) -> dict:
    """
    - Prépare un dict {pan, width, gains, delays, release/attack, ...}
    - Envoie en OSC /upmix (JSON string)
    - Option: écrit ce dict en fichier si demandé
    - Retourne le dict pour debug/tests
    """
    import json

    S, M = gains.shape
    # Pans en degrés depuis barycentre "virtuel" — ici on prend le barycentre des sources
    audience_bary = np.mean(sources, axis=0) if S else np.zeros(2)
    pans = []
    for s in sources:
        rel = s - audience_bary
        pans.append(float(np.degrees(np.arctan2(rel[0], rel[1]))))

    # Widths: placeholder uniforme ; adapte si tu as une loi spécifique
    if S > 0:
        uniform_width = 2.0 * 180.0 / S  # ex. symbolique
    else:
        uniform_width = 0.0
    widths = [uniform_width] * S

    upmix_params = {
        "max_sources": int(max_sources),
        "attack": float(attack),
        "release_min": float(release_min),
        "release_max": float(release_max),
        "slope": float(slope),
        "sources": [
            {
                "pan_deg": float(pans[i]),
                "width_deg": float(widths[i]),
                "gains": gains[i, :].tolist(),            # linéaires
                "delays_sec": spat_delays[i, :].tolist()  # secondes
            }
            for i in range(S)
        ]
    }

    # Envoi OSC (JSON string)
    try:
        from pythonosc.udp_client import SimpleUDPClient
        client = SimpleUDPClient(osc_host, osc_port)
        client.send_message("/upmix", json.dumps(upmix_params))
    except Exception:
        # Laisse passer si l'OSC n'est pas installé/joignable ; au besoin, loggue
        pass

    # Option: écriture fichier
    if write_file and filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(upmix_params, f, ensure_ascii=False, indent=2)

    return upmix_params
