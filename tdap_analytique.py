# -*- coding: utf-8 -*-
"""
TDAP — Triangular Distribution Amplitude Panning
Alignement analytique du vecteur d’énergie (option 2, solution fermée par morceaux)
+ Visualisations complètes (Figures 1→6) et
+ Figure 7 (biais intrinsèque : angle r_E (triangle brut, SANS alignement) vs pan)

Un seul appel pour une scène donnée:
    visualize_tdap_all(spk, pan, width, radius=10.0,
                       include_bias_curve=True, bias_n=360)

Ou, si tu utilises ton module de layouts:
    visualize_tdap_on_layout(layout_id, pan, width, radius=10.0,
                             include_bias_curve=True, bias_n=360)

- Figure 1 : Carte XY (HP, gains, cible, r_E)
- Figure 2 : Azimuts HP vs gains + fenêtre W
- Figure 3 : Axe δ et événements (construction de l’intervalle courant)
- Figure 4 : Quadratique f(δ)=u_perp^T r_E(δ) + racines
- Figure 5 : Profils affines t_i(δ) pour HP actifs + valeur en δ*
- Figure 6 : Angle(r_E(δ)) vs pan + marquage en δ*
- Figure 7 : Angle perçu de r_E (triangle brut, sans alignement) en fonction de pan
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# ============================
# ===  Outils angles       ===
# ============================

def normalize_angle_signed(angle_deg):
    """Vectorisé: normalise en degrés dans [-180°,180°]."""
    a = np.asanyarray(angle_deg, dtype=float)
    ang = np.mod(a, 360.0)
    ang = np.where(ang > 180.0, ang - 360.0, ang)
    return float(ang) if np.isscalar(angle_deg) else ang

def normalize_angle_custom(angle_deg, min_angle):
    """Vectorisé: normalise en degrés dans [min_angle, min_angle+360]."""
    a = np.asanyarray(angle_deg, dtype=float)
    m = np.asanyarray(min_angle, dtype=float)
    res = np.mod(a - m, 360.0) + m
    return float(res) if (np.isscalar(angle_deg) and np.isscalar(min_angle)) else res

# ============================
# ===  Helpers internes    ===
# ============================

def _second_smallest_onepass(vals):
    """2e plus petit en O(N) sans tri. +inf si N<2."""
    m1 = np.inf; m2 = np.inf
    for v in vals:
        if v < m1: m2, m1 = m1, v
        elif v < m2: m2 = v
    return m2

def _bilateral_width_floor(diffs_signed_deg):
    """Borne bilatérale sur diffs ∈ [-180,180]. Fallback: second_smallest."""
    left  = np.abs(diffs_signed_deg[diffs_signed_deg < 0])
    right = np.abs(diffs_signed_deg[diffs_signed_deg > 0])
    dL = np.min(left)  if left.size  else np.inf
    dR = np.min(right) if right.size else np.inf
    if np.isfinite(dL) and np.isfinite(dR): return max(dL, dR)
    return _second_smallest_onepass(np.abs(diffs_signed_deg))

def _deg2rad_math(pan_deg):
    """Convertit 'pan' (0° sur +Y, sens +) en angle math (0 rad sur +X, trigo)."""
    return np.deg2rad((90.0 - pan_deg) % 360.0)

def _angle_from_re_xy(re):
    """Angle (deg) selon convention du code: degrees(atan2(re_x, re_y)) mod 360."""
    return float(np.degrees(np.arctan2(re[0], re[1])) % 360.0)

# ============================
# ===  TDAP analytique     ===
# ============================

def tdap_analytic(pan, width, spk, min_angle=None, eps=1e-9, return_debug=False):
    """
    TDAP avec ALIGNEMENT ANALYTIQUE du vecteur d'énergie sur la direction cible.
    """
    width = float(np.clip(width, 0.0, 300.0))
    spk = np.asarray(spk, dtype=float)
    nb_spk = spk.shape[0]

    spk_az = np.degrees(np.arctan2(spk[:, 0], spk[:, 1]))
    base = pan - 180.0 if min_angle is None else float(min_angle)
    spk_az = normalize_angle_custom(spk_az, base)

    src_angle = _deg2rad_math(pan)
    u = np.array([np.cos(src_angle), np.sin(src_angle)], dtype=float)
    u_perp = np.array([-np.sin(src_angle), np.cos(src_angle)], dtype=float)

    diffs0 = normalize_angle_signed(spk_az - pan)
    width_floor = _bilateral_width_floor(diffs0)
    width = float(max(width, width_floor))

    active_mask = np.abs(diffs0) < width - eps
    if not np.any(active_mask):
        width = float(max(width, width_floor if np.isfinite(width_floor) else 0.0))
        active_mask = np.abs(diffs0) < max(width - eps, 0.0)

    if np.count_nonzero(active_mask) < 2:
        order = np.argsort(np.abs(diffs0))
        keep = order[:2]
        active_mask[:] = False; active_mask[keep] = True

    d = diffs0
    left_edge = -np.inf; right_edge = np.inf
    for i in range(nb_spk):
        di = d[i]
        for e in (di, di - width, di + width):
            if e < 0 and e > left_edge:  left_edge  = e
            if e > 0 and e < right_edge: right_edge = e

    delta_L = left_edge  if np.isfinite(left_edge)  else -180.0
    delta_R = right_edge if np.isfinite(right_edge) else +180.0

    sig = np.sign(d); sig[sig == 0] = 1.0
    A_vec = np.zeros(2); B_vec = np.zeros(2); C_vec = np.zeros(2)
    W = float(width)

    for i in range(nb_spk):
        if not active_mask[i]: continue
        si = sig[i]; ci = W - si * d[i]
        a_i = 1.0; b_i = 2.0 * ci * si; c_i = ci * ci
        A_vec += a_i * spk[i]
        B_vec += b_i * spk[i]
        C_vec += c_i * spk[i]

    a = float(np.dot(u_perp, A_vec))
    b = float(np.dot(u_perp, B_vec))
    c = float(np.dot(u_perp, C_vec))

    delta_star = 0.0; found = False; roots_all = []
    if abs(a) <= eps:
        if abs(b) > eps:
            delta_lin = -c / b
            roots_all = [delta_lin]
            if delta_L + eps < delta_lin < delta_R - eps:
                delta_star = float(delta_lin); found = True
    else:
        disc = b*b - 4.0*a*c
        if disc >= 0.0:
            sd = float(np.sqrt(disc))
            r1 = (-b - sd) / (2.0*a); r2 = (-b + sd) / (2.0*a)
            roots_all = [r1, r2]
            cand = [r for r in (r1, r2) if (delta_L + eps < r < delta_R - eps)]
            if cand:
                delta_star = float(min(cand, key=lambda r: abs(r))); found = True

    if not found: delta_star = 0.0

    gains_unnorm = np.zeros(nb_spk); invW = 1.0 / max(W, eps)
    for i in range(nb_spk):
        if not active_mask[i]: continue
        si = sig[i]; ci = W - si * d[i]
        t = (ci + si * delta_star) * invW
        gains_unnorm[i] = max(t, 0.0)

    norm2 = float(np.sum(gains_unnorm**2))
    if norm2 > eps:
        gains = gains_unnorm / np.sqrt(norm2)
    else:
        gains = np.zeros(nb_spk); order = np.argsort(np.abs(diffs0))
        gains[order[:2]] = 1.0; gains /= np.linalg.norm(gains)

    re = np.sum((gains**2)[:, None] * spk, axis=0)

    if not return_debug:
        return gains, re

    dbg = dict(
        pan=float(pan), width=float(width), W=float(W),
        spk=np.array(spk, float), spk_az=np.array(spk_az, float),
        diffs0=np.array(diffs0, float), active_mask=np.array(active_mask, bool),
        sig=np.array(sig, float),
        A_vec=np.array(A_vec, float), B_vec=np.array(B_vec, float), C_vec=np.array(C_vec, float),
        a=float(a), b=float(b), c=float(c),
        delta_L=float(delta_L), delta_R=float(delta_R),
        delta_star=float(delta_star), roots_all=np.array(roots_all, float),
        src_angle=float(src_angle)
    )
    return gains, re, dbg

# ============================
# ===  Triangle BRUT (no align)
# ============================

def tdap_triangle_raw(pan, width, spk, min_angle=None, eps=1e-9):
    """
    Distribution triangulaire brute centrée sur 'pan' (δ=0), SANS alignement r_E.
    Garantit ≥2 HP actifs via la borne bilatérale. Retourne gains normalisés + r_E.
    """
    width = float(np.clip(width, 0.0, 300.0))
    spk = np.asarray(spk, dtype=float)
    nb_spk = spk.shape[0]

    spk_az = np.degrees(np.arctan2(spk[:, 0], spk[:, 1]))
    base = pan - 180.0 if min_angle is None else float(min_angle)
    spk_az = normalize_angle_custom(spk_az, base)

    diffs = normalize_angle_signed(spk_az - pan)
    width_floor = _bilateral_width_floor(diffs)
    W = float(max(width, width_floor))

    t = np.maximum((W - np.abs(diffs)) / max(W, eps), 0.0)   # rampe triangulaire
    norm2 = float(np.sum(t**2))
    if norm2 > eps:
        g = t / np.sqrt(norm2)
    else:
        g = np.zeros(nb_spk); order = np.argsort(np.abs(diffs))
        g[order[:2]] = 1.0 / np.sqrt(2.0)

    re = np.sum((g**2)[:, None] * spk, axis=0)
    return g, re

# ============================
# ===  Visualisations 1–6  ===
# ============================

def _visualize_fig12(spk, gains, pan, width, re, radius=10.0, title_suffix="TDAP analytique"):
    """Figures 1 & 2 (XY + azimuts/gains)."""
    spk = np.asarray(spk, dtype=float); N = spk.shape[0]
    spk_az = np.degrees(np.arctan2(spk[:, 0], spk[:, 1]))
    ang = _deg2rad_math(pan); source_pt = np.array([np.cos(ang), np.sin(ang)]) * radius

    # Fig 1 — plan XY
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal', adjustable='box')
    ax1.plot(spk[:, 0], spk[:, 1], 'or', label='HP')
    for i in range(N):
        ax1.arrow(spk[i, 0], spk[i, 1], 0, gains[i] * radius * 0.5,
                  linewidth=2, color='k', length_includes_head=True,
                  head_width=radius * 0.03, head_length=radius * 0.06)
    ax1.arrow(0, 0, re[0], re[1], linewidth=3, color='b',
              length_includes_head=True, head_width=radius * 0.04, head_length=radius * 0.08, label='r_E')
    ax1.plot([0, source_pt[0]], [0, source_pt[1]], 'm-', linewidth=2, label='Cible')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_title(f'{title_suffix} — plan XY (pan={pan}°, width={width}°)')
    ax1.grid(True); ax1.legend(loc='best')

    # Fig 2 — azimuts vs gains
    fig2, ax2 = plt.subplots()
    ax2.plot(spk_az, np.zeros_like(spk_az), '^r', label='HP')
    for i in range(N):
        ax2.arrow(spk_az[i], 0, 0, gains[i],
                  linewidth=2, color='k', length_includes_head=True,
                  head_width=3.0, head_length=0.05)
    ax2.arrow(pan % 360, 0, 0, 1.0, linewidth=2, color='m',
              length_includes_head=True, head_width=3.0, head_length=0.08, label='Cible')
    re_angle = _angle_from_re_xy(re)
    ax2.arrow(re_angle, 0, 0, 1.0, linewidth=2, color='b', linestyle='dashed',
              length_includes_head=True, head_width=3.0, head_length=0.08, label='r_E (angle)')

    wL = (pan - width) % 360; wR = (pan + width) % 360
    if wL < wR:
        ax2.axvspan(wL, wR, color='gray', alpha=0.1, label='Fenêtre W')
    else:
        ax2.axvspan(0, wR, color='gray', alpha=0.1, label='Fenêtre W')
        ax2.axvspan(wL, 360, color='gray', alpha=0.1)

    # ax2.set_xlim([0, 360]);
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Azimut (deg)'); ax2.set_ylabel('Gain (lin.)')
    ax2.set_title(f'{title_suffix} — azimuts vs gains')
    ax2.grid(True); ax2.legend(loc='upper right')

    return (fig1, ax1), (fig2, ax2)

def _visualize_fig346(dbg, n_grid=400):
    """Figures 3, 4, 5, 6 (événements δ, quadratique, profils affines, angle r_E)."""
    pan = dbg['pan']; W = dbg['W']; spk = dbg['spk']
    d0 = dbg['diffs0']; active = dbg['active_mask']; sig = dbg['sig']
    A_vec = dbg['A_vec']; B_vec = dbg['B_vec']; C_vec = dbg['C_vec']
    a = dbg['a']; b = dbg['b']; c = dbg['c']
    dL = dbg['delta_L']; dR = dbg['delta_R']
    dstar = dbg['delta_star']; roots_all = dbg['roots_all']

    # Fig 3
    fig3, ax3 = plt.subplots()
    ax3.set_title("Figure 3 — Axe δ et événements (intervalle courant)")
    ax3.set_xlabel("δ (deg)"); ax3.set_yticks([])
    e_sign  = d0; e_left = d0 - W; e_right = d0 + W
    ax3.axvline(0, color='k', linestyle='--', linewidth=1, label='δ=0')
    ax3.axvspan(dL, dR, color='gray', alpha=0.2, label='intervalle courant')
    ax3.plot(e_sign,  np.zeros_like(e_sign)+0.1, 'x', color='C1', label='δ=d_i (chgt signe)')
    ax3.plot(e_left,  np.zeros_like(e_left)+0.0,  '|', color='C2', markersize=12, label='δ=d_i−W (bord actif)')
    ax3.plot(e_right, np.zeros_like(e_right)-0.1, '|', color='C3', markersize=12, label='δ=d_i+W (bord actif)')
    all_events = np.concatenate([e_sign, e_left, e_right])
    ax3.set_xlim(min(dL, np.min(all_events))-5, max(dR, np.max(all_events))+5)
    ax3.legend(loc='upper right')

    # Fig 4
    fig4, ax4 = plt.subplots()
    ax4.set_title("Figure 4 — f(δ)=u_perp^T r_E(δ) sur l’intervalle, racines")
    ax4.set_xlabel("δ (deg)"); ax4.set_ylabel("f(δ)")
    dd = np.linspace(dL, dR, max(3, n_grid))
    fnum = a*dd*dd + b*dd + c
    ax4.plot(dd, fnum, 'b-'); ax4.axhline(0, color='k', linewidth=1)
    for r in np.atleast_1d(roots_all):
        color = 'g' if (dL < r < dR) else '0.6'
        ax4.plot([r], [0], 'o', color=color, markersize=8)
    ax4.plot([dstar], [0], 'o', color='m', markersize=10, label='δ* choisi')
    ax4.legend(loc='best')

    # Fig 5
    fig5, ax5 = plt.subplots()
    ax5.set_title("Figure 5 — Profils affines t_i(δ) pour les HP actifs")
    ax5.set_xlabel("δ (deg)"); ax5.set_ylabel("t_i(δ)")
    for i in range(len(spk)):
        if not active[i]: continue
        ci = W - sig[i] * d0[i]
        ti = (ci + sig[i] * dd) / W
        ax5.plot(dd, ti, '-', alpha=0.85)
        ti_star = (ci + sig[i] * dstar) / W
        ax5.plot([dstar], [ti_star], 'ko', ms=4)
    ax5.axvline(dstar, color='m', linestyle='--', linewidth=1, label='δ*')
    ax5.legend(loc='best'); ax5.grid(True)

    # Fig 6
    fig6, ax6 = plt.subplots()
    ax6.set_title("Figure 6 — Angle de r_E(δ) vs cible pan")
    ax6.set_xlabel("δ (deg)"); ax6.set_ylabel("angle(r_E(δ)) (deg)")
    rE = (A_vec[None, :] * (dd**2)[:, None] + B_vec[None, :] * dd[:, None] + C_vec[None, :])
    ang_deg = (np.degrees(np.arctan2(rE[:, 0], rE[:, 1])) % 360.0)
    ax6.plot(dd, ang_deg, 'b-', label='angle r_E(δ)')
    ax6.axhline(pan % 360.0, color='m', linestyle='--', label='pan (cible)')
    re_star = A_vec*(dstar**2) + B_vec*dstar + C_vec
    ang_star = _angle_from_re_xy(re_star)
    ax6.plot([dstar], [ang_star], 'ko', label='δ*, angle(r_E)')
    ax6.legend(loc='best'); ax6.grid(True)

    return (fig3, ax3), (fig4, ax4), (fig5, ax5), (fig6, ax6)

# ============================
# ===  Figure 7 (biais)    ===
# ============================

def _visualize_fig7_bias_curve(spk, width, bias_n=360, min_angle=None, eps=1e-9):
    """
    Figure 7 — Angle perçu de r_E (triangle BRUT, SANS alignement) en fonction de pan.
      x : pan (0..360)
      y : angle(r_E) (0..360), même convention (0° sur +Y, croît vers +X)
    """
    pans = np.linspace(0.0, 360.0, int(bias_n), endpoint=False)
    y = np.zeros_like(pans)
    for k, pan in enumerate(pans):
        g, re = tdap_triangle_raw(pan, width, spk, min_angle=min_angle, eps=eps)
        y[k] = _angle_from_re_xy(re)

    fig7, ax7 = plt.subplots()
    ax7.set_title("Figure 7 — Angle perçu r_E (triangle brut, sans alignement) vs pan")
    ax7.set_xlabel("pan (deg)"); ax7.set_ylabel("angle r_E (deg)")
    ax7.plot(pans, y, '-', label='angle r_E (brut)')
    # diagonale de référence y=x
    ax7.plot(pans, pans, 'm--', linewidth=1.5, label='y = pan (référence)')
    ax7.set_xlim(0, 360); ax7.set_ylim(0, 360)
    ax7.set_xticks(np.arange(0, 361, 45)); ax7.set_yticks(np.arange(0, 361, 45))
    ax7.grid(True); ax7.legend(loc='best')
    return fig7, ax7

# ============================
# ===  Un seul appel      ===
# ============================

def visualize_tdap_all(spk, pan, width, radius=10.0, show=True,
                       include_bias_curve=False, bias_n=360):
    """
    UN SEUL APPEL pour :
      - calculer les gains TDAP analytiques,
      - produire les Figures 1→6,
      - (optionnel) Figure 7 : biais intrinsèque angle r_E vs pan sans alignement.
    """
    gains, re, dbg = tdap_analytic(pan, width, spk, return_debug=True)
    (fig1, ax1), (fig2, ax2) = _visualize_fig12(spk, gains, pan, width, re, radius=radius)
    (fig3, ax3), (fig4, ax4), (fig5, ax5), (fig6, ax6) = _visualize_fig346(dbg)

    fig7 = ax7 = None
    if include_bias_curve:
        fig7, ax7 = _visualize_fig7_bias_curve(spk, width, bias_n=bias_n)

    if show: plt.show()
    return {'fig1':(fig1,ax1),'fig2':(fig2,ax2),'fig3':(fig3,ax3),'fig4':(fig4,ax4),
            'fig5':(fig5,ax5),'fig6':(fig6,ax6),'fig7':(fig7,ax7),
            'gains':gains,'re':re,'dbg':dbg}

# ============================
# ===  Intégration layouts ===
# ============================

def visualize_tdap_on_layout(layout_id, pan, width, radius=10.0, show=True,
                             include_bias_curve=False, bias_n=360):
    """
    Utilise 'multichannel_layouts' pour extraire les HP (hors LFE) et
    appelle visualize_tdap_all(...) en un coup.
    """
    try:
        import multichannel_layouts as mc
    except Exception as e:
        raise RuntimeError("Le module 'multichannel_layouts' est introuvable. "
                           "Installe-le ou utilise visualize_tdap_all(...) avec tes positions.")
    speakers = mc.get_spk_coordinates(layout_id, radius=radius)
    labels_arr = np.array(speakers['labels'])
    mask = ~np.char.startswith(labels_arr, "LFE")
    spk = speakers['positions_xy'][mask]
    return visualize_tdap_all(spk, pan, width, radius=radius, show=show,
                              include_bias_curve=include_bias_curve, bias_n=bias_n)

# ============================
# ===  Démo autonome      ===
# ============================

if __name__ == '__main__':
    # --- Choisis ici ---
    USE_MC = True            # True pour utiliser multichannel_layouts (si dispo)
    LAYOUT = '7.1'           # id de layout si USE_MC=True
    radius_demo = 10.0
    np.random.seed(1)
    pan = np.random.randint(-180, 180)
    width = np.random.randint(30, 120)

    if USE_MC:
        print(f"[Layout] {LAYOUT} — pan={pan}°, width={width}°")
        handles = visualize_tdap_on_layout(LAYOUT, pan, width, radius=radius_demo,
                                           show=True, include_bias_curve=True, bias_n=360)
    else:
        # Anneau de N HP (démo générique)
        N = 16
        az = np.linspace(0, 2*np.pi, N, endpoint=False)
        spk_demo = np.column_stack([radius_demo * np.cos(az), radius_demo * np.sin(az)])
        print(f"[Anneau] N={N} — pan={pan}°, width={width}°")
        handles = visualize_tdap_all(spk_demo, pan, width, radius=radius_demo,
                                     show=True, include_bias_curve=True, bias_n=360)

    # Exemple d’accès :
    # gains = handles['gains']; re = handles['re']; dbg = handles['dbg']
