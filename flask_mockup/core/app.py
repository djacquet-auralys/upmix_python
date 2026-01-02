# api/app.py

from __future__ import annotations
import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Ajuste ces imports au chemin réel de ton projet
from core.upmix import (
    place_upmix_sources,
    compute_gains_delays,
    compute_seat_score,
    send_upmix_params,
)

app = Flask(__name__)
CORS(app)  # enlève si front/back même origin

def as_ndarray(x, shape_last=None, name="array"):
    arr = np.asarray(x, dtype=float)
    if shape_last is not None and arr.ndim > 0 and arr.shape[-1] != shape_last:
        raise ValueError(f"{name} doit avoir shape (...,{shape_last}), reçu {arr.shape}")
    return arr

@app.route("/api/upmix/v1/place-sources", methods=["POST"])
def api_place_sources():
    data = request.get_json(force=True)
    audience_bary = as_ndarray(data["audience_bary"], 2, "audience_bary")
    panorama_center = as_ndarray(data["panorama_center"], 2, "panorama_center")
    speaker_pts = as_ndarray(data["speaker_pts"], 2, "speaker_pts")
    sources = place_upmix_sources(audience_bary, panorama_center, speaker_pts)
    return jsonify({"sources": sources.tolist()})

@app.route("/api/upmix/v1/gains-delays", methods=["POST"])
def api_gains_delays():
    data = request.get_json(force=True)
    spread = float(data["spread"])
    panorama_width = float(data["panorama_width"])  # degrés
    sources = as_ndarray(data["sources"], 2, "sources")
    audience_bary = as_ndarray(data["audience_bary"], 2, "audience_bary")
    speaker_pts = as_ndarray(data["speaker_pts"], 2, "speaker_pts")

    gains, spat_delays = compute_gains_delays(
        spread, panorama_width, sources, audience_bary, speaker_pts
    )
    return jsonify({
        "gains": gains.tolist(),
        "spat_delays": spat_delays.tolist(),
        "delays_unit": "seconds"
    })

@app.route("/api/upmix/v1/compute-seat-score", methods=["POST"])
def api_compute_seat_score():
    data = request.get_json(force=True)
    seats = as_ndarray(data["seats"], 2, "seats")
    speaker_pts = as_ndarray(data["speaker_pts"], 2, "speaker_pts")
    spk_orientations = as_ndarray(data["spk_orientations"], None, "spk_orientations")
    directivity = float(data["directivity"])
    gains = as_ndarray(data["gains"], None, "gains")
    spat_delays = as_ndarray(data["spat_delays"], None, "spat_delays")

    # Validations rapides
    if gains.ndim != 2 or spat_delays.ndim != 2:
        return jsonify({"error": "gains et spat_delays doivent être (S,M)"}), 400
    if speaker_pts.shape[0] != gains.shape[1]:
        return jsonify({"error": "Mismatch: nb HP != gains.shape[1]"}), 400
    if speaker_pts.shape[0] != spat_delays.shape[1]:
        return jsonify({"error": "Mismatch: nb HP != spat_delays.shape[1]"}), 400
    if spk_orientations.shape[0] != speaker_pts.shape[0]:
        return jsonify({"error": "Mismatch: len(orientations) != nb HP"}), 400

    seat_score = compute_seat_score(
        seats, speaker_pts, spk_orientations, directivity, gains, spat_delays
    )
    return jsonify({"seat_score": seat_score.tolist()})

@app.route("/api/upmix/v1/seat-score", methods=["POST"])
def api_seat_score_pipeline():
    """
    Route pipeline complète: place sources -> gains/delays -> seat_score
    Toutes les entrées viennent de l’UI (notamment seats)
    """
    data = request.get_json(force=True)

    audience_bary = as_ndarray(data["audience_bary"], 2, "audience_bary")
    panorama_center = as_ndarray(data["panorama_center"], 2, "panorama_center")
    speaker_pts = as_ndarray(data["speaker_pts"], 2, "speaker_pts")
    spk_orientations = as_ndarray(data["spk_orientations"], None, "spk_orientations")
    directivity = float(data["directivity"])
    panorama_width = float(data["panorama_width"])
    spread = float(data["spread"])
    seats = as_ndarray(data["seats"], 2, "seats")

    # Étape 1: sources
    sources = place_upmix_sources(audience_bary, panorama_center, speaker_pts)

    # Étape 2: gains + délais
    gains, spat_delays = compute_gains_delays(
        spread, panorama_width, sources, audience_bary, speaker_pts
    )

    # Étape 3: score sièges
    seat_score = compute_seat_score(
        seats, speaker_pts, spk_orientations, directivity, gains, spat_delays
    )

    return jsonify({
        "seat_score": seat_score.tolist(),
        "nb_sources": int(sources.shape[0]),
        "sources": sources.tolist(),
        "gains_shape": list(gains.shape),
        "delays_unit": "seconds"
    })

@app.route("/api/upmix/v1/send-upmix", methods=["POST"])
def api_send_upmix():
    data = request.get_json(force=True)
    sources = as_ndarray(data["sources"], 2, "sources")
    gains = as_ndarray(data["gains"], None, "gains")
    spat_delays = as_ndarray(data["spat_delays"], None, "spat_delays")
    upmix_params = send_upmix_params(
        sources=sources,
        max_sources=int(data.get("max_sources", sources.shape[0])),
        release_max=float(data["release_max"]),
        release_min=float(data["release_min"]),
        slope=float(data["slope"]),
        gains=gains,
        spat_delays=spat_delays,
        attack=float(data["attack"]),
        osc_host=str(data.get("osc_host", "127.0.0.1")),
        osc_port=int(data.get("osc_port", 4000)),
        write_file=bool(data.get("write_file", False)),
        filepath=data.get("filepath", None),
    )
    return jsonify({"status": "ok", "upmix_params": upmix_params})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
