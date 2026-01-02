"""
Modules de traitement audio pour l'algorithme upmix.
"""

from .biquad_filter import (
    BiquadFilter,
    CascadeBiquadFilter,
    FilterType,
    compute_biquad_coeffs,
)
from .crossover import (
    Crossover,
    apply_crossover,
    apply_crossover_multichannel,
    compute_lf_mono1_multichannel,
    compute_lf_mono1_stereo,
    sum_power_constant,
)
from .lfe_processor import (
    LFEProcessor,
    create_lfe_from_sum,
    detect_lfe_channels,
    extract_existing_lfe,
    get_non_lfe_channels,
    process_lfe,
)
from .re_model_light import (
    compute_energy_vector,
    energy_vector_to_angle,
    estimate_panning,
    estimate_source_width,
    get_available_layouts,
    get_energy_vector_magnitude,
    get_layout_info,
    get_speaker_unit_vectors,
)
from .stft_processor import STFTProcessor, compute_latency, create_sqrt_hann_window
