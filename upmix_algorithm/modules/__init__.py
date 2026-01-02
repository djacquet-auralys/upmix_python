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
from .extractor import (
    SourceExtractor,
    apply_mask_to_stft,
    extract_multiple_sources,
    extract_source,
    get_channel_angles,
    select_closest_channel,
)
from .lfe_processor import (
    LFEProcessor,
    create_lfe_from_sum,
    detect_lfe_channels,
    extract_existing_lfe,
    get_non_lfe_channels,
    process_lfe,
)
from .mask_generator import (
    RampSmooth,
    apply_freq_blur,
    apply_temporal_smoothing,
    create_mask_lut,
    generate_extraction_mask,
    interpolate_lut,
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
from .respatializer import (
    Respatializer,
    add_lf_mono1_to_sources,
    add_lfe_to_output,
    apply_delay,
    apply_gain_and_delay,
    compute_default_gains,
    get_available_output_layouts,
    get_output_layout_info,
    ms_to_samples,
    parse_source_params,
    spatialize_sources,
)
from .stft_processor import STFTProcessor, compute_latency, create_sqrt_hann_window
