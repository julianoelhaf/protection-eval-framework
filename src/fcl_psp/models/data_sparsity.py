"""Window flattening plus optional sensor-*availability* degradation.

Two distinct responsibilities live here:

1. ``apply_sparsity_transform`` — the flattening every run goes through: it reshapes
   (N, L, F) windows into the time-major (N, L*F) matrix the classical estimators
   consume. This runs on the default path with no degradation applied.

2. The masking helpers — voltage-/current-only masks, relay, bus and phase failure,
   downsampling, and block zeroing across timesteps. These model **sensor
   availability** (channels missing or dropping out).

**Scope note.** The degradation axes in (2) are *not* part of the results reported in
the standardized-framework manuscript, which defers sensor-availability robustness to
the companion study and instead reports measurement-*fidelity* axes (additive noise,
CT saturation, synchronization jitter) — those live in ``fcl_psp.perturbation``. Every
option here is disabled by default (see ``config/data_sparsity/default.yaml``), so the
headline numbers take the no-degradation path.
"""

import logging
from typing import Tuple

import numpy as np
from psp_helper.config import MainConfig
from psp_helper.constants import BUS_TO_RELAY_MAPPING, CURRENT_CHANNELS, VOLTAGE_CHANNELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def apply_voltage_only_mask(reshaped_data: np.ndarray, config: MainConfig) -> np.ndarray:
    """
    Apply 'voltage_only' masking to the reshaped data.

    Args:
        reshaped_data (np.ndarray): The reshaped data to apply the mask to.
        config (MainConfig): Configuration object containing data sparsity settings.

    Returns:
        np.ndarray: The masked data with voltage features set to 0.
    """
    if config.data_sparsity.voltage_loss:
        _, _, n_features = reshaped_data.shape
        num_groups = n_features // 6  # Number of 6-channel groups

        voltage_indices = np.hstack(
            [np.array(VOLTAGE_CHANNELS) + 6 * i for i in range(num_groups)]
        )
        reshaped_data[:, :, voltage_indices] = 0  # ✅ Fixed indexing

        logger.debug(
            f"Applied 'voltage_only' masking to {len(voltage_indices)} features. Shape: {reshaped_data.shape}"
        )
    else:
        logger.debug("No 'voltage_only' masking applied.")

    return reshaped_data


def apply_current_only_mask(reshaped_data: np.ndarray, config: MainConfig) -> np.ndarray:
    """
    Apply 'current_only' masking to the reshaped data.

    Args:
        reshaped_data (np.ndarray): The reshaped data to apply the mask to.
        config (MainConfig): Configuration object containing data sparsity settings.

    Returns:
        np.ndarray: The masked data with current features set to 0.
    """
    if config.data_sparsity.current_loss:
        _, _, n_features = reshaped_data.shape
        num_groups = n_features // 6  # Number of 6-channel groups

        current_indices = np.hstack(
            [np.array(CURRENT_CHANNELS) + 6 * i for i in range(num_groups)]
        )
        reshaped_data[:, :, current_indices] = 0  # ✅ Fixed indexing

        logger.debug(
            f"Applied 'current_only' masking to {len(current_indices)} features. Shape: {reshaped_data.shape}"
        )
    else:
        logger.debug("No 'current_only' masking applied.")

    return reshaped_data


def simulate_protective_relay_failure(reshaped_data: np.ndarray, config: MainConfig) -> np.ndarray:
    """
    Simulate the complete failure of protective relays by zeroing out all associated measurements.

    This function sets to zero all voltage and current measurements (6 channels per relay)
    for specific relays, mimicking real-world relay failures.

    Args:
        reshaped_data (np.ndarray): The reshaped data of shape (n_samples, n_timesteps, n_features).
        config (MainConfig): Configuration object containing data sparsity settings.

    Returns:
        np.ndarray: The data with selected relay measurements set to 0.
    """
    if (
        len(config.data_sparsity.relay_failure_ids) == 1
        and config.data_sparsity.relay_failure_ids[0] == 0
    ):
        logger.debug("No relay failures applied.")
        return reshaped_data
    for relay_id in config.data_sparsity.relay_failure_ids:
        if relay_id == 0:
            continue
        # Each relay has 6 associated measurement channels: (3 voltages + 3 currents)
        start_channel = (relay_id - 1) * 6  # Relay IDs are 1-indexed
        end_channel = start_channel + 6
        reshaped_data[:, :, start_channel:end_channel] = 0  # Mask all timesteps

        logger.debug(
            f"Simulated failure of protective relay {relay_id}: "
            f"Set features {start_channel}-{end_channel-1} to zero "
            f"across all {reshaped_data.shape[1]} timesteps."
        )

    return reshaped_data


def apply_block_zeroing_across_timesteps(
    reshaped_data: np.ndarray, config: MainConfig
) -> np.ndarray:
    """
    Apply block-based zeroing across timesteps using a time duration instead of a ratio.

    This function simulates communication blackouts by setting **contiguous time windows**
    to zero across all features, based on a defined duration in seconds.

    Args:
        reshaped_data (np.ndarray): Input data of shape (n_samples, n_timesteps, n_features).
        config: Configuration object containing data sparsity settings.

    Returns:
        np.ndarray: Data with missing time segments across all features.

    Raises:
        ValueError: If `reshaped_data` is empty or `zeroing_duration_s` is invalid.
    """

    if reshaped_data.size == 0 or reshaped_data.ndim != 3:
        raise ValueError(
            "Input data must be a non-empty 3D array (n_samples, n_timesteps, n_features)."
        )

    # Extract parameters from config
    zeroing_duration_s = config.data_sparsity.zeroing_duration_s
    sampling_frequency = config.dataset.sampling_frequency  # Frequency in Hz (samples per second)

    if zeroing_duration_s <= 0:
        logger.debug("No block zeroing applied.")
        return reshaped_data

    # Convert time duration to number of timesteps
    n_timesteps_to_zero = int(zeroing_duration_s * sampling_frequency)
    logger.debug(f"Zeroing {n_timesteps_to_zero} timesteps ({zeroing_duration_s}s).")

    _, n_timesteps, _ = reshaped_data.shape

    if n_timesteps_to_zero == 0:
        logger.warning("Zeroing duration too short; no timesteps will be zeroed.")
        return reshaped_data

    # Ensure the zero block fits within the available timesteps
    if n_timesteps_to_zero >= n_timesteps:
        logger.warning(
            f"Zeroing duration ({zeroing_duration_s}s) exceeds or equals the number of available timesteps ({n_timesteps})."
        )
        raise ValueError("Zeroing duration exceeds available timesteps.")

    # Start_idx is after 10% of the data. Fixed for reproducibility.
    start_idx = int(n_timesteps * 0.1)

    reshaped_data[:, start_idx : start_idx + n_timesteps_to_zero, :] = 0

    logger.debug(
        f"Zeroed {n_timesteps_to_zero} timesteps (duration: {zeroing_duration_s}s) starting at index {start_idx}."
    )
    return reshaped_data


def downsample_data(data: np.ndarray, config: MainConfig) -> np.ndarray:
    """
    Downsample the data by a factor of `config.data_sparsity.downsampling_factor`.

    The function ensures that the downsampling factor is valid and that the number of timesteps
    is evenly divisible by the factor to prevent data misalignment.

    Args:
        data (np.ndarray): The data to downsample, shape (n_samples, n_timesteps, n_features).
        config (MainConfig): Configuration object containing data sparsity settings.

    Returns:
        np.ndarray: The downsampled data.

    Raises:
        ValueError: If `downsampling_factor` is not in a valid range or not divisible into `n_timesteps`.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D array (n_samples, n_timesteps, n_features).")

    _, n_timesteps, _ = data.shape
    downsample_factor = config.data_sparsity.downsampling_factor

    # Ensure downsampling_factor is an integer between 1 and n_timesteps
    if not isinstance(downsample_factor, int) or not (1 <= downsample_factor <= n_timesteps):
        raise ValueError(
            f"downsampling_factor must be an integer between 1 and {n_timesteps}, got {downsample_factor}."
        )

    # Ensure that the number of timesteps is evenly divisible by downsample_factor
    if n_timesteps % downsample_factor != 0:
        raise ValueError(
            f"n_timesteps ({n_timesteps}) must be divisible by downsampling_factor ({downsample_factor})."
        )

    # Apply downsampling
    if downsample_factor > 1:
        data = data[:, ::downsample_factor, :]
        logger.debug(
            f"Downsampled data by a factor of {downsample_factor}. New shape: {data.shape}"
        )
    else:
        logger.debug("No downsampling applied.")

    return data


def simulate_bus_failure(reshaped_data: np.ndarray, config: "MainConfig") -> np.ndarray:
    """
    Simulate the complete failure of a bus by zeroing out all associated measurements.

    This function sets to zero all voltage and current measurements (6 channels per bus)
    for specific buses, mimicking real-world bus failures.

    Args:
        reshaped_data (np.ndarray): The reshaped data of shape (n_samples, n_timesteps, n_features).
        config (MainConfig): Configuration object containing data sparsity settings.

    Returns:
        np.ndarray: The data with selected bus measurements set to 0.

    Raises:
        ValueError: If input data is not a 3D NumPy array.
        ValueError: If bus_failure_ids contain invalid bus IDs.
    """
    if reshaped_data.ndim != 3:
        raise ValueError("Input data must be a 3D array (n_samples, n_timesteps, n_features).")

    n_samples, n_timesteps, n_features = reshaped_data.shape

    if config.data_sparsity.bus_failure_id == 0:
        logger.debug("No bus failure applied.")
        return reshaped_data  # No bus failure applied

    bus_failure_id = config.data_sparsity.bus_failure_id

    # Ensure bus_failure_ids is a valid list of bus IDs
    if not bus_failure_id:
        raise ValueError(
            f"bus_failure_ids must be a non-empty list of valid bus IDs. Got: {bus_failure_id}"
        )

    if bus_failure_id not in BUS_TO_RELAY_MAPPING:
        raise ValueError(
            f"Invalid bus ID: {bus_failure_id}. Must be in {list(BUS_TO_RELAY_MAPPING.keys())}"
        )
    if max(BUS_TO_RELAY_MAPPING[bus_failure_id]) * 6 > n_features:
        raise ValueError(
            f"Relay mapping for bus ID {bus_failure_id} exceeds the number of features ({n_features})."
        )

    # Zero out measurements for all relays associated with the bus
    for relay_id in BUS_TO_RELAY_MAPPING[bus_failure_id]:
        start_channel = relay_id * 6
        end_channel = start_channel + 6

        # Ensure the selected feature indices are within bounds
        if end_channel > n_features:
            raise ValueError(
                f"Relay ID {relay_id} results in out-of-bounds feature indexing: {start_channel}-{end_channel} "
                f"(available features: {n_features}). Check BUS_TO_RELAY_MAPPING."
            )

        reshaped_data[:, :, start_channel:end_channel] = 0

    logger.debug(
        f"Simulated failure of bus {bus_failure_id}: "
        f"Set all associated relay features on relays {BUS_TO_RELAY_MAPPING[bus_failure_id]} to zero."
    )

    return reshaped_data


def simulate_phase_failure(reshaped_data: np.ndarray, config: "MainConfig") -> np.ndarray:
    """
    Simulate the complete failure of a channel by zeroing out all associated measurements.

    This function sets to zero all to voltage channel and current channel measurements set in config.data_sparsity.phase_failure_id,
    which corresponds to the phase of the signal. (A, B, C) or None.

    Args:
        reshaped_data (np.ndarray): The reshaped data of shape (n_samples, n_timesteps, n_features).
        config (MainConfig): Configuration object containing data sparsity settings.

    Returns:
        np.ndarray: The data with selected channel measurements set to 0.

    """

    if reshaped_data.ndim != 3:
        raise ValueError("Input data must be a 3D array (n_samples, n_timesteps, n_features).")

    n_samples, n_timesteps, n_features = reshaped_data.shape

    if config.data_sparsity.phase_failure_id == "None":
        logger.debug("No channel failure applied.")
        return reshaped_data  # No channel failure applied

    phase_failure_id = config.data_sparsity.phase_failure_id

    # Ensure phase_failure_id is a valid phase ID
    if phase_failure_id not in ["A", "B", "C"]:
        raise ValueError(f"Invalid phase ID: {phase_failure_id}. Must be in ['A', 'B', 'C']")

    # if A: then every 3rd channel is set to 0
    # if B: then every 3rd channel + 1 is set to 0
    # if C: then every 3rd channel + 2 is set to 0

    if phase_failure_id == "A":
        start_channel = 0
    elif phase_failure_id == "B":
        start_channel = 1
    elif phase_failure_id == "C":
        start_channel = 2

    for i in range(start_channel, n_features, 3):
        reshaped_data[:, :, i] = 0

    logger.debug(
        f"Simulated failure of phase {phase_failure_id}: "
        f"Set all associated relay features on phase {phase_failure_id} to zero."
    )

    return reshaped_data


def sparsity_is_disabled(config: MainConfig) -> bool:
    """True when every data-sparsity option sits at its documented 'disabled' sentinel.

    Mirrors the individual early-return guards of the helpers below, so the transform
    can skip copying entirely on the default (no-degradation) path. Keep in sync with
    ``config/data_sparsity/default.yaml``.
    """
    ds = config.data_sparsity
    relay_ids = list(ds.relay_failure_ids)
    return (
        not ds.voltage_loss
        and not ds.current_loss
        and int(ds.bus_failure_id) == 0
        and str(ds.phase_failure_id) == "None"
        and float(ds.zeroing_duration_s) <= 0.0
        and int(ds.downsampling_factor) <= 1
        and len(relay_ids) == 1
        and int(relay_ids[0]) == 0
    )


def apply_sparsity_transform(
    windows: np.ndarray, config: MainConfig
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Reshape 3D window data (samples, timesteps, features) into a 2D array suitable for model input.
    Additionally, implement data sparsity techniques by calling helper functions.

    Args:
        windows (np.ndarray): Input data of shape (n_samples, n_timesteps, n_features).
        config (MainConfig): Configuration object containing data sparsity settings.

    Returns:
        Tuple[np.ndarray, Tuple[int, int, int]]: Reshaped data and its final 3D shape.

    Note:
        ``windows`` is never modified. When no degradation is configured the returned
        array may be a *view* of the input (see ``sparsity_is_disabled``); every caller
        slices it with fancy indexing, which copies, before any in-place work.
    """
    n_samples, n_timesteps, n_features = windows.shape

    # Fast path: nothing would be masked, so do not materialize copies of what is
    # typically a multi-GB memmap (FC at 20 ms is ~209k x 128 x 48 float32 = ~5 GB).
    # Reshaping alone keeps peak memory at one lazily-read array instead of three.
    if sparsity_is_disabled(config):
        logger.debug("Data sparsity disabled; reshaping without copying.")
        flat = windows.reshape(n_samples, n_timesteps * n_features)
        logger.debug(f"Reshaped data to {flat.shape} (no sparsity techniques applied).")
        return flat, (n_samples, n_timesteps, n_features)

    original_zero_ratio = 1 - np.count_nonzero(windows) / windows.size
    logger.debug(f"Original zeroed out ratio: {original_zero_ratio:.2%}")

    # One writable copy: the helpers mask in place, and the input must stay intact.
    # Change detection uses the zero ratio below rather than a second full copy.
    reshaped_data = np.array(windows, copy=True)

    try:
        # Apply various data sparsity techniques based on config
        reshaped_data = downsample_data(reshaped_data, config)
        reshaped_data = apply_voltage_only_mask(reshaped_data, config)
        reshaped_data = apply_current_only_mask(reshaped_data, config)
        reshaped_data = simulate_protective_relay_failure(reshaped_data, config)
        reshaped_data = simulate_bus_failure(reshaped_data, config)
        reshaped_data = apply_block_zeroing_across_timesteps(reshaped_data, config)
        reshaped_data = simulate_phase_failure(reshaped_data, config)

        n_samples, n_timesteps, n_features = reshaped_data.shape
        new_shape = (n_samples, n_timesteps, n_features)
        logger.debug(f"Applied data sparsity techniques. New shape: {new_shape}.")

        reshaped_data = reshaped_data.reshape(n_samples, n_timesteps * n_features)
        logger.debug(f"Reshaped data to {reshaped_data.shape}.")

    except ValueError as e:
        # Fall back to the untouched input rather than a partially masked array.
        logger.error(f"Invalid data sparsity configuration: {e}")
        return (
            windows.reshape(n_samples, n_timesteps * n_features),
            (n_samples, n_timesteps, n_features),
        )

    # Log final zero ratio if changed
    zeroed_ratio = 1 - np.count_nonzero(reshaped_data) / reshaped_data.size
    change_ratio = zeroed_ratio - original_zero_ratio
    if change_ratio != 0:
        logger.debug(f"Final zeroed out ratio: {zeroed_ratio:.2%} (change: {change_ratio:.2%})")
    else:
        logger.debug(f"Final zeroed out ratio: {zeroed_ratio:.2%} (no change)")

    return reshaped_data, new_shape
