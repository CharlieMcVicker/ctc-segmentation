#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020, Technische Universität München;
#                 Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Contribution: s920128 @ Github for prepare_tokenized_text()

"""CTC segmentation.

This file contains the core functions of CTC segmentation.
to extract utterance alignments within an audio file with
a given transcription.
For a description, see:
"CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition"
https://arxiv.org/abs/2007.09127 or
https://link.springer.com/chapter/10.1007%2F978-3-030-60276-5_27
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import warnings
import numpy as np

logger = logging.getLogger("ctc_segmentation")

# import for table of character probabilities mapped to time
try:
    from .ctc_segmentation_dyn import cython_fill_table
except ImportError:
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from .ctc_segmentation_dyn import cython_fill_table


class CtcSegmentationParameters:
    """Default values and configuration for CTC segmentation.

    May need adjustment according to localization or ASR settings.
    The character set is taken from the model dict, i.e., usually are generated
    with SentencePiece. An ASR model trained in the corresponding language and
    character set is needed. If the character set contains any punctuation
    characters, "#", the Greek char "ε", or the space placeholder, adapt
    these settings.

    Attributes:
        max_prob: Lower bound probability floor for invalid transitions (default: -1e10).
        skip_prob: Probability floor for skipped ground truth transitions (default: -1e10).
        min_window_size: Minimum window size considered for alignment trellis (default: 8000).
        max_window_size: Maximum window size before raising alignment failure (default: 100000).
        index_duration: Time duration of one CTC frame index in seconds (default: 0.025).
        score_min_mean_over_L: Window length L for utterance confidence calculation (default: 30).
        space: Character used to represent word delimiter spaces (default: "·").
        blank: Vocabulary index of the blank token (default: 0).
        replace_spaces_with_blanks: Replace space characters with blanks (default: False).
        blank_transition_cost_zero: Allow free stay transitions on blank (default: False).
        preamble_transition_cost_zero: Allow free stay transitions in preamble (default: True).
        backtrack_from_max_t: Backtrack from the last frame instead of argmax (default: False).
        self_transition: Symbol representing stay transitions in state list (default: "ε").
        start_of_ground_truth: Symbol indicating start of ground truth sequence (default: "#").
        excluded_characters: String of punctuation characters excluded during text preparation.
        tokenized_meta_symbol: SentencePiece/BPE prefix symbol (default: "▁").
        char_list: Vocabulary list or mapping of character tokens (default: None).
        syncope_penalty: Log-space penalty subtracted for syncope skip transitions (default: 0.25).
        syncope_tokens: Sequence of tokens (strings or integer IDs) that can be skipped via syncope (default: None).
        is_syncope_token: Optional sequence or 1D mask marking syncope token positions in ground truth for
            blank-constrained syncope transitions (default: None). Enforces the Consonant Preservation Invariant
            and Blank-Only Stride Invariants (Case A and Case B).
        intrusive_tokens: Sequence of tokens (strings or integer IDs) permitted as intrusive acoustic
            detours between ground-truth label transitions (e.g. epenthesis, aspiration, glottal stops) (default: None).
        intrusive_penalty: Default log-space penalty subtracted from intrusive detour transitions (default: 0.1).
        intrusive_penalties: Uniform float penalty or per-token dictionary (e.g. `{"h": 0.15, "'": 0.3}`) mapping
            token strings or integer IDs to individual transition penalties (default: None).
        intrusive_min_logprobs: Uniform float or per-token dictionary (e.g. `{"h": -1.2, "'": -0.7}`) specifying
            minimum log-probability thresholds (or linear probabilities in (0, 1]) gating intrusive candidate
            evaluation to filter diffuse noise while retaining sharp transient events (default: None).
        intrusive_max_stride: Maximum blank frame stride permitted for blank-tolerant intrusive transitions (default: 4).
        is_intrusive_site: Optional sequence or 1D mask of size len(ground_truth) restricting intrusive detour
            transitions strictly to licensed ground-truth positions (default: None).
        is_intrusive_token: 1D mask across vocabulary indices marking eligible intrusive tokens (default: None).
    """

    max_prob: float = -10000000000.0
    skip_prob: float = -10000000000.0
    min_window_size: int = 8000
    max_window_size: int = 100000
    index_duration: float = 0.025
    score_min_mean_over_L: int = 30
    space: str = "·"
    blank: int = 0
    replace_spaces_with_blanks: bool = False
    blank_transition_cost_zero: bool = False
    preamble_transition_cost_zero: bool = True
    backtrack_from_max_t: bool = False
    self_transition: str = "ε"
    start_of_ground_truth: str = "#"
    excluded_characters: str = ".,»«•❍·"
    tokenized_meta_symbol: str = "▁"
    char_list: Optional[Union[List[str], Tuple[str, ...], Set[str], Sequence[str]]] = None
    syncope_penalty: float = 0.25
    syncope_tokens: Optional[Sequence[Union[str, int]]] = None
    is_syncope_token: Optional[Union[Sequence[Union[bool, int]], np.ndarray]] = None
    # Intrusive Token Subsystem
    intrusive_tokens: Optional[Sequence[Union[str, int]]] = None
    intrusive_penalty: float = 0.1
    intrusive_penalties: Optional[Union[float, Dict[Union[str, int], float]]] = None
    intrusive_min_logprobs: Optional[Union[float, Dict[Union[str, int], float]]] = None
    _intrusive_max_stride: int = 4
    is_intrusive_site: Optional[Union[Sequence[Union[bool, int]], np.ndarray]] = None
    is_intrusive_token: Optional[np.ndarray] = None
    # legacy Parameters (deprecated, use index_duration instead)
    _subsampling_factor: Optional[Union[int, float]] = None
    _frame_duration_ms: Optional[Union[int, float]] = None

    @property
    def intrusive_max_stride(self) -> int:
        """Maximum blank frame stride permitted for intrusive token transitions."""
        return self._intrusive_max_stride

    @intrusive_max_stride.setter
    def intrusive_max_stride(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(
                f"intrusive_max_stride must be an integer, got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(
                f"intrusive_max_stride must be a non-negative integer, got {value}"
            )
        self._intrusive_max_stride = int(value)

    @property
    def syncopy_penalty(self) -> float:
        """Deprecated alias for syncope_penalty.

        .. deprecated::
            Use `syncope_penalty` instead.
        """
        warnings.warn(
            "syncopy_penalty is deprecated and will be removed in a future version. "
            "Use syncope_penalty instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.syncope_penalty

    @syncopy_penalty.setter
    def syncopy_penalty(self, value: float) -> None:
        warnings.warn(
            "syncopy_penalty is deprecated and will be removed in a future version. "
            "Use syncope_penalty instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.syncope_penalty = value

    @property
    def optional_vowel_tokens(self) -> Optional[Sequence[Union[str, int]]]:
        """Deprecated alias for syncope_tokens.

        .. deprecated::
            Use `syncope_tokens` instead.
        """
        warnings.warn(
            "optional_vowel_tokens is deprecated and will be removed in a future version. "
            "Use syncope_tokens instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.syncope_tokens

    @optional_vowel_tokens.setter
    def optional_vowel_tokens(self, value: Optional[Sequence[Union[str, int]]]) -> None:
        warnings.warn(
            "optional_vowel_tokens is deprecated and will be removed in a future version. "
            "Use syncope_tokens instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.syncope_tokens = value

    @property
    def is_optional_vowel(self) -> Optional[np.ndarray]:
        """Deprecated alias for is_syncope_token.

        .. deprecated::
            Use `is_syncope_token` instead.
        """
        warnings.warn(
            "is_optional_vowel is deprecated and will be removed in a future version. "
            "Use is_syncope_token instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_syncope_token

    @is_optional_vowel.setter
    def is_optional_vowel(self, value: Optional[np.ndarray]) -> None:
        warnings.warn(
            "is_optional_vowel is deprecated and will be removed in a future version. "
            "Use is_syncope_token instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.is_syncope_token = value

    @property
    def subsampling_factor(self) -> Optional[Union[int, float]]:
        """Legacy parameter.

        .. deprecated::
            Use `index_duration` instead.
        """
        warnings.warn(
            "subsampling_factor is deprecated and will be removed in a future version. "
            "Use index_duration instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._subsampling_factor

    @subsampling_factor.setter
    def subsampling_factor(self, value: Optional[Union[int, float]]) -> None:
        warnings.warn(
            "subsampling_factor is deprecated and will be removed in a future version. "
            "Use index_duration instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._subsampling_factor = value
        if self._subsampling_factor is not None and self._frame_duration_ms is not None:
            self.index_duration = self._frame_duration_ms * self._subsampling_factor / 1000.0

    @property
    def frame_duration_ms(self) -> Optional[Union[int, float]]:
        """Legacy parameter.

        .. deprecated::
            Use `index_duration` instead.
        """
        warnings.warn(
            "frame_duration_ms is deprecated and will be removed in a future version. "
            "Use index_duration instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._frame_duration_ms

    @frame_duration_ms.setter
    def frame_duration_ms(self, value: Optional[Union[int, float]]) -> None:
        warnings.warn(
            "frame_duration_ms is deprecated and will be removed in a future version. "
            "Use index_duration instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._frame_duration_ms = value
        if self._subsampling_factor is not None and self._frame_duration_ms is not None:
            self.index_duration = self._frame_duration_ms * self._subsampling_factor / 1000.0

    @property
    def index_duration_in_seconds(self) -> float:
        """Derive index duration from frame duration and subsampling.

        Legacy property. This property will be removed in later versions
        and replaced by index_duration.
        """
        warnings.warn(
            "index_duration_in_seconds is deprecated and will be removed in a future version. "
            "Use index_duration instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.index_duration

    @index_duration_in_seconds.setter
    def index_duration_in_seconds(self, value: float) -> None:
        warnings.warn(
            "index_duration_in_seconds is deprecated and will be removed in a future version. "
            "Use index_duration instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.index_duration = value

    @property
    def flags(self) -> int:
        """Get configuration flags to pass to the table_fill operation."""
        flags = int(self.blank_transition_cost_zero)
        flags += 2 * int(self.preamble_transition_cost_zero)
        return flags

    def update_excluded_characters(self) -> None:
        """Remove known tokens from the list of excluded characters."""
        if self.char_list is not None:
            self.excluded_characters = "".join(
                [
                    char
                    for char in self.excluded_characters
                    if char not in self.char_list
                ]
            )
        logger.debug(f"Excluded characters: {self.excluded_characters}")

    def __init__(self, **kwargs: Any) -> None:
        """Set all parameters as attribute at init."""
        self.set(**kwargs)

    def set(self, **kwargs: Any) -> None:
        """Update CtcSegmentationParameters.

        Args:
            **kwargs: Key-value dict that contains all properties
                with their new values. Unknown properties are ignored.
        """
        for key in kwargs:
            if (
                not key.startswith("_")
                and hasattr(self, key)
                and kwargs[key] is not None
            ):
                setattr(self, key, kwargs[key])

    def __repr__(self) -> str:
        """Print all attribute as dictionary."""
        output = "CtcSegmentationParameters( "
        for attribute, value in self.__dict__.items():
            if attribute.startswith("_"):
                continue
            output += f"{attribute}={value}, "
        output += ")"
        return output


@dataclass(frozen=True)
class TrellisRuntimeContext:
    """Internal compiled runtime context passed to Cython core and backtracking.

    Maintains the architectural separation between the high-level, human-friendly
    `CtcSegmentationParameters` userspace configuration object and the low-level,
    C-contiguous NumPy arrays required for zero-overhead Cython DP trellis filling
    and backtracking.

    Attributes:
        is_syncope_token: 1D `int8` array of shape `(L,)` indicating which ground-truth
            states are eligible for blank-constrained syncope skip transitions.
        syncope_penalty: Float log-space penalty subtracted when taking a syncope skip.
        intrusive_token_ids: 1D `int64` array of shape `(K,)` with vocabulary token IDs
            of eligible intrusive detour tokens.
        intrusive_penalties: 1D `float32` array of shape `(K,)` with per-token log-space
            transition penalties subtracted when taking an intrusive detour.
        intrusive_min_logprobs: 1D `float32` array of shape `(K,)` with per-token minimum
            log-probability thresholds gating intrusive candidate evaluation.
        is_intrusive_site: 1D `int8` array of shape `(L,)` (or empty if unrestricted)
            restricting intrusive detour arrival states to licensed ground-truth positions.
        intrusive_max_stride: Integer maximum blank frame stride bridging intrusive tokens.
        blank: Integer vocabulary index of the CTC blank / PAD token.
        flags: Packed integer bitmask of runtime configuration flags.
    """

    # Ground truth & syncope arrays
    is_syncope_token: np.ndarray  # 1D np.int8 array of shape (L,)
    syncope_penalty: float

    # Intrusive detour arrays
    intrusive_token_ids: np.ndarray  # 1D np.int64 array of shape (K,)
    intrusive_penalties: np.ndarray  # 1D np.float32 array of shape (K,)
    intrusive_min_logprobs: np.ndarray  # 1D np.float32 array of shape (K,)
    is_intrusive_site: np.ndarray  # 1D np.int8 array of shape (L,)
    intrusive_max_stride: int

    blank: int
    flags: int

    @classmethod
    def compile(
        cls,
        config: CtcSegmentationParameters,
        ground_truth: np.ndarray,
    ) -> "TrellisRuntimeContext":
        """Compile and validate user configuration into C-contiguous NumPy runtime arrays.

        Validates all user-specified syncope and intrusive token configurations,
        resolves token strings against `char_list`, normalizes linear probabilities
        in (0, 1] to log-space thresholds, and builds contiguous arrays ready for
        the Cython core.

        Args:
            config: An instance of `CtcSegmentationParameters`.
            ground_truth: 2D label matrix of shape `(L, max_char_len)`.

        Returns:
            An immutable `TrellisRuntimeContext` instance containing compiled arrays.

        Raises:
            ValueError: If mask lengths do not match ground truth length, or token
                strings are not present in `char_list`.
            TypeError: If configuration parameters have invalid types.
        """
        num_cols = ground_truth.shape[0]

        # 1. Compile syncope mask
        if config.is_syncope_token is not None:
            syncope_mask = np.ascontiguousarray(config.is_syncope_token, dtype=np.int8)
            if syncope_mask.ndim != 1 or syncope_mask.shape[0] != num_cols:
                raise ValueError(
                    f"is_syncope_token length ({syncope_mask.shape[0]}) does not match "
                    f"ground_truth length ({num_cols})"
                )
        else:
            syncope_mask = np.zeros((0,), dtype=np.int8)

        # 2. Compile intrusive token IDs
        intrusive_ids = []
        if config.intrusive_tokens:
            for token in config.intrusive_tokens:
                if isinstance(token, str):
                    if config.char_list is None or token not in config.char_list:
                        raise ValueError(f"Intrusive token '{token}' not found in char_list")
                    char_list_seq = (
                        config.char_list
                        if isinstance(config.char_list, list)
                        else list(config.char_list)
                    )
                    intrusive_ids.append(char_list_seq.index(token))
                elif isinstance(token, (int, np.integer)) and not isinstance(token, bool):
                    intrusive_ids.append(int(token))
                else:
                    raise TypeError(f"Invalid intrusive token type: {type(token)}")
        intrusive_token_ids = np.ascontiguousarray(np.array(intrusive_ids, dtype=np.int64))
        K = len(intrusive_token_ids)

        # 3. Compile per-token penalties
        penalties_arr = np.zeros(K, dtype=np.float32)
        default_penalty = float(getattr(config, "intrusive_penalty", 0.1))
        if K > 0:
            if config.intrusive_penalties is None:
                penalties_arr.fill(default_penalty)
            elif isinstance(config.intrusive_penalties, (int, float, np.floating)) and not isinstance(
                config.intrusive_penalties, bool
            ):
                penalties_arr.fill(float(config.intrusive_penalties))
            elif isinstance(config.intrusive_penalties, dict):
                char_list_seq = (
                    list(config.char_list)
                    if config.char_list is not None
                    else None
                )
                for idx, t_id in enumerate(intrusive_token_ids):
                    t_str = (
                        char_list_seq[t_id]
                        if char_list_seq and 0 <= t_id < len(char_list_seq)
                        else None
                    )
                    if t_id in config.intrusive_penalties:
                        penalties_arr[idx] = float(config.intrusive_penalties[t_id])
                    elif t_str is not None and t_str in config.intrusive_penalties:
                        penalties_arr[idx] = float(config.intrusive_penalties[t_str])
                    else:
                        penalties_arr[idx] = default_penalty
            else:
                raise TypeError(
                    f"Invalid intrusive_penalties type: {type(config.intrusive_penalties)}"
                )

        # 4. Compile per-token min log-probability thresholds
        min_lpz_arr = np.full(K, -np.inf, dtype=np.float32)
        if K > 0 and config.intrusive_min_logprobs is not None:
            if isinstance(config.intrusive_min_logprobs, (int, float, np.floating)) and not isinstance(
                config.intrusive_min_logprobs, bool
            ):
                val = float(config.intrusive_min_logprobs)
                log_val = np.log(val) if 0.0 < val <= 1.0 else val
                min_lpz_arr.fill(log_val)
            elif isinstance(config.intrusive_min_logprobs, dict):
                char_list_seq = (
                    list(config.char_list)
                    if config.char_list is not None
                    else None
                )
                for idx, t_id in enumerate(intrusive_token_ids):
                    t_str = (
                        char_list_seq[t_id]
                        if char_list_seq and 0 <= t_id < len(char_list_seq)
                        else None
                    )
                    val = None
                    if t_id in config.intrusive_min_logprobs:
                        val = config.intrusive_min_logprobs[t_id]
                    elif t_str is not None and t_str in config.intrusive_min_logprobs:
                        val = config.intrusive_min_logprobs[t_str]
                    if val is not None:
                        val = float(val)
                        min_lpz_arr[idx] = np.log(val) if 0.0 < val <= 1.0 else val
            else:
                raise TypeError(
                    f"Invalid intrusive_min_logprobs type: {type(config.intrusive_min_logprobs)}"
                )

        # 5. Compile intrusive site mask
        if config.is_intrusive_site is not None:
            site_mask = np.ascontiguousarray(config.is_intrusive_site, dtype=np.int8)
            if site_mask.ndim != 1 or site_mask.shape[0] != num_cols:
                raise ValueError(
                    f"is_intrusive_site length ({site_mask.shape[0]}) does not match "
                    f"ground_truth length ({num_cols})"
                )
        else:
            site_mask = np.zeros((0,), dtype=np.int8)

        return cls(
            is_syncope_token=syncope_mask,
            syncope_penalty=float(config.syncope_penalty),
            intrusive_token_ids=intrusive_token_ids,
            intrusive_penalties=penalties_arr,
            intrusive_min_logprobs=min_lpz_arr,
            is_intrusive_site=site_mask,
            intrusive_max_stride=int(config.intrusive_max_stride),
            blank=int(config.blank),
            flags=int(config.flags),
        )


def ctc_segmentation(
    config: CtcSegmentationParameters,
    lpz: np.ndarray,
    ground_truth: np.ndarray,
    is_syncope_token: Optional[Union[np.ndarray, Sequence[int]]] = None,
    is_optional_vowel: Optional[Union[np.ndarray, Sequence[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract character-level utterance alignments using dynamic programming.

    Aligns CTC posterior probabilities `lpz` with ground-truth label matrix
    `ground_truth`. Supports blank-constrained syncope-skip transitions (for optional
    token omission while strictly enforcing the Consonant Preservation Invariant and
    Blank-Only Stride Invariants across blank/PAD states) and blank-tolerant intrusive
    detour transitions (for acoustic surface token insertions like aspiration, epenthesis,
    or glottal stops with per-token transition penalties, minimum log-probability threshold
    gating, and phonotactic site masking).

    Args:
        config: An instance of `CtcSegmentationParameters` configuring trellis alignment,
            windowing, syncope penalties, intrusive token penalties, min-logprob gating,
            and stride tolerances.
        lpz: Log-probabilities obtained from CTC output of shape `(time_frames, vocab_size)`.
        ground_truth: Ground truth label matrix of shape `(labels_len, max_char_len)`.
        is_syncope_token: Optional 1D mask marking optional syncope token positions in
            ground truth for blank-constrained syncope skips.
        is_optional_vowel: Deprecated alias for `is_syncope_token`.

    Returns:
        A tuple of `(timings, char_probs, state_list)`:
            - timings: 1D array of shape `(labels_len,)` with aligned character start times in seconds.
            - char_probs: 1D array of shape `(time_frames,)` with frame-wise aligned log probabilities.
            - state_list: List of strings of length `time_frames` with aligned character/token symbols
              (including ground truth tokens, intrusive tokens, and stay/self transitions).

    Raises:
        AssertionError: If audio is shorter than text or alignment fails.
        ValueError: If configuration parameters or mask dimensions are invalid.
    """
    if is_syncope_token is None and is_optional_vowel is not None:
        warnings.warn(
            "is_optional_vowel is deprecated and will be removed in a future version. "
            "Use is_syncope_token instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        is_syncope_token = is_optional_vowel

    if is_syncope_token is not None:
        config.is_syncope_token = is_syncope_token

    blank = config.blank
    offset = 0
    audio_duration = lpz.shape[0] * config.index_duration
    logger.info(
        f"CTC segmentation of {len(ground_truth)} chars "
        f"to {audio_duration:.2f}s audio "
        f"({lpz.shape[0]} indices)."
    )
    if len(ground_truth) > lpz.shape[0] and config.skip_prob <= config.max_prob:
        raise AssertionError("Audio is shorter than text!")

    runtime_ctx = TrellisRuntimeContext.compile(config, ground_truth)

    window_size = config.min_window_size
    # Try multiple window lengths if it fails
    while True:
        # Create table of alignment probabilities
        table = np.zeros(
            [min(window_size, lpz.shape[0]), len(ground_truth)], dtype=np.float32
        )
        table.fill(config.max_prob)
        # Use array to log window offsets per character
        offsets = np.zeros([len(ground_truth)], dtype=np.int64)
        # Run actual alignment of utterances
        t, c = cython_fill_table(
            table,
            lpz.astype(np.float32),
            np.array(ground_truth, dtype=np.int64),
            offsets,
            runtime_ctx.is_syncope_token,
            runtime_ctx.syncope_penalty,
            runtime_ctx.intrusive_token_ids,
            runtime_ctx.intrusive_penalties,
            runtime_ctx.intrusive_min_logprobs,
            runtime_ctx.is_intrusive_site,
            runtime_ctx.intrusive_max_stride,
            runtime_ctx.blank,
            runtime_ctx.flags,
        )
        if config.backtrack_from_max_t:
            t = table.shape[0] - 1
        logger.debug(
            f"Max. joint probability to align text to audio: "
            f"{table[:, c].max()} at time index {t}"
        )
        # Backtracking
        timings = np.zeros([len(ground_truth)])
        char_probs = np.zeros([lpz.shape[0]])
        state_list = [""] * lpz.shape[0]
        try:
            # Do until start is reached
            while t != 0 or c != 0:
                # Calculate the possible transition probs towards the current cell
                min_s = None
                min_switch_prob_delta = np.inf
                max_lpz_prob = config.max_prob
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] != -1:
                        offset = offsets[c] - (offsets[c - 1 - s] if c - s > 0 else 0)
                        switch_prob = (
                            lpz[t + offsets[c], ground_truth[c, s]]
                            if c > 0
                            else config.max_prob
                        )
                        est_switch_prob = table[t, c] - table[t - 1 + offset, c - 1 - s]
                        if abs(switch_prob - est_switch_prob) < min_switch_prob_delta:
                            min_switch_prob_delta = abs(switch_prob - est_switch_prob)
                            min_s = s
                        max_lpz_prob = max(max_lpz_prob, switch_prob)
                stay_prob = (
                    max(lpz[t + offsets[c], blank], max_lpz_prob)
                    if t > 0
                    else config.max_prob
                )
                est_stay_prob = table[t, c] - table[t - 1, c]
                stay_prob_delta = abs(stay_prob - est_stay_prob)

                # Check syncope skip transitions with blank constraints
                min_syncope_skip_delta = np.inf
                best_syncope_c_prev = None
                best_syncope_s = None
                if runtime_ctx.is_syncope_token.shape[0] > 0 and c >= 2:
                    for s in range(ground_truth.shape[1]):
                        if ground_truth[c, s] != -1:
                            if runtime_ctx.is_syncope_token[c - 1] == 1:
                                # Valid candidate predecessors
                                candidates = [c - 2]
                                if c >= 3 and (ground_truth[c - 2, 0] == blank or ground_truth[c - 2, 0] == -1):
                                    candidates.append(c - 3)

                                for c_prev in candidates:
                                    delta_offset = offsets[c] - offsets[c_prev]
                                    t_prev = t - 1 + delta_offset
                                    if 0 <= t_prev < table.shape[0]:
                                        est_v_prob = table[t, c] - table[t_prev, c_prev]
                                        expected_v_prob = (
                                            lpz[t + offsets[c], ground_truth[c, s]]
                                            - runtime_ctx.syncope_penalty
                                        )
                                        v_delta = abs(est_v_prob - expected_v_prob)
                                        if v_delta < min_syncope_skip_delta:
                                            min_syncope_skip_delta = v_delta
                                            best_syncope_c_prev = c_prev
                                            best_syncope_s = s
                            elif (
                                c >= 3
                                and runtime_ctx.is_syncope_token[c - 2] == 1
                                and (
                                    ground_truth[c - 1, 0] == blank
                                    or ground_truth[c - 1, 0] == -1
                                )
                            ):
                                candidates = [c - 3]
                                if c >= 4 and (ground_truth[c - 3, 0] == blank or ground_truth[c - 3, 0] == -1):
                                    candidates.append(c - 4)

                                for c_prev in candidates:
                                    delta_offset = offsets[c] - offsets[c_prev]
                                    t_prev = t - 1 + delta_offset
                                    if 0 <= t_prev < table.shape[0]:
                                        est_v_prob = table[t, c] - table[t_prev, c_prev]
                                        expected_v_prob = (
                                            lpz[t + offsets[c], ground_truth[c, s]]
                                            - runtime_ctx.syncope_penalty
                                        )
                                        v_delta = abs(est_v_prob - expected_v_prob)
                                        if v_delta < min_syncope_skip_delta:
                                            min_syncope_skip_delta = v_delta
                                            best_syncope_c_prev = c_prev
                                            best_syncope_s = s

                # Check intrusive detour transitions
                min_intrusive_delta = np.inf
                best_intrusive_j = None
                best_intrusive_s = None
                best_intrusive_stride = 0
                if (
                    c >= 1
                    and t >= 2
                    and len(runtime_ctx.intrusive_token_ids) > 0
                    and (runtime_ctx.is_intrusive_site.shape[0] == 0 or runtime_ctx.is_intrusive_site[c] == 1)
                ):
                    for s in range(ground_truth.shape[1]):
                        if ground_truth[c, s] == -1 or c - 1 - s < 0:
                            continue
                        offset = offsets[c] - offsets[c - 1 - s]
                        for delta in range(0, min(runtime_ctx.intrusive_max_stride + 1, t - 1)):
                                t_prev = t - 2 - delta + offset
                                if 0 <= t_prev < table.shape[0]:
                                    blank_sum = 0.0
                                    for b_t in range(t - delta, t):
                                        blank_sum += lpz[b_t + offsets[c], blank]
                                    for j_idx, j in enumerate(runtime_ctx.intrusive_token_ids):
                                        t_detour_frame = t - 1 - delta + offsets[c]
                                        if lpz[t_detour_frame, j] >= runtime_ctx.intrusive_min_logprobs[j_idx]:
                                            expected_p = (
                                                table[t_prev, c - 1 - s]
                                                + lpz[t_detour_frame, j]
                                                - runtime_ctx.intrusive_penalties[j_idx]
                                                + blank_sum
                                                + lpz[t + offsets[c], ground_truth[c, s]]
                                            )
                                            diff = abs(table[t, c] - expected_p)
                                            if diff < min_intrusive_delta:
                                                min_intrusive_delta = diff
                                                best_intrusive_j = j
                                                best_intrusive_s = s
                                                best_intrusive_stride = delta

                # Check which transition has been taken
                if (
                    min_syncope_skip_delta < min_switch_prob_delta
                    and min_syncope_skip_delta < stay_prob_delta
                    and min_syncope_skip_delta < min_intrusive_delta
                    and best_syncope_c_prev is not None
                ):
                    # Apply reverse syncope skip transition
                    if c > 0:
                        timings[c] = (offsets[c] + t) * config.index_duration
                        char_probs[offsets[c] + t] = lpz[
                            t + offsets[c], ground_truth[c, best_syncope_s]
                        ]
                        char_index = ground_truth[c, best_syncope_s]
                        state_list[offsets[c] + t] = config.char_list[char_index]
                    offset = offsets[c] - offsets[best_syncope_c_prev]
                    c = best_syncope_c_prev
                    t -= 1 - offset
                elif (
                    min_intrusive_delta < min_switch_prob_delta
                    and min_intrusive_delta < stay_prob_delta
                    and best_intrusive_j is not None
                ):
                    # Apply reverse intrusive detour transition
                    if c > 0:
                        for s_idx in range(0, best_intrusive_s + 1):
                            timings[c - s_idx] = (
                                offsets[c] + t
                            ) * config.index_duration
                        char_index = ground_truth[c, best_intrusive_s]
                        char_probs[offsets[c] + t] = lpz[
                            t + offsets[c], char_index
                        ]
                        state_list[offsets[c] + t] = config.char_list[char_index]
                        for b_t in range(t - best_intrusive_stride, t):
                            char_probs[offsets[c] + b_t] = lpz[
                                offsets[c] + b_t, blank
                            ]
                            state_list[offsets[c] + b_t] = config.self_transition
                        char_probs[offsets[c] + t - 1 - best_intrusive_stride] = lpz[
                            t - 1 - best_intrusive_stride + offsets[c], best_intrusive_j
                        ]
                        state_list[offsets[c] + t - 1 - best_intrusive_stride] = config.char_list[best_intrusive_j]
                    offset = offsets[c] - (offsets[c - 1 - best_intrusive_s] if c - 1 - best_intrusive_s >= 0 else 0)
                    c -= 1 + best_intrusive_s
                    t -= (2 + best_intrusive_stride) - offset
                elif stay_prob_delta > min_switch_prob_delta:
                    # Apply reverse switch transition
                    if c > 0:
                        # Log timing and character - frame alignment
                        for s in range(0, min_s + 1):
                            timings[c - s] = (
                                offsets[c] + t
                            ) * config.index_duration
                        char_probs[offsets[c] + t] = max_lpz_prob
                        char_index = ground_truth[c, min_s]
                        state_list[offsets[c] + t] = config.char_list[char_index]
                    # Update column index and delta t
                    offset = offsets[c] - (offsets[c - 1 - min_s] if c - min_s > 0 else 0)
                    c -= 1 + min_s
                    t -= 1 - offset
                else:
                    # Apply reverse stay transition
                    char_probs[offsets[c] + t] = stay_prob
                    state_list[offsets[c] + t] = config.self_transition
                    t -= 1
        except IndexError:
            logger.debug(f"Failed to backtrack table with size {table.shape}")
            if window_size >= config.max_window_size or window_size >= lpz.shape[0]:
                raise AssertionError("Alignment failed!")
            window_size *= 2
            continue
        break
    return timings, char_probs, state_list


def _create_syncope_mask(
    config: CtcSegmentationParameters,
    ground_truth: Union[Sequence[Any], np.ndarray],
    is_token_ids: bool = False,
) -> np.ndarray:
    """Create a 1D int8 mask indicating syncope token positions in ground_truth."""
    mask = np.zeros(len(ground_truth), dtype=np.int8)
    tokens = config.syncope_tokens
    if tokens is None:
        return mask
    syncope_set = set(tokens)
    for idx, item in enumerate(ground_truth):
        if is_token_ids:
            if item in syncope_set:
                mask[idx] = 1
            elif (
                config.char_list is not None
                and isinstance(item, (int, np.integer))
                and 0 <= item < len(config.char_list)
                and config.char_list[item] in syncope_set
            ):
                mask[idx] = 1
        else:
            if item in syncope_set:
                mask[idx] = 1
            elif (
                config.char_list is not None
                and item in config.char_list
                and config.char_list.index(item) in syncope_set
            ):
                mask[idx] = 1
    return mask


def _create_optional_vowel_mask(
    config: CtcSegmentationParameters,
    ground_truth: Union[Sequence[Any], np.ndarray],
    is_token_ids: bool = False,
) -> np.ndarray:
    """Deprecated alias for _create_syncope_mask."""
    warnings.warn(
        "_create_optional_vowel_mask is deprecated and will be removed in a future version. "
        "Use _create_syncope_mask instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _create_syncope_mask(config, ground_truth, is_token_ids=is_token_ids)


def _parse_intrusive_token_ids(
    tokens: Optional[Sequence[Union[str, int]]],
    char_list: Optional[Union[List[str], Tuple[str, ...], Set[str], Sequence[str]]] = None,
) -> List[int]:
    """Parse and validate intrusive token specifications (strings or integer IDs)."""
    if tokens is None:
        return []
    c_list = list(char_list) if char_list is not None else None
    token_ids = []
    for t_item in tokens:
        if isinstance(t_item, str):
            if c_list is not None and t_item in c_list:
                token_ids.append(c_list.index(t_item))
            else:
                raise ValueError(f"Intrusive token '{t_item}' not found in char_list")
        elif isinstance(t_item, (int, np.integer)):
            t_int = int(t_item)
            if c_list is not None and not (0 <= t_int < len(c_list)):
                raise ValueError(
                    f"Intrusive token id {t_int} out of range for char_list of length {len(c_list)}"
                )
            if t_int < 0:
                raise ValueError(
                    f"Intrusive token id must be non-negative, got {t_int}"
                )
            token_ids.append(t_int)
        else:
            raise TypeError(f"Unsupported intrusive token type: {type(t_item)}")
    return token_ids


def _create_intrusive_mask(
    config: CtcSegmentationParameters,
    char_list: Optional[Sequence[str]] = None,
) -> Optional[np.ndarray]:
    """Create a 1D int8 mask indicating intrusive token indices in vocabulary (char_list)."""
    c_list = char_list if char_list is not None else config.char_list
    if config.intrusive_tokens is None or c_list is None:
        return None
    ids = _parse_intrusive_token_ids(config.intrusive_tokens, c_list)
    mask = np.zeros(len(c_list), dtype=np.int8)
    for idx in ids:
        mask[idx] = 1
    return mask


def _get_intrusive_token_ids(
    config: CtcSegmentationParameters,
) -> np.ndarray:
    """Extract and validate intrusive token IDs from configuration."""
    if config.intrusive_tokens is not None:
        token_ids = _parse_intrusive_token_ids(
            config.intrusive_tokens, config.char_list
        )
        return np.array(token_ids, dtype=np.int64)
    elif config.is_intrusive_token is not None:
        return np.where(
            np.asarray(config.is_intrusive_token, dtype=np.int8) == 1
        )[0].astype(np.int64)
    return np.zeros(0, dtype=np.int64)


def prepare_text(
    config: CtcSegmentationParameters,
    text: Sequence[str],
    char_list: Optional[Union[List[str], Sequence[str]]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Prepare the given text for CTC segmentation.

    Creates a matrix of character symbols to represent the given text,
    then creates list of char indices depending on the models char list.

    :param config: an instance of CtcSegmentationParameters
    :param text: iterable of utterance transcriptions
    :param char_list: a set or list that includes all characters/symbols,
                        characters not included in this list are ignored
    :return: label matrix, character index matrix
    """
    # temporary compatibility fix for previous espnet versions
    if isinstance(config.blank, str):
        config.blank = 0
    if char_list is not None:
        config.char_list = char_list
    blank = config.char_list[config.blank]
    ground_truth = config.start_of_ground_truth
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if not ground_truth.endswith(config.space):
            ground_truth += config.space
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add chars of utterance
        for char in utt:
            if char.isspace() and config.replace_spaces_with_blanks:
                if not ground_truth.endswith(config.space):
                    ground_truth += config.space
            elif char in config.char_list and char not in config.excluded_characters:
                ground_truth += char
    # Add space to the end
    if not ground_truth.endswith(config.space):
        ground_truth += config.space
    logger.debug(f"ground_truth: {ground_truth}")
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = max([len(c) for c in config.char_list])
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int64) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i - s < 0:
                continue
            span = ground_truth[i - s : i + 1]
            span = span.replace(config.space, blank)
            if span in config.char_list:
                char_index = config.char_list.index(span)
                ground_truth_mat[i, s] = char_index
    config.is_syncope_token = _create_syncope_mask(
        config, ground_truth, is_token_ids=False
    )
    if config.is_intrusive_token is None and config.intrusive_tokens is not None:
        config.is_intrusive_token = _create_intrusive_mask(config)
    return ground_truth_mat, utt_begin_indices


def prepare_tokenized_text(
    config: CtcSegmentationParameters,
    text: Sequence[str],
) -> Tuple[np.ndarray, List[int]]:
    """Prepare the given tokenized text for CTC segmentation.

    :param config: an instance of CtcSegmentationParameters
    :param text: string with tokens separated by spaces
    :return: label matrix, character index matrix
    """
    ground_truth = [config.start_of_ground_truth]
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if ground_truth[-1] != config.space:
            ground_truth.append(config.space)
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add tokens of utterance
        for token in utt.split():
            if token in config.char_list:
                if config.replace_spaces_with_blanks and not token.startswith(
                    config.tokenized_meta_symbol
                ):
                    ground_truth.append(config.space)
                ground_truth.append(token)
    # Add space to the end
    if ground_truth[-1] != config.space:
        ground_truth.append(config.space)
    logger.debug(f"ground_truth: {ground_truth}")
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = 1
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int64) * -1
    for i in range(1, len(ground_truth)):
        if ground_truth[i] == config.space:
            ground_truth_mat[i, 0] = config.blank
        else:
            char_index = config.char_list.index(ground_truth[i])
            ground_truth_mat[i, 0] = char_index
    config.is_syncope_token = _create_syncope_mask(
        config, ground_truth, is_token_ids=False
    )
    if config.is_intrusive_token is None and config.intrusive_tokens is not None:
        config.is_intrusive_token = _create_intrusive_mask(config)
    return ground_truth_mat, utt_begin_indices


def prepare_token_list(
    config: CtcSegmentationParameters,
    text: Sequence[np.ndarray],
) -> Tuple[np.ndarray, List[int]]:
    """Prepare the given token list for CTC segmentation.

    This function expects the text input in form of a list
    of numpy arrays: [np.array([2, 5]), np.array([7, 9])]

    :param config: an instance of CtcSegmentationParameters
    :param text: list of numpy arrays with tokens
    :return: label matrix, character index matrix
    """
    ground_truth = [-1]
    utt_begin_indices = []
    for utt in text:
        # It's not possible to detect spaces when sequence is
        # already tokenized, so we skip replace_spaces_with_blanks
        # Insert blanks between utterances
        if ground_truth[-1] != config.blank:
            ground_truth.append(config.blank)
        # Start-of-new-utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Append tokens to list
        ground_truth.extend(utt.tolist())
    # Add a blank to the end
    if ground_truth[-1] != config.blank:
        ground_truth.append(config.blank)
    logger.debug(f"ground_truth: {ground_truth}")
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    ground_truth_mat = np.array(ground_truth, dtype=np.int64).reshape(-1, 1)
    config.is_syncope_token = _create_syncope_mask(
        config, ground_truth, is_token_ids=True
    )
    if config.is_intrusive_token is None and config.intrusive_tokens is not None:
        config.is_intrusive_token = _create_intrusive_mask(config)
    return ground_truth_mat, utt_begin_indices


def determine_utterance_segments(
    config: CtcSegmentationParameters,
    utt_begin_indices: Sequence[int],
    char_probs: np.ndarray,
    timings: np.ndarray,
    text: Sequence[Any],
) -> List[Tuple[float, float, float]]:
    """Utterance-wise alignments from char-wise alignments.

    :param config: an instance of CtcSegmentationParameters
    :param utt_begin_indices: list of time indices of utterance start
    :param char_probs:  character positioned probabilities obtained from backtracking
    :param timings: mapping of time indices to seconds
    :param text: list of utterances
    :return: segments, a list of: utterance start and end [s], and its confidence score
    """

    def compute_time(index: int, align_type: str) -> float:
        """Compute start and end time of utterance.

        :param index:  frame index value
        :param align_type:  one of ["begin", "end"]
        :return: start/end time of utterance in seconds
        """
        middle = (timings[index] + timings[index - 1]) / 2
        if align_type == "begin":
            return max(timings[index + 1] - 0.5, middle)
        elif align_type == "end":
            return min(timings[index - 1] + 0.5, middle)
        return middle

    segments = []
    min_prob = np.float64(-10000000000.0)
    for i in range(len(text)):
        start = compute_time(utt_begin_indices[i], "begin")
        end = compute_time(utt_begin_indices[i + 1], "end")
        start_t = int(round(start / config.index_duration))
        end_t = int(round(end / config.index_duration))
        # Compute confidence score by using the min mean probability
        #   after splitting into segments of L frames
        n = config.score_min_mean_over_L
        if end_t <= start_t:
            min_avg = min_prob
        elif end_t - start_t <= n:
            min_avg = char_probs[start_t:end_t].mean()
        else:
            min_avg = np.float64(0.0)
            for t in range(start_t, end_t - n):
                min_avg = min(min_avg, char_probs[t : t + n].mean())
        segments.append((start, end, min_avg))
    return segments
