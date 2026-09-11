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
    """Default values for CTC segmentation.

    May need adjustment according to localization or ASR settings.
    The character set is taken from the model dict, i.e., usually are generated
    with SentencePiece. An ASR model trained in the corresponding language and
    character set is needed. If the character set contains any punctuation
    characters, "#", the Greek char "ε", or the space placeholder, adapt
    these settings.
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
    syncopy_penalty: float = 0.25
    optional_vowel_tokens: Optional[Sequence[Union[str, int]]] = None
    is_optional_vowel: Optional[np.ndarray] = None
    # legacy Parameters (deprecated, use index_duration instead)
    _subsampling_factor: Optional[Union[int, float]] = None
    _frame_duration_ms: Optional[Union[int, float]] = None

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


def ctc_segmentation(
    config: CtcSegmentationParameters,
    lpz: np.ndarray,
    ground_truth: np.ndarray,
    is_optional_vowel: Optional[Union[np.ndarray, Sequence[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract character-level utterance alignments.

    :param config: an instance of CtcSegmentationParameters
    :param lpz: probabilities obtained from CTC output
    :param ground_truth:  ground truth text in the form of a label sequence
    :param is_optional_vowel: optional 1D mask marking optional vowel positions
    :return:
    """
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
    if is_optional_vowel is None:
        if (
            getattr(config, "is_optional_vowel", None) is not None
            and len(config.is_optional_vowel) == len(ground_truth)
        ):
            is_optional_vowel_arr = config.is_optional_vowel
        else:
            is_optional_vowel_arr = np.zeros(len(ground_truth), dtype=np.int8)
    else:
        is_optional_vowel_arr = np.asarray(is_optional_vowel, dtype=np.int8)

    syncopy_penalty = float(getattr(config, "syncopy_penalty", 0.25))

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
            is_optional_vowel_arr,
            syncopy_penalty,
            config.blank,
            config.flags,
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

                # Check vowel skip transitions
                min_vowel_skip_delta = np.inf
                best_vowel_c_prev = None
                best_vowel_s = None
                if c >= 2:
                    for s in range(ground_truth.shape[1]):
                        if ground_truth[c, s] != -1:
                            if is_optional_vowel_arr[c - 1] == 1:
                                for c_prev in range(max(0, c - 3), c - 1):
                                    delta_offset = offsets[c] - offsets[c_prev]
                                    t_prev = t - 1 + delta_offset
                                    if 0 <= t_prev < table.shape[0]:
                                        est_v_prob = table[t, c] - table[t_prev, c_prev]
                                        expected_v_prob = (
                                            lpz[t + offsets[c], ground_truth[c, s]]
                                            - syncopy_penalty
                                        )
                                        v_delta = abs(est_v_prob - expected_v_prob)
                                        if v_delta < min_vowel_skip_delta:
                                            min_vowel_skip_delta = v_delta
                                            best_vowel_c_prev = c_prev
                                            best_vowel_s = s
                            elif (
                                c >= 3
                                and is_optional_vowel_arr[c - 2] == 1
                                and (
                                    ground_truth[c - 1, 0] == blank
                                    or ground_truth[c - 1, 0] == -1
                                )
                            ):
                                for c_prev in range(max(0, c - 4), c - 2):
                                    delta_offset = offsets[c] - offsets[c_prev]
                                    t_prev = t - 1 + delta_offset
                                    if 0 <= t_prev < table.shape[0]:
                                        est_v_prob = table[t, c] - table[t_prev, c_prev]
                                        expected_v_prob = (
                                            lpz[t + offsets[c], ground_truth[c, s]]
                                            - syncopy_penalty
                                        )
                                        v_delta = abs(est_v_prob - expected_v_prob)
                                        if v_delta < min_vowel_skip_delta:
                                            min_vowel_skip_delta = v_delta
                                            best_vowel_c_prev = c_prev
                                            best_vowel_s = s

                # Check which transition has been taken
                if (
                    min_vowel_skip_delta < min_switch_prob_delta
                    and min_vowel_skip_delta < stay_prob_delta
                    and best_vowel_c_prev is not None
                ):
                    # Apply reverse vowel skip transition
                    if c > 0:
                        timings[c] = (offsets[c] + t) * config.index_duration
                        char_probs[offsets[c] + t] = lpz[
                            t + offsets[c], ground_truth[c, best_vowel_s]
                        ]
                        char_index = ground_truth[c, best_vowel_s]
                        state_list[offsets[c] + t] = config.char_list[char_index]
                    offset = offsets[c] - offsets[best_vowel_c_prev]
                    c = best_vowel_c_prev
                    t -= 1 - offset
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
                    offset = offsets[c] - (offsets[c - 1 - min_s] if c - min_s > 0 else 0)
                    c -= 1 + min_s
                    t -= 1 - offset
                else:
                    # Apply reverse stay transition
                    char_probs[offsets[c] + t] = stay_prob
                    state_list[offsets[c] + t] = config.self_transition
                    t -= 1
        except IndexError:
            logger.warning(
                "IndexError: Backtracking was not successful, "
                "the window size might be too small."
            )
            window_size *= 2
            if window_size < config.max_window_size:
                logger.warning("Increasing the window size to: " + str(window_size))
                continue
            else:
                logger.error("Maximum window size reached.")
                logger.error("Check data and character list!")
                raise
        break
    return timings, char_probs, state_list


def _create_optional_vowel_mask(
    config: CtcSegmentationParameters,
    ground_truth: Union[Sequence[Any], np.ndarray],
    is_token_ids: bool = False,
) -> np.ndarray:
    """Create a 1D int8 mask indicating optional vowel positions in ground_truth."""
    mask = np.zeros(len(ground_truth), dtype=np.int8)
    if config.optional_vowel_tokens is None:
        return mask
    optional_set = set(config.optional_vowel_tokens)
    for idx, item in enumerate(ground_truth):
        if is_token_ids:
            if item in optional_set:
                mask[idx] = 1
            elif (
                config.char_list is not None
                and isinstance(item, (int, np.integer))
                and 0 <= item < len(config.char_list)
                and config.char_list[item] in optional_set
            ):
                mask[idx] = 1
        else:
            if item in optional_set:
                mask[idx] = 1
            elif (
                config.char_list is not None
                and item in config.char_list
                and config.char_list.index(item) in optional_set
            ):
                mask[idx] = 1
    return mask


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
    config.is_optional_vowel = _create_optional_vowel_mask(
        config, ground_truth, is_token_ids=False
    )
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
    config.is_optional_vowel = _create_optional_vowel_mask(
        config, ground_truth, is_token_ids=False
    )
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
    config.is_optional_vowel = _create_optional_vowel_mask(
        config, ground_truth, is_token_ids=True
    )
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
