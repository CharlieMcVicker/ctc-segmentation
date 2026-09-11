#!/usr/bin/env python3
# encoding: utf-8

"""Test suite and verification matrix for intrusive token transitions."""

import numpy as np
import pytest

from ctc_segmentation import (
    CtcSegmentationParameters,
    ctc_segmentation,
    determine_utterance_segments,
    prepare_text,
    prepare_token_list,
    prepare_tokenized_text,
)


def make_emissions(spoken_chars, char_list, frames_per_char=2, blank_frames=2):
    """Generate synthetic emission matrix for a sequence of characters."""
    total_frames = blank_frames + len(spoken_chars) * frames_per_char + blank_frames
    V = len(char_list)
    lpz = np.full((total_frames, V), -10.0, dtype=np.float32)

    # Initial blank
    lpz[0:blank_frames, 0] = 0.0

    current_t = blank_frames
    for ch in spoken_chars:
        if ch in char_list:
            char_idx = char_list.index(ch)
            lpz[current_t : current_t + frames_per_char, char_idx] = 0.0
        current_t += frames_per_char

    # Final blank
    lpz[current_t : current_t + blank_frames, 0] = 0.0
    return lpz


def test_intrusive_h_inserted():
    """Test 1: When intrusive 'h' is spoken with high acoustic energy, 'h' is reconstructed."""
    word = "akeya"
    spoken_chars = ["a", "k", "e", "y", "h", "a"]
    char_list = ["•"] + sorted(list(set(word + "h")))

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.5,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    # State list must contain intrusive 'h'
    assert "h" in states, f"Expected intrusive 'h' in states, got: {states}"

    # Verify reconstructed order: 'y' followed by 'h' followed by 'a'
    non_empty_states = [s for s in states if s and s != config.self_transition]
    assert "h" in non_empty_states
    y_idx = non_empty_states.index("y")
    h_idx = non_empty_states.index("h")
    a_idx = non_empty_states.index("a", h_idx)
    assert y_idx < h_idx < a_idx

    # Utterance segment confidence is high
    segments = determine_utterance_segments(config, utt_indices, char_probs, timings, [word])
    assert segments[0][2] > -1.0


def test_unvoiced_h_not_inserted():
    """Test 2: When no acoustic evidence for 'h' exists, 'h' is not inserted."""
    word = "akeya"
    spoken_chars = ["a", "k", "e", "y", "a"]
    char_list = ["•"] + sorted(list(set(word + "h")))

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.5,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=2, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    # State list must NOT contain intrusive 'h'
    assert "h" not in states, f"Expected 'h' not to be in states, got: {states}"
    for ch in ["a", "k", "e", "y"]:
        assert ch in states


def test_intrusive_glottal_stop():
    """Test 3: Intrusive glottal stop "'" is inserted between 'a' and 'n' in 'tsanvsv'."""
    word = "tsanvsv"
    spoken_chars = ["t", "s", "a", "'", "n", "v", "s", "v"]
    char_list = ["•"] + sorted(list(set(word + "'")))

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["'"],
        intrusive_penalty=0.5,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    assert "'" in states, f"Expected glottal stop \"'\" in states, got: {states}"
    non_empty_states = [s for s in states if s and s != config.self_transition]
    a_idx = non_empty_states.index("a")
    glottal_idx = non_empty_states.index("'")
    n_idx = non_empty_states.index("n")
    assert a_idx < glottal_idx < n_idx


def test_mutual_exclusivity():
    """Test 4: Single-slot mutual exclusivity ensures at most one intrusive token is chosen."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h'")))

    # Case A: Audio has high energy for 'h'
    spoken_chars_h = ["a", "k", "e", "y", "h", "a"]
    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_penalty=0.5,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz_h = make_emissions(spoken_chars_h, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz_h, gt_mat)
    non_empty = [s for s in states if s and s != config.self_transition]

    # Must contain 'h' but never stack "'"
    assert "h" in non_empty
    assert "'" not in non_empty

    # Reconstructed string shouldn't have consecutive intrusive tokens
    intrusive_count = sum(1 for s in non_empty if s in ["h", "'"])
    assert intrusive_count == 1


def test_combined_syncope_and_intrusion():
    """Test 5: Syncope vowel omission and intrusive consonant insertion interact correctly."""
    word = "adalenisgv"
    # Canonical: a-d-a-l-e-n-i-s-g-v
    # Realized: medial vowel 'a' (in 'da') dropped via syncope, coda 'h' inserted before 's' via intrusive detour
    # -> a-d-l-e-n-i-h-s-g-v
    spoken_chars = ["a", "d", "l", "e", "n", "i", "h", "s", "g", "v"]
    char_list = ["•"] + sorted(list(set(word + "h")))

    config = CtcSegmentationParameters(
        char_list=char_list,
        syncope_tokens=["a"],
        syncope_penalty=0.25,
        intrusive_tokens=["h"],
        intrusive_penalty=0.5,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    # 'h' should be inserted due to intrusive detour
    assert "h" in states, f"Expected 'h' to be inserted, got: {states}"

    # Timing for dropped medial 'a' at ground truth position 4 should be 0.0
    # Ground truth: '#', '·', 'a', 'd', 'a', 'l', 'e', 'n', 'i', 's', 'g', 'v', '·'
    # indices:       0    1    2    3    4    5    6    7    8    9   10   11   12
    # The first 'a' is at index 2, medial 'a' is at index 4
    assert config.is_syncope_token[4] == 1
    assert timings[4] == 0.0, f"Expected dropped medial 'a' at pos 4 to have 0.0s timing, got: {timings[4]}"

    # Utterance segment confidence should remain high
    segments = determine_utterance_segments(config, utt_indices, char_probs, timings, [word])
    assert segments[0][2] > -1.0


def test_backwards_compatibility():
    """Test 6: Baseline alignment is identical when intrusive_tokens=None."""
    word = "akeya"
    spoken_chars = ["a", "k", "e", "y", "a"]
    char_list = ["•"] + sorted(list(set(word + "h'")))

    config_none = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=None,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    config_default = CtcSegmentationParameters(
        char_list=char_list,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat_1, utt_1 = prepare_text(config_none, [word], char_list)
    gt_mat_2, utt_2 = prepare_text(config_default, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=2, blank_frames=2)

    timings_1, probs_1, states_1 = ctc_segmentation(config_none, lpz, gt_mat_1)
    timings_2, probs_2, states_2 = ctc_segmentation(config_default, lpz, gt_mat_2)

    np.testing.assert_array_almost_equal(timings_1, timings_2)
    np.testing.assert_array_almost_equal(probs_1, probs_2)
    assert states_1 == states_2


def test_intrusive_transitions_prepare_token_list():
    """Test 7: Intrusive token transitions using prepare_token_list."""
    char_list = ["•", "a", "k", "e", "y", "h"]
    # Token IDs for "akeya": [1, 2, 3, 4, 1]
    token_array = np.array([1, 2, 3, 4, 1], dtype=np.int64)

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.5,
        min_window_size=50,
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_token_list(config, [token_array])
    # Spoken audio contains intrusive 'h': a, k, e, y, h, a
    spoken_chars = ["a", "k", "e", "y", "h", "a"]
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=2, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    # Intrusive 'h' must be captured in states
    assert "h" in states, f"Expected 'h' in states, got: {states}"
    non_empty = [s for s in states if s and s != config.self_transition]
    assert "h" in non_empty
    y_idx = non_empty.index("y")
    h_idx = non_empty.index("h")
    a_idx = non_empty.index("a", h_idx)
    assert y_idx < h_idx < a_idx

    segments = determine_utterance_segments(config, utt_indices, char_probs, timings, [token_array])
    assert segments[0][2] > -1.0


def test_intrusive_transitions_prepare_tokenized_text():
    """Test 8: Intrusive token transitions using prepare_tokenized_text."""
    char_list = ["•", "tok_a", "tok_key", "tok_end", "h"]
    tokenized_text = ["tok_a tok_key tok_end"]

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.5,
        min_window_size=50,
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_tokenized_text(config, tokenized_text)
    # Spoken tokens: tok_a, tok_key, h, tok_end
    spoken_tokens = ["tok_a", "tok_key", "h", "tok_end"]
    lpz = make_emissions(spoken_tokens, char_list, frames_per_char=2, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    assert "h" in states, f"Expected 'h' in states, got: {states}"
    non_empty = [s for s in states if s and s != config.self_transition]
    assert "h" in non_empty
    key_idx = non_empty.index("tok_key")
    h_idx = non_empty.index("h")
    end_idx = non_empty.index("tok_end", h_idx)
    assert key_idx < h_idx < end_idx

    segments = determine_utterance_segments(config, utt_indices, char_probs, timings, tokenized_text)
    assert segments[0][2] > -1.0


def test_intrusive_window_partitioning():
    """Test 9: Intrusive token backtracking across dynamic window partitions and shifts."""
    word1 = "akeya"
    word2 = "tsanvsv"
    char_list = ["•"] + sorted(list(set(word1 + word2 + "h'")))

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_penalty=0.5,
        min_window_size=20,  # Small window forces window shifting across audio frames
        score_min_mean_over_L=2,
    )

    text = [word1, word2, word1]
    gt_mat, utt_indices = prepare_text(config, text, char_list)

    # Build audio emissions with inter-utterance blank periods (represented by index 0 "•")
    spoken_chars = (
        ["a", "k", "e", "y", "h", "a"]
        + ["•"]
        + ["t", "s", "a", "'", "n", "v", "s", "v"]
        + ["•"]
        + ["a", "k", "e", "y", "h", "a"]
    )
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)
    assert lpz.shape[0] > config.min_window_size

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    # Both intrusive tokens must be captured in states
    assert "h" in states, f"Expected 'h' in states, got: {states}"
    assert "'" in states, f"Expected \"'\" in states, got: {states}"

    # Verify segments confidence across all window partitions
    segments = determine_utterance_segments(config, utt_indices, char_probs, timings, text)
    assert len(segments) == 3
    for i, seg in enumerate(segments):
        assert seg[2] > -1.0, f"Utterance {i} segment confidence too low: {seg[2]}"


