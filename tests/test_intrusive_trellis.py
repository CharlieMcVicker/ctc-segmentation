#!/usr/bin/env python3
# encoding: utf-8

"""Test suite and verification matrix for intrusive token transitions."""

import numpy as np
import pytest

from ctc_segmentation import (
    CtcSegmentationParameters,
    TrellisRuntimeContext,
    ctc_segmentation,
    determine_utterance_segments,
    prepare_text,
    prepare_token_list,
    prepare_tokenized_text,
)
from ctc_segmentation.ctc_segmentation_dyn import cython_fill_table


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


@pytest.mark.parametrize("delta", [0, 1, 2, 3, 4])
def test_blank_tolerant_intrusive_strides(delta):
    """Test 10: Intrusive token 'h' is detected across 0, 1, 2, 3, 4 intervening CTC blank frames."""
    word = "akeya"
    # Ground truth: a-k-e-y-a
    # Spoken: a, k, e, y, h, [PAD]*delta, a
    char_list = ["•"] + sorted(list(set(word + "h")))
    spoken_chars = ["a", "k", "e", "y", "h"] + ["•"] * delta + ["a"]

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.1,
        intrusive_max_stride=4,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    # Must contain 'h'
    assert "h" in states, f"Expected 'h' with delta={delta} blanks, got states: {states}"

    non_empty = [s for s in states if s and s != config.self_transition]
    assert "h" in non_empty
    y_idx = non_empty.index("y")
    h_idx = non_empty.index("h")
    a_idx = non_empty.index("a", h_idx)
    assert y_idx < h_idx < a_idx, f"Order mismatch with delta={delta}: {non_empty}"

    segments = determine_utterance_segments(config, utt_indices, char_probs, timings, [word])
    assert segments[0][2] > -1.0


def test_multi_token_intrusions_with_blanks():
    """Test 11: Multi-candidate intrusive transitions (h, ') each detected across intervening blank frames."""
    word_h = "akeya"
    word_g = "tsanvsv"
    char_list = ["•"] + sorted(list(set(word_h + word_g + "h'")))

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_penalty=0.1,
        intrusive_max_stride=4,
        min_window_size=max(50, len(word_h + word_g) * 4 + 20),
        score_min_mean_over_L=2,
    )

    # Case A: 'h' followed by 2 intervening blanks
    spoken_h = ["a", "k", "e", "y", "h", "•", "•", "a"]
    gt_mat_h, utt_h = prepare_text(config, [word_h], char_list)
    lpz_h = make_emissions(spoken_h, char_list, frames_per_char=1, blank_frames=2)
    _, _, states_h = ctc_segmentation(config, lpz_h, gt_mat_h)

    assert "h" in states_h
    assert "'" not in states_h
    non_empty_h = [s for s in states_h if s and s != config.self_transition]
    assert non_empty_h.index("y") < non_empty_h.index("h") < non_empty_h.index("a", non_empty_h.index("h"))

    # Case B: "'" (glottal stop) followed by 3 intervening blanks
    spoken_g = ["t", "s", "a", "'", "•", "•", "•", "n", "v", "s", "v"]
    gt_mat_g, utt_g = prepare_text(config, [word_g], char_list)
    lpz_g = make_emissions(spoken_g, char_list, frames_per_char=1, blank_frames=2)
    _, _, states_g = ctc_segmentation(config, lpz_g, gt_mat_g)

    assert "'" in states_g
    assert "h" not in states_g
    non_empty_g = [s for s in states_g if s and s != config.self_transition]
    assert non_empty_g.index("a") < non_empty_g.index("'") < non_empty_g.index("n")


def test_intrusive_max_stride_limit_enforcement():
    """Test 12: Blank stride exceeding intrusive_max_stride is not accepted as intrusive detour."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h")))
    # 5 intervening blanks between 'h' and 'a'
    spoken_chars = ["a", "k", "e", "y", "h", "•", "•", "•", "•", "•", "a"]

    # With default intrusive_max_stride=4 (cannot bridge 5 blanks)
    config_stride4 = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.1,
        intrusive_max_stride=4,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config_stride4, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    _, _, states_stride4 = ctc_segmentation(config_stride4, lpz, gt_mat)
    # Stride of 5 blanks exceeds max_stride of 4, so 'h' cannot be inserted via intrusive detour
    assert "h" not in states_stride4

    # With intrusive_max_stride=5 (can bridge 5 blanks)
    config_stride5 = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.1,
        intrusive_max_stride=5,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    _, _, states_stride5 = ctc_segmentation(config_stride5, lpz, gt_mat)
    assert "h" in states_stride5


def test_clean_speech_with_blanks_no_spurious_intrusions():
    """Test 13: Clean speech containing blank regions without acoustic intrusions does not insert spurious tokens."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h'")))
    # Multiple blank frames between canonical characters, but no intrusive phonemes
    spoken_chars = ["a", "•", "•", "k", "•", "e", "•", "•", "•", "y", "•", "•", "a"]

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_penalty=0.1,
        intrusive_max_stride=4,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    # Neither 'h' nor "'" should be inserted
    assert "h" not in states, f"Expected 'h' not to be in states, got: {states}"
    assert "'" not in states, f"Expected \"'\" not to be in states, got: {states}"

    for ch in ["a", "k", "e", "y"]:
        assert ch in states
    segments = determine_utterance_segments(config, utt_indices, char_probs, timings, [word])
    assert segments[0][2] > -1.0


def test_trellis_runtime_context_compile_prob_conversion():
    """Test TrellisRuntimeContext.compile linear-to-logprob conversion and raw logprob preservation."""
    char_list = ["•", "a", "k", "e", "y", "h", "'"]
    ground_truth = np.zeros((10, 2), dtype=np.int64)

    # 1. Uniform linear probability in (0, 1] converted to log-probability
    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_min_logprobs=0.40,
    )
    ctx = TrellisRuntimeContext.compile(config, ground_truth)
    assert np.allclose(ctx.intrusive_min_logprobs, np.log(0.40), atol=1e-5)
    assert ctx.intrusive_min_logprobs.dtype == np.float32
    assert ctx.intrusive_min_logprobs.flags.c_contiguous

    # 2. Raw log-probability (<= 0) preserved as-is
    config_raw = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_min_logprobs=-1.5,
    )
    ctx_raw = TrellisRuntimeContext.compile(config_raw, ground_truth)
    assert np.allclose(ctx_raw.intrusive_min_logprobs, -1.5, atol=1e-5)

    # 3. Edge case: 1.0 converts to ln(1.0) == 0.0
    config_one = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_min_logprobs=1.0,
    )
    ctx_one = TrellisRuntimeContext.compile(config_one, ground_truth)
    assert np.isclose(ctx_one.intrusive_min_logprobs[0], 0.0, atol=1e-5)


def test_trellis_runtime_context_compile_dict_lookup():
    """Test TrellisRuntimeContext.compile dictionary mapping and token ID resolution."""
    char_list = ["•", "a", "k", "e", "y", "h", "'"]
    ground_truth = np.zeros((8, 2), dtype=np.int64)

    # h is index 5, ' is index 6
    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'", 2],  # mixed token strings and integer token ID
        intrusive_penalties={"h": 1.2, "'": 0.5, 2: 0.8},
        intrusive_min_logprobs={"h": 0.40, "'": -1.5, 2: 0.25},
    )
    ctx = TrellisRuntimeContext.compile(config, ground_truth)

    assert np.array_equal(ctx.intrusive_token_ids, np.array([5, 6, 2], dtype=np.int64))
    assert ctx.intrusive_token_ids.flags.c_contiguous

    assert np.allclose(ctx.intrusive_penalties, np.array([1.2, 0.5, 0.8], dtype=np.float32))
    assert ctx.intrusive_penalties.flags.c_contiguous

    expected_min_logprobs = np.array([np.log(0.40), -1.5, np.log(0.25)], dtype=np.float32)
    assert np.allclose(ctx.intrusive_min_logprobs, expected_min_logprobs, atol=1e-5)
    assert ctx.intrusive_min_logprobs.flags.c_contiguous

    # Partial dict mapping: unmapped tokens get default penalty and -np.inf threshold
    config_partial = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_penalty=0.3,
        intrusive_penalties={"h": 1.0},
        intrusive_min_logprobs={"h": 0.50},
    )
    ctx_partial = TrellisRuntimeContext.compile(config_partial, ground_truth)
    assert np.isclose(ctx_partial.intrusive_penalties[0], 1.0)
    assert np.isclose(ctx_partial.intrusive_penalties[1], 0.3)
    assert np.isclose(ctx_partial.intrusive_min_logprobs[0], np.log(0.50), atol=1e-5)
    assert np.isneginf(ctx_partial.intrusive_min_logprobs[1])


def test_trellis_runtime_context_compile_size_validation():
    """Test TrellisRuntimeContext.compile validates dimension matches against ground_truth."""
    char_list = ["•", "a", "k", "e", "y"]
    ground_truth = np.zeros((6, 2), dtype=np.int64)

    # 1. is_syncope_token length mismatch raises ValueError
    config_bad_syncope = CtcSegmentationParameters(
        char_list=char_list,
        is_syncope_token=[0, 1, 0, 0],  # len 4 != 6
    )
    with pytest.raises(ValueError, match="is_syncope_token length .* does not match ground_truth length"):
        TrellisRuntimeContext.compile(config_bad_syncope, ground_truth)

    # 2. is_intrusive_site length mismatch raises ValueError
    config_bad_site = CtcSegmentationParameters(
        char_list=char_list,
        is_intrusive_site=[True, False, False, False, False, False, True],  # len 7 != 6
    )
    with pytest.raises(ValueError, match="is_intrusive_site length .* does not match ground_truth length"):
        TrellisRuntimeContext.compile(config_bad_site, ground_truth)

    # 3. Multidimensional mask raises ValueError
    config_2d = CtcSegmentationParameters(
        char_list=char_list,
        is_syncope_token=np.zeros((6, 2), dtype=np.int8),
    )
    with pytest.raises(ValueError, match="is_syncope_token length"):
        TrellisRuntimeContext.compile(config_2d, ground_truth)

    # 4. Correct matching lengths compile cleanly to C-contiguous int8
    config_valid = CtcSegmentationParameters(
        char_list=char_list,
        is_syncope_token=[0, 1, 0, 1, 0, 0],
        is_intrusive_site=[True, True, False, False, True, False],
    )
    ctx_valid = TrellisRuntimeContext.compile(config_valid, ground_truth)
    assert ctx_valid.is_syncope_token.shape == (6,)
    assert ctx_valid.is_syncope_token.dtype == np.int8
    assert ctx_valid.is_syncope_token.flags.c_contiguous
    assert ctx_valid.is_intrusive_site.shape == (6,)
    assert ctx_valid.is_intrusive_site.dtype == np.int8
    assert ctx_valid.is_intrusive_site.flags.c_contiguous


def test_trellis_runtime_context_compile_defaults():
    """Test TrellisRuntimeContext.compile default attributes when optional fields are None."""
    char_list = ["•", "a", "b"]
    ground_truth = np.zeros((4, 2), dtype=np.int64)

    # No intrusive or syncope tokens
    config = CtcSegmentationParameters(char_list=char_list)
    ctx = TrellisRuntimeContext.compile(config, ground_truth)

    assert ctx.intrusive_token_ids.shape == (0,)
    assert ctx.intrusive_token_ids.dtype == np.int64
    assert ctx.intrusive_penalties.shape == (0,)
    assert ctx.intrusive_penalties.dtype == np.float32
    assert ctx.intrusive_min_logprobs.shape == (0,)
    assert ctx.intrusive_min_logprobs.dtype == np.float32
    assert ctx.is_syncope_token.shape == (0,)
    assert ctx.is_syncope_token.dtype == np.int8
    assert ctx.is_intrusive_site.shape == (0,)
    assert ctx.is_intrusive_site.dtype == np.int8
    assert ctx.intrusive_max_stride == 4
    assert ctx.blank == 0

    # With intrusive tokens, intrusive_min_logprobs=None -> -np.inf
    config_intrusive = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["a", "b"],
    )
    ctx_intrusive = TrellisRuntimeContext.compile(config_intrusive, ground_truth)
    assert ctx_intrusive.intrusive_token_ids.shape == (2,)
    assert np.all(np.isneginf(ctx_intrusive.intrusive_min_logprobs))
    assert np.allclose(ctx_intrusive.intrusive_penalties, 0.1)


def test_trellis_runtime_context_compile_type_errors():
    """Test TrellisRuntimeContext.compile error handling for invalid token types or parameters."""
    ground_truth = np.zeros((4, 2), dtype=np.int64)

    # 1. Intrusive token not found in char_list
    config_missing = CtcSegmentationParameters(
        char_list=["•", "a"],
        intrusive_tokens=["z"],
    )
    with pytest.raises(ValueError, match="Intrusive token 'z' not found in char_list"):
        TrellisRuntimeContext.compile(config_missing, ground_truth)

    # 2. Invalid intrusive token element type (e.g. float or object)
    config_invalid_token = CtcSegmentationParameters(
        char_list=["•", "a"],
        intrusive_tokens=[3.14],  # type: ignore
    )
    with pytest.raises(TypeError, match="Invalid intrusive token type"):
        TrellisRuntimeContext.compile(config_invalid_token, ground_truth)

    # 3. Invalid intrusive_penalties type
    config_bad_penalties = CtcSegmentationParameters(
        char_list=["•", "a"],
        intrusive_tokens=["a"],
        intrusive_penalties="invalid_type",  # type: ignore
    )
    with pytest.raises(TypeError, match="Invalid intrusive_penalties type"):
        TrellisRuntimeContext.compile(config_bad_penalties, ground_truth)

    # 4. Invalid intrusive_min_logprobs type
    config_bad_min_logprobs = CtcSegmentationParameters(
        char_list=["•", "a"],
        intrusive_tokens=["a"],
        intrusive_min_logprobs="invalid_type",  # type: ignore
    )
    with pytest.raises(TypeError, match="Invalid intrusive_min_logprobs type"):
        TrellisRuntimeContext.compile(config_bad_min_logprobs, ground_truth)


def test_cython_fill_table_min_logprob_gating():
    """Test cython_fill_table skips intrusive detour when posterior logprob is below threshold."""
    # ground_truth: '#' (0), 'a' (1)
    # columns: c=0 ('#'), c=1 ('a')
    ground_truth = np.array([[0], [1]], dtype=np.int64)
    # lpz: shape (5 frames, 3 vocab tokens: 0=blank, 1='a', 2='h')
    lpz = np.full((5, 3), -10.0, dtype=np.float32)
    lpz[0, 0] = 0.0  # t=0 blank
    lpz[1, 2] = -2.0  # t=1 'h' with posterior logprob -2.0 (linear prob ~0.135)
    lpz[2, 1] = 0.0  # t=2 'a'

    offsets = np.zeros(2, dtype=np.int64)
    is_syncope_token = np.zeros(2, dtype=np.int8)
    intrusive_token_ids = np.array([2], dtype=np.int64)
    intrusive_penalties = np.array([0.1], dtype=np.float32)
    is_intrusive_site = np.zeros(0, dtype=np.int8)

    # Case 1: Threshold is strict (e.g. -1.0), so -2.0 is below threshold -> detour blocked
    intrusive_min_logprobs_strict = np.array([-1.0], dtype=np.float32)
    table1 = np.full((5, 2), -1e10, dtype=np.float32)
    cython_fill_table(
        table1,
        lpz,
        ground_truth,
        offsets,
        is_syncope_token,
        0.25,
        intrusive_token_ids,
        intrusive_penalties,
        intrusive_min_logprobs_strict,
        is_intrusive_site,
        4,
        0,
        0,
    )

    # Case 2: Threshold is loose (e.g. -3.0), so -2.0 satisfies threshold -> detour allowed
    intrusive_min_logprobs_loose = np.array([-3.0], dtype=np.float32)
    table2 = np.full((5, 2), -1e10, dtype=np.float32)
    cython_fill_table(
        table2,
        lpz,
        ground_truth,
        offsets,
        is_syncope_token,
        0.25,
        intrusive_token_ids,
        intrusive_penalties,
        intrusive_min_logprobs_loose,
        is_intrusive_site,
        4,
        0,
        0,
    )

    # At t=2, c=1: table2 should have higher score from the detour transition than table1
    # detour score: table[0, 0] (0.0) + lpz[1, 2] (-2.0) - penalty (0.1) + lpz[2, 1] (0.0) = -2.1
    # switch score without detour at t=2: table[1, 0] (blank stay: 0.0 + lpz[1,0](-10) = -10) + lpz[2, 1](0.0) = -10
    assert table2[2, 1] > table1[2, 1]
    assert np.isclose(table2[2, 1], -2.1, atol=1e-5)


def test_cython_fill_table_site_masking():
    """Test cython_fill_table respects is_intrusive_site masking array."""
    # ground_truth: '#' (0), 'a' (1), 'b' (2)
    # columns: c=0, c=1, c=2
    ground_truth = np.array([[0], [1], [2]], dtype=np.int64)
    lpz = np.full((6, 4), -10.0, dtype=np.float32)
    lpz[0, 0] = 0.0  # t=0 blank
    lpz[1, 3] = 0.0  # t=1 intrusive 'h' (id 3)
    lpz[2, 1] = 0.0  # t=2 'a' (id 1)

    offsets = np.zeros(3, dtype=np.int64)
    is_syncope_token = np.zeros(3, dtype=np.int8)
    intrusive_token_ids = np.array([3], dtype=np.int64)
    intrusive_penalties = np.array([0.1], dtype=np.float32)
    intrusive_min_logprobs = np.array([-np.inf], dtype=np.float32)

    # Case 1: Site mask blocks column 1 (is_intrusive_site[1] == 0)
    site_mask_blocked = np.array([0, 0, 1], dtype=np.int8)
    table_blocked = np.full((6, 3), -1e10, dtype=np.float32)
    cython_fill_table(
        table_blocked,
        lpz,
        ground_truth,
        offsets,
        is_syncope_token,
        0.25,
        intrusive_token_ids,
        intrusive_penalties,
        intrusive_min_logprobs,
        site_mask_blocked,
        4,
        0,
        0,
    )

    # Case 2: Site mask allows column 1 (is_intrusive_site[1] == 1)
    site_mask_allowed = np.array([0, 1, 1], dtype=np.int8)
    table_allowed = np.full((6, 3), -1e10, dtype=np.float32)
    cython_fill_table(
        table_allowed,
        lpz,
        ground_truth,
        offsets,
        is_syncope_token,
        0.25,
        intrusive_token_ids,
        intrusive_penalties,
        intrusive_min_logprobs,
        site_mask_allowed,
        4,
        0,
        0,
    )

    # At t=2, c=1: table_allowed should have detour score ~ -0.1, whereas table_blocked should be much lower
    assert table_allowed[2, 1] > table_blocked[2, 1]
    assert np.isclose(table_allowed[2, 1], -0.1, atol=1e-5)


def test_cython_fill_table_per_token_penalties():
    """Test cython_fill_table applies distinct per-token penalties."""
    ground_truth = np.array([[0], [1]], dtype=np.int64)
    lpz = np.full((5, 4), -10.0, dtype=np.float32)
    lpz[0, 0] = 0.0  # t=0 blank
    lpz[1, 2] = 0.0  # t=1 token 2 (e.g. 'h')
    lpz[1, 3] = 0.0  # t=1 token 3 (e.g. "'")
    lpz[2, 1] = 0.0  # t=2 'a'

    offsets = np.zeros(2, dtype=np.int64)
    is_syncope_token = np.zeros(2, dtype=np.int8)
    intrusive_token_ids = np.array([2, 3], dtype=np.int64)
    is_intrusive_site = np.zeros(0, dtype=np.int8)
    intrusive_min_logprobs = np.array([-np.inf, -np.inf], dtype=np.float32)

    # Set token 2 penalty = 0.8, token 3 penalty = 0.2
    intrusive_penalties = np.array([0.8, 0.2], dtype=np.float32)
    table = np.full((5, 2), -1e10, dtype=np.float32)
    cython_fill_table(
        table,
        lpz,
        ground_truth,
        offsets,
        is_syncope_token,
        0.25,
        intrusive_token_ids,
        intrusive_penalties,
        intrusive_min_logprobs,
        is_intrusive_site,
        4,
        0,
        0,
    )

    # The max detour should take token 3 with penalty 0.2 (table[2, 1] = 0.0 - 0.2 + 0.0 = -0.2)
    assert np.isclose(table[2, 1], -0.2, atol=1e-5)


def test_intrusive_gating_and_penalties_end_to_end():
    """End-to-end test verifying gating and per-token penalty preferences via ctc_segmentation."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h'")))

    # Spoken audio has 'h' at moderate probability (lpz = -0.5, ~60% prob)
    spoken_chars = ["a", "k", "e", "y", "h", "a"]
    gt_mat, utt_indices = prepare_text(CtcSegmentationParameters(char_list=char_list), [word], char_list)

    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)
    h_idx = char_list.index("h")
    # 'a'(2), 'k'(3), 'e'(4), 'y'(5), 'h'(6), 'a'(7)
    lpz[6, h_idx] = -0.5
    lpz[6, 0] = -5.0

    # 1. With intrusive_min_logprobs = 0.80 (~ -0.22 logprob), 'h' (-0.5) is rejected
    config_strict = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_min_logprobs=0.80,
        intrusive_penalty=0.1,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_strict = ctc_segmentation(config_strict, lpz, gt_mat)
    assert "h" not in states_strict

    # 2. With intrusive_min_logprobs = 0.50 (~ -0.69 logprob), 'h' (-0.5) is accepted
    config_loose = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_min_logprobs=0.50,
        intrusive_penalty=0.1,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_loose = ctc_segmentation(config_loose, lpz, gt_mat)
    assert "h" in states_loose


def test_intrusive_site_masking_end_to_end():
    """End-to-end test verifying is_intrusive_site restricts detours to specified positions."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h")))
    # Ground truth: '#', '·', 'a', 'k', 'e', 'y', 'a', '·'
    # indices:       0    1    2    3    4    5    6    7
    gt_mat, utt_indices = prepare_text(CtcSegmentationParameters(char_list=char_list), [word], char_list)
    L = len(gt_mat)

    # Audio has spoken intrusive 'h' between 'y' and 'a' (target column is 6 'a')
    spoken_chars = ["a", "k", "e", "y", "h", "a"]
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    # 1. Site mask that disables position 6 (where 'a' is)
    site_mask_blocked = np.ones(L, dtype=np.int8)
    site_mask_blocked[6] = 0
    config_blocked = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.1,
        is_intrusive_site=site_mask_blocked,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_blocked = ctc_segmentation(config_blocked, lpz, gt_mat)
    assert "h" not in states_blocked

    # 2. Site mask that enables position 6
    site_mask_allowed = np.zeros(L, dtype=np.int8)
    site_mask_allowed[6] = 1
    config_allowed = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.1,
        is_intrusive_site=site_mask_allowed,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_allowed = ctc_segmentation(config_allowed, lpz, gt_mat)
    assert "h" in states_allowed


def test_intrusive_per_token_penalties_end_to_end():
    """End-to-end test verifying per-token penalty dict selects the lower penalty candidate."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h'")))
    gt_mat, _ = prepare_text(CtcSegmentationParameters(char_list=char_list), [word], char_list)

    # Audio has equal energy for both 'h' and "'" at frame 6
    spoken_chars = ["a", "k", "e", "y", "h", "a"]
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)
    h_idx = char_list.index("h")
    g_idx = char_list.index("'")
    lpz[6, h_idx] = 0.0
    lpz[6, g_idx] = 0.0

    # Case A: 'h' has penalty 0.1, "'" has penalty 0.8 -> 'h' is chosen
    config_prefer_h = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_penalties={"h": 0.1, "'": 0.8},
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_h = ctc_segmentation(config_prefer_h, lpz, gt_mat)
    assert "h" in states_h
    assert "'" not in states_h

    # Case B: "'" has penalty 0.1, 'h' has penalty 0.8 -> "'" is chosen
    config_prefer_g = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_penalties={"h": 0.8, "'": 0.1},
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_g = ctc_segmentation(config_prefer_g, lpz, gt_mat)
    assert "'" in states_g
    assert "h" not in states_g


def test_trellis_runtime_context_backtracking_compatibility():
    """Verify ctc_segmentation backtracking with TrellisRuntimeContext and legacy configurations."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h")))
    gt_mat, _ = prepare_text(CtcSegmentationParameters(char_list=char_list), [word], char_list)

    spoken_chars = ["a", "k", "e", "y", "h", "a"]
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    # 1. Legacy scalar intrusive_penalty configuration
    config_scalar = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.15,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    ctx_scalar = TrellisRuntimeContext.compile(config_scalar, gt_mat)
    assert np.isclose(ctx_scalar.intrusive_penalties[0], 0.15, atol=1e-5)
    timings, probs, states = ctc_segmentation(config_scalar, lpz, gt_mat)
    assert "h" in states
    assert len(timings) == len(gt_mat)

    # 2. Legacy is_optional_vowel parameter passed into ctc_segmentation()
    config_syncope = CtcSegmentationParameters(
        char_list=char_list,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    vowel_mask = np.zeros(len(gt_mat), dtype=np.int8)
    vowel_mask[2] = 1  # 'a'
    with pytest.deprecated_call():
        timings_v, probs_v, states_v = ctc_segmentation(
            config_syncope, lpz, gt_mat, is_optional_vowel=vowel_mask
        )
    assert len(timings_v) == len(gt_mat)


# ==============================================================================
# Verification Matrix for Min-Logprob Intrusive Gating (TASK-35)
# ==============================================================================


@pytest.mark.parametrize("prob_h", [0.05, 0.15, 0.30])
def test_diffuse_noise_rejection_gating(prob_h):
    """AC #1: Diffuse noise (e.g. P=0.15 for /h/) is suppressed when intrusive_min_logprobs is set."""
    word = "akeya"
    spoken_chars = ["a", "k", "e", "y", "a"]
    char_list = ["•"] + sorted(list(set(word + "h")))
    h_idx = char_list.index("h")

    # Frames: [blank*2, a*2, k*2, e*2, y*2, diffuse_h*1, a*2, blank*2]
    total_frames = 2 + 8 + 1 + 2 + 2  # 15 frames
    lpz_diffuse = np.full((total_frames, len(char_list)), -10.0, dtype=np.float32)
    lpz_diffuse[0:2, 0] = 0.0  # initial blank
    lpz_diffuse[2:4, char_list.index("a")] = 0.0
    lpz_diffuse[4:6, char_list.index("k")] = 0.0
    lpz_diffuse[6:8, char_list.index("e")] = 0.0
    lpz_diffuse[8:10, char_list.index("y")] = 0.0
    # Diffuse noise frame at t=10:
    lpz_diffuse[10, h_idx] = float(np.log(prob_h))
    lpz_diffuse[10, 0] = -5.0
    # Final 'a' at t=11..13:
    lpz_diffuse[11:13, char_list.index("a")] = 0.0
    lpz_diffuse[13:15, 0] = 0.0  # final blank

    gt_mat, utt_indices = prepare_text(CtcSegmentationParameters(char_list=char_list), [word], char_list)

    # 1. Without threshold gating (intrusive_min_logprobs=None, low penalty):
    # Diffuse noise is mistakenly captured as an intrusive detour
    config_ungated = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.01,
        intrusive_min_logprobs=None,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_ungated = ctc_segmentation(config_ungated, lpz_diffuse, gt_mat)
    assert "h" in states_ungated, "Expected ungated trellis to capture diffuse noise"

    # 2. With scalar threshold intrusive_min_logprobs=0.40:
    # Since prob_h < 0.40 (ln(prob_h) < ln(0.40)), diffuse noise is rejected
    config_gated_scalar = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.01,
        intrusive_min_logprobs=0.40,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    timings_scalar, probs_scalar, states_scalar = ctc_segmentation(config_gated_scalar, lpz_diffuse, gt_mat)
    assert "h" not in states_scalar, f"Expected diffuse noise P={prob_h} to be suppressed by scalar threshold"

    # 3. With dictionary threshold intrusive_min_logprobs={"h": 0.40}:
    config_gated_dict = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.01,
        intrusive_min_logprobs={"h": 0.40},
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    timings_dict, probs_dict, states_dict = ctc_segmentation(config_gated_dict, lpz_diffuse, gt_mat)
    assert "h" not in states_dict, f"Expected diffuse noise P={prob_h} to be suppressed by dict threshold"

    # Verify segmentation confidence remains valid
    segments = determine_utterance_segments(config_gated_scalar, utt_indices, probs_scalar, timings_scalar, [word])
    assert segments[0][2] > -3.0


@pytest.mark.parametrize("token,word,peak_prob", [
    ("h", "akeya", 0.70),
    ("'", "tsanvsv", 0.70),
    ("h", "akeya", 0.95),
    ("'", "tsanvsv", 0.55),
])
def test_genuine_acoustic_peaks_acceptance(token, word, peak_prob):
    """AC #2: Genuine acoustic peaks (e.g. P>=0.40) trigger correct intrusive alignment and state reconstruction."""
    char_list = ["•"] + sorted(list(set(word + "h'")))
    tok_idx = char_list.index(token)

    # Construct spoken tokens containing intrusive token
    if token == "h":
        spoken_chars = ["a", "k", "e", "y", "h", "a"]
    else:
        spoken_chars = ["t", "s", "a", "'", "n", "v", "s", "v"]

    gt_mat, utt_indices = prepare_text(CtcSegmentationParameters(char_list=char_list), [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    # Set the intrusive peak frame probability to peak_prob
    intrusive_t = 2 + spoken_chars.index(token)
    lpz[intrusive_t, tok_idx] = float(np.log(peak_prob))
    lpz[intrusive_t, 0] = -5.0

    # Config with threshold 0.40 (peak_prob >= 0.40 -> passes gating)
    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=[token],
        intrusive_penalty=0.2,
        intrusive_min_logprobs=0.40,
        min_window_size=50,
        score_min_mean_over_L=2,
    )

    timings, probs, states = ctc_segmentation(config, lpz, gt_mat)

    # Verify token is accepted in states
    assert token in states, f"Expected genuine acoustic peak P={peak_prob} for {token} to be accepted"

    # Verify state reconstruction order
    non_empty = [s for s in states if s and s != config.self_transition]
    assert token in non_empty
    tok_pos = non_empty.index(token)
    if token == "h":
        assert non_empty.index("y") < tok_pos < non_empty.index("a", tok_pos)
    else:
        assert non_empty.index("a") < tok_pos < non_empty.index("n", tok_pos)

    # Verify segments confidence
    segments = determine_utterance_segments(config, utt_indices, probs, timings, [word])
    assert segments[0][2] > -1.0


def test_heterogeneous_per_token_thresholds_and_penalties():
    """AC #3: Heterogeneous per-token thresholds and penalties for differential gating between glottal stop and aspiration."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h'")))
    h_idx = char_list.index("h")
    g_idx = char_list.index("'")

    gt_mat, _ = prepare_text(CtcSegmentationParameters(char_list=char_list), [word], char_list)

    # Base spoken frames: [blank, blank, a, k, e, y, candidate_frame, a, blank, blank]
    def build_candidate_lpz(prob_h, prob_g):
        total_frames = 10
        lpz = np.full((total_frames, len(char_list)), -10.0, dtype=np.float32)
        lpz[0:2, 0] = 0.0
        lpz[2, char_list.index("a")] = 0.0
        lpz[3, char_list.index("k")] = 0.0
        lpz[4, char_list.index("e")] = 0.0
        lpz[5, char_list.index("y")] = 0.0
        lpz[6, h_idx] = float(np.log(prob_h)) if prob_h > 0 else -20.0
        lpz[6, g_idx] = float(np.log(prob_g)) if prob_g > 0 else -20.0
        lpz[6, 0] = -5.0
        lpz[7, char_list.index("a")] = 0.0
        lpz[8:10, 0] = 0.0
        return lpz

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_min_logprobs={"h": 0.40, "'": 0.20},
        intrusive_penalties={"h": 1.0, "'": 0.1},
        min_window_size=50,
        score_min_mean_over_L=2,
    )

    # Scenario A: Diffuse aspiration P=0.30, glottal stop P=0.30
    # h threshold is 0.40 (0.30 < 0.40 -> h gated out)
    # ' threshold is 0.20 (0.30 >= 0.20 -> ' passes gating and penalty is 0.1)
    lpz_a = build_candidate_lpz(prob_h=0.30, prob_g=0.30)
    _, _, states_a = ctc_segmentation(config, lpz_a, gt_mat)
    assert "'" in states_a, "Expected glottal stop to pass 0.20 threshold"
    assert "h" not in states_a, "Expected aspiration P=0.30 to be gated out by 0.40 threshold"

    # Scenario B: Strong aspiration P=0.85, weak glottal energy P=0.25
    # Both pass thresholds (0.85 >= 0.40, 0.25 >= 0.20)
    # Score h: ln(0.85) - 1.0 = -0.1625 - 1.0 = -1.1625
    # Score ': ln(0.25) - 0.1 = -1.3863 - 0.1 = -1.4863
    # h has higher score (-1.1625 > -1.4863) -> h is chosen
    lpz_b = build_candidate_lpz(prob_h=0.85, prob_g=0.25)
    _, _, states_b = ctc_segmentation(config, lpz_b, gt_mat)
    assert "h" in states_b, "Expected high-probability aspiration to win over weak glottal stop"
    assert "'" not in states_b, "Expected glottal stop to lose score competition against strong aspiration"

    # Scenario C: Sub-threshold energy for both (P=0.30 for h, P=0.15 for ')
    # 0.30 < 0.40 and 0.15 < 0.20 -> both suppressed
    lpz_c = build_candidate_lpz(prob_h=0.30, prob_g=0.15)
    _, _, states_c = ctc_segmentation(config, lpz_c, gt_mat)
    assert "h" not in states_c, "Expected sub-threshold h to be suppressed"
    assert "'" not in states_c, "Expected sub-threshold ' to be suppressed"


def test_site_restricted_transitions_masking():
    """AC #4: Site-restricted transitions using is_intrusive_site mask blocks detours at non-designated sites."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h")))
    h_idx = char_list.index("h")
    # Ground truth: '#'(0), '·'(1), 'a'(2), 'k'(3), 'e'(4), 'y'(5), 'a'(6), '·'(7)
    gt_mat, _ = prepare_text(CtcSegmentationParameters(char_list=char_list), [word], char_list)
    L = len(gt_mat)
    assert L == 8

    # Audio with high energy for 'h' at TWO sites:
    # Site 1: between 'k' (t=3) and 'e' (t=5), candidate at t=4
    # Site 2: between 'y' (t=6) and 'a' (t=8), candidate at t=7
    # Frames: [blank, blank, a, k, h_site1, e, y, h_site2, a, blank, blank]
    total_frames = 11
    lpz = np.full((total_frames, len(char_list)), -10.0, dtype=np.float32)
    lpz[0:2, 0] = 0.0
    lpz[2, char_list.index("a")] = 0.0
    lpz[3, char_list.index("k")] = 0.0
    lpz[4, h_idx] = 0.0  # Strong 'h' energy at site 1 (before 'e', column 4)
    lpz[5, char_list.index("e")] = 0.0
    lpz[6, char_list.index("y")] = 0.0
    lpz[7, h_idx] = 0.0  # Strong 'h' energy at site 2 (before 'a', column 6)
    lpz[8, char_list.index("a")] = 0.0
    lpz[9:11, 0] = 0.0

    # 1. Mask allows ONLY site 2 (column 6 before final 'a')
    mask_only_site2 = np.zeros(L, dtype=np.int8)
    mask_only_site2[6] = 1
    config_site2 = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.1,
        is_intrusive_site=mask_only_site2,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_site2 = ctc_segmentation(config_site2, lpz, gt_mat)
    non_empty_site2 = [s for s in states_site2 if s and s != config_site2.self_transition]
    # 'h' count must be exactly 1
    assert non_empty_site2.count("h") == 1
    # 'h' must be after 'y' and before 'a', NOT between 'k' and 'e'
    k_pos = non_empty_site2.index("k")
    e_pos = non_empty_site2.index("e")
    y_pos = non_empty_site2.index("y")
    h_pos = non_empty_site2.index("h")
    a_pos = non_empty_site2.index("a", h_pos)
    assert k_pos + 1 == e_pos, "Expected no 'h' between 'k' and 'e'"
    assert y_pos < h_pos < a_pos, "Expected 'h' between 'y' and 'a'"

    # 2. Mask allows ONLY site 1 (column 4 before 'e')
    mask_only_site1 = np.zeros(L, dtype=np.int8)
    mask_only_site1[4] = 1
    config_site1 = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.1,
        is_intrusive_site=mask_only_site1,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_site1 = ctc_segmentation(config_site1, lpz, gt_mat)
    non_empty_site1 = [s for s in states_site1 if s and s != config_site1.self_transition]
    assert non_empty_site1.count("h") == 1
    k_pos1 = non_empty_site1.index("k")
    h_pos1 = non_empty_site1.index("h")
    e_pos1 = non_empty_site1.index("e")
    y_pos1 = non_empty_site1.index("y")
    a_pos1 = non_empty_site1.index("a", y_pos1)
    assert k_pos1 < h_pos1 < e_pos1, "Expected 'h' between 'k' and 'e'"
    assert y_pos1 + 1 == a_pos1, "Expected no 'h' between 'y' and 'a'"

    # 3. Mask is all zeros (blocks ALL sites)
    mask_all_blocked = np.zeros(L, dtype=np.int8)
    config_blocked = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_penalty=0.1,
        is_intrusive_site=mask_all_blocked,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_blocked = ctc_segmentation(config_blocked, lpz, gt_mat)
    assert "h" not in states_blocked, "Expected all detours to be blocked when is_intrusive_site is all zeros"


def test_baseline_alignment_bitwise_identical_regression():
    """AC #5: Regression tests confirming baseline alignment without gating remains bitwise identical."""
    words = ["akeya", "tsanvsv", "adalenisgv"]
    char_list = ["•"] + sorted(list(set("".join(words) + "h'")))

    # Generate synthetic audio emissions for all words
    spoken_chars = ["a", "k", "e", "y", "a", "•", "t", "s", "a", "n", "v", "s", "v", "•", "a", "d", "a", "l", "e", "n", "i", "s", "g", "v"]
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=2, blank_frames=3)

    # 1. Baseline parameters (default)
    config_base = CtcSegmentationParameters(
        char_list=char_list,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    gt_base, utt_base = prepare_text(config_base, words, char_list)
    timings_base, probs_base, states_base = ctc_segmentation(config_base, lpz, gt_base)
    segs_base = determine_utterance_segments(config_base, utt_base, probs_base, timings_base, words)

    # 2. intrusive_tokens=None
    config_none = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=None,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    gt_none, utt_none = prepare_text(config_none, words, char_list)
    timings_none, probs_none, states_none = ctc_segmentation(config_none, lpz, gt_none)
    segs_none = determine_utterance_segments(config_none, utt_none, probs_none, timings_none, words)

    # 3. intrusive_tokens=[]
    config_empty = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=[],
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    gt_empty, utt_empty = prepare_text(config_empty, words, char_list)
    timings_empty, probs_empty, states_empty = ctc_segmentation(config_empty, lpz, gt_empty)
    segs_empty = determine_utterance_segments(config_empty, utt_empty, probs_empty, timings_empty, words)

    # 4. intrusive_min_logprobs set but intrusive_tokens=None
    config_unused_gating = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=None,
        intrusive_min_logprobs=0.40,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    gt_gated, utt_gated = prepare_text(config_unused_gating, words, char_list)
    timings_gated, probs_gated, states_gated = ctc_segmentation(config_unused_gating, lpz, gt_gated)
    segs_gated = determine_utterance_segments(config_unused_gating, utt_gated, probs_gated, timings_gated, words)

    # 5. Bitwise identical assertions across all non-gated / baseline runs
    np.testing.assert_array_equal(timings_base, timings_none)
    np.testing.assert_array_equal(timings_base, timings_empty)
    np.testing.assert_array_equal(timings_base, timings_gated)

    np.testing.assert_array_equal(probs_base, probs_none)
    np.testing.assert_array_equal(probs_base, probs_empty)
    np.testing.assert_array_equal(probs_base, probs_gated)

    assert states_base == states_none
    assert states_base == states_empty
    assert states_base == states_gated

    assert segs_base == segs_none
    assert segs_base == segs_empty
    assert segs_base == segs_gated



