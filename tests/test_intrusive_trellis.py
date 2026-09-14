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
