#!/usr/bin/env python3
# encoding: utf-8

"""Test suite and verification matrix for intrusive token transitions using Relative Contrastive Acoustic Gating."""

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
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    assert "h" in states, f"Expected intrusive 'h' in states, got: {states}"

    non_empty_states = [s for s in states if s and s != config.self_transition]
    assert "h" in non_empty_states
    y_idx = non_empty_states.index("y")
    h_idx = non_empty_states.index("h")
    a_idx = non_empty_states.index("a", h_idx)
    assert y_idx < h_idx < a_idx

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
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=2, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

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

    spoken_chars_h = ["a", "k", "e", "y", "h", "a"]
    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz_h = make_emissions(spoken_chars_h, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz_h, gt_mat)
    non_empty = [s for s in states if s and s != config.self_transition]

    assert "h" in non_empty
    assert "'" not in non_empty
    intrusive_count = sum(1 for s in non_empty if s in ["h", "'"])
    assert intrusive_count == 1


def test_combined_syncope_and_intrusion():
    """Test 5: Syncope vowel omission and intrusive consonant insertion interact correctly."""
    word = "adalenisgv"
    spoken_chars = ["a", "d", "l", "e", "n", "i", "h", "s", "g", "v"]
    char_list = ["•"] + sorted(list(set(word + "h")))

    config = CtcSegmentationParameters(
        char_list=char_list,
        syncope_tokens=["a"],
        intrusive_tokens=["h"],
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    assert "h" in states, f"Expected 'h' to be inserted, got: {states}"
    assert config.is_syncope_token[4] == 1
    assert timings[4] == 0.0, f"Expected dropped medial 'a' at pos 4 to have 0.0s timing, got: {timings[4]}"

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
    token_array = np.array([1, 2, 3, 4, 1], dtype=np.int64)

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        min_window_size=50,
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_token_list(config, [token_array])
    spoken_chars = ["a", "k", "e", "y", "h", "a"]
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=2, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

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
        min_window_size=50,
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_tokenized_text(config, tokenized_text)
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
        min_window_size=20,
        score_min_mean_over_L=2,
    )

    text = [word1, word2, word1]
    gt_mat, utt_indices = prepare_text(config, text, char_list)

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

    assert "h" in states, f"Expected 'h' in states, got: {states}"
    assert "'" in states, f"Expected \"'\" in states, got: {states}"

    segments = determine_utterance_segments(config, utt_indices, char_probs, timings, text)
    assert len(segments) == 3
    for i, seg in enumerate(segments):
        assert seg[2] > -1.0, f"Utterance {i} segment confidence too low: {seg[2]}"


@pytest.mark.parametrize("delta", [0, 1, 2, 3, 4])
def test_blank_tolerant_intrusive_strides(delta):
    """Test 10: Intrusive token 'h' is detected across 0, 1, 2, 3, 4 intervening CTC blank frames."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h")))
    spoken_chars = ["a", "k", "e", "y", "h"] + ["•"] * delta + ["a"]

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_max_stride=4,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

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
    spoken_chars = ["a", "k", "e", "y", "h", "•", "•", "•", "•", "•", "a"]

    config_stride4 = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        intrusive_max_stride=4,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config_stride4, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    _, _, states_stride4 = ctc_segmentation(config_stride4, lpz, gt_mat)
    assert "h" not in states_stride4

    config_stride5 = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
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
    spoken_chars = ["a", "•", "•", "k", "•", "e", "•", "•", "•", "y", "•", "•", "a"]

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'"],
        intrusive_max_stride=4,
        min_window_size=max(50, len(word) * 4 + 20),
        score_min_mean_over_L=2,
    )

    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)

    assert "h" not in states, f"Expected 'h' not to be in states, got: {states}"
    assert "'" not in states, f"Expected \"'\" not to be in states, got: {states}"

    for ch in ["a", "k", "e", "y"]:
        assert ch in states
    segments = determine_utterance_segments(config, utt_indices, char_probs, timings, [word])
    assert segments[0][2] > -1.0


def test_trellis_runtime_context_compile_structure():
    """Test TrellisRuntimeContext.compile produces valid C-contiguous arrays without penalty fields."""
    char_list = ["•", "a", "k", "e", "y", "h", "'"]
    ground_truth = np.zeros((8, 2), dtype=np.int64)

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h", "'", 2],
        is_syncope_token=[0, 1, 0, 0, 0, 0, 0, 0],
        is_intrusive_site=[0, 0, 1, 1, 0, 0, 1, 0],
        intrusive_max_stride=3,
    )
    ctx = TrellisRuntimeContext.compile(config, ground_truth)

    assert np.array_equal(ctx.intrusive_token_ids, np.array([5, 6, 2], dtype=np.int64))
    assert ctx.intrusive_token_ids.flags.c_contiguous

    assert np.array_equal(ctx.is_syncope_token, np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8))
    assert ctx.is_syncope_token.flags.c_contiguous

    assert np.array_equal(ctx.is_intrusive_site, np.array([0, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8))
    assert ctx.is_intrusive_site.flags.c_contiguous

    assert ctx.intrusive_max_stride == 3
    assert ctx.blank == 0


def test_trellis_runtime_context_compile_size_validation():
    """Test TrellisRuntimeContext.compile validates dimension matches against ground_truth."""
    char_list = ["•", "a", "k", "e", "y"]
    ground_truth = np.zeros((6, 2), dtype=np.int64)

    config_bad_syncope = CtcSegmentationParameters(
        char_list=char_list,
        is_syncope_token=[0, 1, 0, 0],
    )
    with pytest.raises(ValueError, match="is_syncope_token length .* does not match ground_truth length"):
        TrellisRuntimeContext.compile(config_bad_syncope, ground_truth)

    config_bad_site = CtcSegmentationParameters(
        char_list=char_list,
        is_intrusive_site=[True, False, False, False, False, False, True],
    )
    with pytest.raises(ValueError, match="is_intrusive_site length .* does not match ground_truth length"):
        TrellisRuntimeContext.compile(config_bad_site, ground_truth)

    config_2d = CtcSegmentationParameters(
        char_list=char_list,
        is_syncope_token=np.zeros((6, 2), dtype=np.int8),
    )
    with pytest.raises(ValueError, match="is_syncope_token length"):
        TrellisRuntimeContext.compile(config_2d, ground_truth)

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

    config = CtcSegmentationParameters(char_list=char_list)
    ctx = TrellisRuntimeContext.compile(config, ground_truth)

    assert ctx.intrusive_token_ids.shape == (0,)
    assert ctx.intrusive_token_ids.dtype == np.int64
    assert ctx.is_syncope_token.shape == (0,)
    assert ctx.is_syncope_token.dtype == np.int8
    assert ctx.is_intrusive_site.shape == (0,)
    assert ctx.is_intrusive_site.dtype == np.int8
    assert ctx.intrusive_max_stride == 4
    assert ctx.blank == 0


def test_trellis_runtime_context_compile_type_errors():
    """Test TrellisRuntimeContext.compile error handling for invalid token types or parameters."""
    ground_truth = np.zeros((4, 2), dtype=np.int64)

    config_missing = CtcSegmentationParameters(
        char_list=["•", "a"],
        intrusive_tokens=["z"],
    )
    with pytest.raises(ValueError, match="Intrusive token 'z' not found in char_list"):
        TrellisRuntimeContext.compile(config_missing, ground_truth)

    config_invalid_token = CtcSegmentationParameters(
        char_list=["•", "a"],
        intrusive_tokens=[3.14],
    )
    with pytest.raises(TypeError, match="Unsupported intrusive token type"):
        TrellisRuntimeContext.compile(config_invalid_token, ground_truth)


def test_cython_fill_table_relative_contrast_gating():
    """Test cython_fill_table with Relative Contrastive Gating for intrusive detours."""
    ground_truth = np.array([[0], [1]], dtype=np.int64)
    lpz = np.full((5, 3), -10.0, dtype=np.float32)
    lpz[0, 0] = 0.0  # t=0 blank
    lpz[1, 2] = -2.0  # t=1 'h' (detour token)
    lpz[1, 1] = -5.0  # t=1 'a' (target anchor has -5.0, so 'h' > 'a' -> gate OPEN)
    lpz[2, 1] = 0.0  # t=2 'a'

    offsets = np.zeros(2, dtype=np.int64)
    is_syncope_token = np.zeros(2, dtype=np.int8)
    intrusive_token_ids = np.array([2], dtype=np.int64)
    is_intrusive_site = np.zeros(0, dtype=np.int8)

    table = np.full((5, 2), -1e10, dtype=np.float32)
    cython_fill_table(
        table,
        lpz,
        ground_truth,
        offsets,
        is_syncope_token,
        intrusive_token_ids,
        is_intrusive_site,
        4,
        0,
        0,
    )

    assert np.isclose(table[2, 1], -2.0, atol=1e-5)


def test_cython_fill_table_site_masking():
    """Test cython_fill_table respects is_intrusive_site masking array."""
    ground_truth = np.array([[0], [1], [2]], dtype=np.int64)
    lpz = np.full((6, 4), -10.0, dtype=np.float32)
    lpz[0, 0] = 0.0  # t=0 blank
    lpz[1, 3] = 0.0  # t=1 intrusive 'h' (id 3)
    lpz[1, 1] = -5.0 # t=1 target anchor 'a' (id 1)
    lpz[2, 1] = 0.0  # t=2 'a' (id 1)

    offsets = np.zeros(3, dtype=np.int64)
    is_syncope_token = np.zeros(3, dtype=np.int8)
    intrusive_token_ids = np.array([3], dtype=np.int64)

    site_mask_blocked = np.array([0, 0, 1], dtype=np.int8)
    table_blocked = np.full((6, 3), -1e10, dtype=np.float32)
    cython_fill_table(
        table_blocked,
        lpz,
        ground_truth,
        offsets,
        is_syncope_token,
        intrusive_token_ids,
        site_mask_blocked,
        4,
        0,
        0,
    )

    site_mask_allowed = np.array([0, 1, 1], dtype=np.int8)
    table_allowed = np.full((6, 3), -1e10, dtype=np.float32)
    cython_fill_table(
        table_allowed,
        lpz,
        ground_truth,
        offsets,
        is_syncope_token,
        intrusive_token_ids,
        site_mask_allowed,
        4,
        0,
        0,
    )

    assert table_allowed[2, 1] > table_blocked[2, 1]
    assert np.isclose(table_allowed[2, 1], 0.0, atol=1e-5)


def test_intrusive_site_masking_end_to_end():
    """End-to-end test verifying is_intrusive_site restricts detours to specified positions."""
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h")))
    gt_mat, utt_indices = prepare_text(CtcSegmentationParameters(char_list=char_list), [word], char_list)
    L = len(gt_mat)

    spoken_chars = ["a", "k", "e", "y", "h", "a"]
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)

    site_mask_blocked = np.ones(L, dtype=np.int8)
    site_mask_blocked[6] = 0
    config_blocked = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        is_intrusive_site=site_mask_blocked,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_blocked = ctc_segmentation(config_blocked, lpz, gt_mat)
    assert "h" not in states_blocked

    site_mask_allowed = np.zeros(L, dtype=np.int8)
    site_mask_allowed[6] = 1
    config_allowed = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        is_intrusive_site=site_mask_allowed,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_allowed = ctc_segmentation(config_allowed, lpz, gt_mat)
    assert "h" in states_allowed


def test_relative_contrastive_intrusive_invariants():
    """Verify Relative Contrastive Intrusive Gating invariants:
    
    1. Detour triggers when P(intrusive) > P(target_anchor) even when intrusive posterior peak is sub-0.5.
    2. Detour is strictly blocked when P(target_anchor) >= P(intrusive) (tie or anchor advantage).
    3. Detour is strictly blocked on pure noise / blank frames.
    """
    word = "akeya"
    char_list = ["•"] + sorted(list(set(word + "h")))
    h_idx = char_list.index("h")
    a_idx = char_list.index("a")

    gt_mat, utt_indices = prepare_text(CtcSegmentationParameters(char_list=char_list), [word], char_list)

    # Invariant 1: Low absolute peak (e.g. logprob -2.0, ~0.135 linear prob), but strictly > target anchor (-6.0)
    spoken_chars = ["a", "k", "e", "y", "h", "a"]
    lpz_subpeak = make_emissions(spoken_chars, char_list, frames_per_char=1, blank_frames=2)
    lpz_subpeak[6, h_idx] = -2.0
    lpz_subpeak[6, a_idx] = -6.0

    config = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=["h"],
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    _, _, states_subpeak = ctc_segmentation(config, lpz_subpeak, gt_mat)
    assert "h" in states_subpeak, "Expected sub-0.5 intrusive peak to trigger detour when P(h) > P(a)"

    # Invariant 2: Anchor favored or tied: lpz[t_detour, a] >= lpz[t_detour, h] -> detour blocked
    lpz_tied = lpz_subpeak.copy()
    lpz_tied[6, a_idx] = -1.5
    _, _, states_tied = ctc_segmentation(config, lpz_tied, gt_mat)
    assert "h" not in states_tied, "Expected detour to be blocked when target anchor is favored"

    # Invariant 3: Pure noise / blank frames without candidate energy
    spoken_clean = ["a", "k", "e", "y", "a"]
    lpz_clean = make_emissions(spoken_clean, char_list, frames_per_char=2, blank_frames=2)
    _, _, states_clean = ctc_segmentation(config, lpz_clean, gt_mat)
    assert "h" not in states_clean, "Expected no spurious intrusive detours on clean speech"


def test_baseline_alignment_bitwise_identical_regression():
    """Regression tests confirming baseline alignment without gating remains bitwise identical."""
    words = ["akeya", "tsanvsv", "adalenisgv"]
    char_list = ["•"] + sorted(list(set("".join(words) + "h'")))

    spoken_chars = ["a", "k", "e", "y", "a", "•", "t", "s", "a", "n", "v", "s", "v", "•", "a", "d", "a", "l", "e", "n", "i", "s", "g", "v"]
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=2, blank_frames=3)

    config_base = CtcSegmentationParameters(
        char_list=char_list,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    gt_base, utt_base = prepare_text(config_base, words, char_list)
    timings_base, probs_base, states_base = ctc_segmentation(config_base, lpz, gt_base)
    segs_base = determine_utterance_segments(config_base, utt_base, probs_base, timings_base, words)

    config_none = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=None,
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    gt_none, utt_none = prepare_text(config_none, words, char_list)
    timings_none, probs_none, states_none = ctc_segmentation(config_none, lpz, gt_none)
    segs_none = determine_utterance_segments(config_none, utt_none, probs_none, timings_none, words)

    config_empty = CtcSegmentationParameters(
        char_list=char_list,
        intrusive_tokens=[],
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    gt_empty, utt_empty = prepare_text(config_empty, words, char_list)
    timings_empty, probs_empty, states_empty = ctc_segmentation(config_empty, lpz, gt_empty)
    segs_empty = determine_utterance_segments(config_empty, utt_empty, probs_empty, timings_empty, words)

    np.testing.assert_array_equal(timings_base, timings_none)
    np.testing.assert_array_equal(timings_base, timings_empty)

    np.testing.assert_array_equal(probs_base, probs_none)
    np.testing.assert_array_equal(probs_base, probs_empty)

    assert states_base == states_none
    assert states_base == states_empty

    assert segs_base == segs_none
    assert segs_base == segs_empty
