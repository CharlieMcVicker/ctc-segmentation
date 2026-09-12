#!/usr/bin/env python3
# encoding: utf-8

"""Test suite and benchmarks for syncope-aware CTC trellis using Cherokee examples."""

import csv
import os
import time
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


FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "medial_vowel_syncope_examples.csv"
)


def load_syncope_examples():
    """Load Cherokee medial vowel syncope examples from test fixture."""
    examples = []
    with open(FIXTURE_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(row)
    return examples


def make_emissions(spoken_chars, char_list, frames_per_char=3, blank_frames=2):
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


def test_explicit_vowel_spoken_cherokee():
    """Test 1: When citation vowels are spoken, standard path is taken with >0.0s duration."""
    examples = load_syncope_examples()
    for row in examples:
        word = row["syllabary_transliteration"].lower()
        dropped_vowel = row["dropped_vowel"].lower()

        char_list = ["•"] + sorted(list(set(word)))
        config = CtcSegmentationParameters(
            char_list=char_list,
            syncope_tokens=[dropped_vowel],
            syncope_penalty=0.25,
            min_window_size=max(50, len(word) * 4 + 10),
            score_min_mean_over_L=2,
        )

        gt_mat, utt_indices = prepare_text(config, [word], char_list)
        lpz_spoken = make_emissions(list(word), char_list, frames_per_char=3, blank_frames=2)

        timings, char_probs, states = ctc_segmentation(config, lpz_spoken, gt_mat)

        # In explicit pronunciation, dropped vowel must appear in state_list
        assert dropped_vowel in states, f"Expected '{dropped_vowel}' in states for word '{word}'"

        # Vowel state should receive non-zero timing
        vowel_indices = [
            i for i, ch in enumerate(config.char_list) if ch == dropped_vowel
        ]
        # At least one ground truth position matching dropped_vowel should have duration > 0
        vowel_gt_positions = [
            i for i in range(len(gt_mat)) if config.is_syncope_token[i] == 1
        ]
        assert any(timings[pos] > 0.0 for pos in vowel_gt_positions), (
            f"Vowel '{dropped_vowel}' received 0.0s duration in explicit pronunciation of '{word}'"
        )


def test_syncopated_vowel_omitted_cherokee():
    """Test 2: When medial vowel is omitted, syncope skip is selected with high confidence."""
    examples = load_syncope_examples()
    for row in examples:
        word = row["syllabary_transliteration"].lower()
        dropped_vowel = row["dropped_vowel"].lower()

        # Build surface spoken sequence by removing the first occurrence of the dropped vowel
        # between consonants (matching medial syncope)
        vowel_idx_in_word = word.find(dropped_vowel, 1)  # medial position
        if vowel_idx_in_word == -1:
            vowel_idx_in_word = word.find(dropped_vowel)
        spoken_word_chars = list(word)
        del spoken_word_chars[vowel_idx_in_word]

        char_list = ["•"] + sorted(list(set(word)))
        config = CtcSegmentationParameters(
            char_list=char_list,
            syncope_tokens=[dropped_vowel],
            syncope_penalty=0.25,
            min_window_size=max(50, len(word) * 4 + 10),
            score_min_mean_over_L=2,
        )

        gt_mat, utt_indices = prepare_text(config, [word], char_list)
        lpz_syncopated = make_emissions(
            spoken_word_chars, char_list, frames_per_char=3, blank_frames=2
        )

        timings, char_probs, states = ctc_segmentation(config, lpz_syncopated, gt_mat)

        # In syncopated pronunciation, the skipped vowel state in ground truth gets 0.0s timing
        # Find which ground truth position corresponds to the skipped vowel
        # The ground truth string starts with '#·' (indices 0, 1), so word char index k is at k + 2
        skipped_gt_pos = vowel_idx_in_word + 2
        assert timings[skipped_gt_pos] == 0.0, (
            f"Expected skipped vowel at pos {skipped_gt_pos} to have 0.0s timing in '{word}'"
        )

        # Segments confidence score must remain high (>= 0.90 confidence / > -0.5 log-prob)
        segments = determine_utterance_segments(
            config, utt_indices, char_probs, timings, [word]
        )
        start, end, conf = segments[0]
        assert conf > -0.5, (
            f"Utterance '{word}' confidence {conf} degraded despite valid syncope"
        )


def test_benchmark_cherokee_syncope_vs_baseline():
    """Test 3: Benchmark syncope-aware trellis vs baseline on Cherokee syncope corpus."""
    examples = load_syncope_examples()
    assert len(examples) == 10

    syncope_confidences = []
    baseline_confidences = []
    start_time = time.perf_counter()

    for row in examples:
        word = row["syllabary_transliteration"].lower()
        dropped_vowel = row["dropped_vowel"].lower()

        vowel_idx_in_word = word.find(dropped_vowel, 1)
        if vowel_idx_in_word == -1:
            vowel_idx_in_word = word.find(dropped_vowel)
        spoken_word_chars = list(word)
        del spoken_word_chars[vowel_idx_in_word]

        char_list = ["•"] + sorted(list(set(word)))

        # 1. Syncope-aware run
        config_syncope = CtcSegmentationParameters(
            char_list=char_list,
            syncope_tokens=[dropped_vowel],
            syncope_penalty=0.25,
            min_window_size=max(50, len(word) * 4 + 10),
            score_min_mean_over_L=2,
        )
        gt_mat_s, utt_indices_s = prepare_text(config_syncope, [word], char_list)
        lpz = make_emissions(spoken_word_chars, char_list, frames_per_char=3, blank_frames=2)
        timings_s, probs_s, _ = ctc_segmentation(config_syncope, lpz, gt_mat_s)
        segs_s = determine_utterance_segments(config_syncope, utt_indices_s, probs_s, timings_s, [word])
        syncope_confidences.append(segs_s[0][2])

        # 2. Baseline run (no syncope awareness)
        config_base = CtcSegmentationParameters(
            char_list=char_list,
            syncope_tokens=None,
            min_window_size=max(50, len(word) * 4 + 10),
            score_min_mean_over_L=2,
        )
        gt_mat_b, utt_indices_b = prepare_text(config_base, [word], char_list)
        timings_b, probs_b, _ = ctc_segmentation(config_base, lpz, gt_mat_b)
        segs_b = determine_utterance_segments(config_base, utt_indices_b, probs_b, timings_b, [word])
        baseline_confidences.append(segs_b[0][2])

    elapsed = time.perf_counter() - start_time

    # Verification:
    # 1. Syncope-aware trellis produces consistently higher confidence under syncope
    assert np.mean(syncope_confidences) > np.mean(baseline_confidences)
    for s_conf, b_conf in zip(syncope_confidences, baseline_confidences):
        assert s_conf >= b_conf

    # 2. Runtime complexity remains O(T x S) and fast (< 1.0s for all 10 benchmarks)
    assert elapsed < 1.0, f"Benchmark took {elapsed:.3f}s, expected < 1.0s"


def test_blank_constrained_contiguous_syncope():
    """Test 4: Verify contiguous character word with syncope aligns successfully with 0.0s timing."""
    word = "atalenihskv"
    dropped_vowel = "i"
    spoken_chars = list("atalenhskv")

    char_list = ["•"] + sorted(list(set(word)))
    config = CtcSegmentationParameters(
        char_list=char_list,
        syncope_tokens=[dropped_vowel],
        syncope_penalty=0.25,
        min_window_size=max(50, len(word) * 4 + 10),
        score_min_mean_over_L=2,
    )
    gt_mat, utt_indices = prepare_text(config, [word], char_list)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=3, blank_frames=2)
    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)
    segs = determine_utterance_segments(config, utt_indices, char_probs, timings, [word])

    # Ground truth format: '# atalenihskv '
    # Index 0: '#', Index 1: ' ', Index 2: 'a', ..., Index 7: 'n', Index 8: 'i', Index 9: 'h', ..., Index 13: ' '
    vowel_idx = 8
    n_idx = 7
    h_idx = 9

    # 1. Dropped vowel index gets 0.0s timing
    assert timings[vowel_idx] == 0.0, (
        f"Expected dropped vowel 'i' at index {vowel_idx} to have 0.0s timing, got {timings[vowel_idx]}"
    )

    # 2. Surrounding consonants 'n' and 'h' receive positive durations / timestamps
    assert timings[n_idx] > 0.0, f"Expected 'n' at index {n_idx} to have positive timing, got {timings[n_idx]}"
    assert timings[h_idx] > 0.0, f"Expected 'h' at index {h_idx} to have positive timing, got {timings[h_idx]}"
    duration_n = timings[h_idx] - timings[n_idx]
    assert duration_n > 0.0, f"Expected positive duration between 'n' and 'h', got {duration_n}"

    # 3. Utterance confidence is high (> -0.5)
    start, end, conf = segs[0]
    assert conf > -0.5, f"Utterance '{word}' confidence {conf} degraded despite valid syncope"


def test_blank_constrained_consonant_protection():
    """Test 5: Verify non-blank consonant tokens are protected against accidental syncope skipping."""
    # Word: yihstv (contains consonant 'y' and syncope tokens 'i', 'v')
    # Spoken audio: ohsta (contains 'o', 'h', 's', 't', 'a')
    # In earlier unconstrained implementation, the trellis jumped from [PAD] (c=1) directly to 'h' (c=4),
    # bypassing both consonant 'y' (c=2) and vowel 'i' (c=3).
    # With blank constraints, c-2 is 'y' (not blank), so the 2-step jump c-3 -> c is prohibited.
    typo_word = "yihstv"
    spoken_typo = list("ohsta")
    typo_char_list = ["•"] + sorted(list(set(typo_word + "ohsta")))
    config_typo = CtcSegmentationParameters(
        char_list=typo_char_list,
        syncope_tokens=["i", "v"],
        syncope_penalty=2.0,
        min_window_size=max(50, len(typo_word) * 4 + 10),
        score_min_mean_over_L=2,
    )
    gt_mat_t, utt_indices_t = prepare_text(config_typo, [typo_word], typo_char_list)
    lpz_t = make_emissions(spoken_typo, typo_char_list, frames_per_char=3, blank_frames=2)
    timings_t, char_probs_t, states_t = ctc_segmentation(config_typo, lpz_t, gt_mat_t)
    segs_t = determine_utterance_segments(config_typo, utt_indices_t, char_probs_t, timings_t, [typo_word])

    # Verify consonant 'y' is not skipped via a syncope jump
    # 'y' is at index 2. Because 2-step jump over 'y'+'i' is prohibited,
    # the alignment cannot match 'ohsta' without penalizing the consonant mismatch.
    assert segs_t[0][2] < -2.0, f"Expected severely degraded confidence for consonant mismatch, got {segs_t[0][2]}"
    assert "y" in states_t or -10.0 in char_probs_t, "Consonant 'y' should not be silently bypassed"


def test_blank_constrained_inter_character_blanks_compatibility():
    """Test 6: Verify blank-mediated skips work seamlessly when intermediate states are blanks."""
    # Test representation with explicit inter-character blanks: C1 - [PAD] - V - [PAD] - C2
    char_list = ["•", "k", "a", "t"]
    config = CtcSegmentationParameters(
        char_list=char_list,
        blank=0,
        syncope_tokens=["a"],
        syncope_penalty=0.25,
        replace_spaces_with_blanks=True,
        min_window_size=50,
        score_min_mean_over_L=2,
    )

    # 1. Using prepare_tokenized_text with replace_spaces_with_blanks=True
    # Text 'k a t' produces ground truth: [#] - [PAD] - [PAD] - k - [PAD] - a - [PAD] - t - [PAD]
    gt_mat, utt_indices = prepare_tokenized_text(config, ["k a t"])
    spoken = ["k", "t"]
    lpz = make_emissions(spoken, char_list, frames_per_char=3, blank_frames=4)
    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)
    segs = determine_utterance_segments(config, utt_indices, char_probs, timings, ["k a t"])

    # Intermediate blank (index 4), syncope vowel 'a' (index 5), and trailing blank (index 6) are skipped
    assert timings[4] == 0.0, f"Expected intermediate blank at index 4 to have 0.0s timing, got {timings[4]}"
    assert timings[5] == 0.0, f"Expected syncope vowel 'a' at index 5 to have 0.0s timing, got {timings[5]}"
    assert timings[6] == 0.0, f"Expected intermediate blank at index 6 to have 0.0s timing, got {timings[6]}"
    assert timings[3] > 0.0, f"Expected 'k' at index 3 to have positive timing, got {timings[3]}"
    assert timings[7] > 0.0, f"Expected 't' at index 7 to have positive timing, got {timings[7]}"
    assert segs[0][2] > -0.5, f"Expected high confidence for valid syncope with inter-character blanks, got {segs[0][2]}"

    # 2. Explicit ground truth token array with C1 - [PAD] - V - [PAD] - C2
    # c=0: #, c=1: [PAD], c=2: k, c=3: [PAD], c=4: a (syncope), c=5: [PAD], c=6: t, c=7: [PAD]
    gt_mat_explicit = np.array([[-1], [0], [1], [0], [2], [0], [3], [0]], dtype=np.int64)
    is_syncope_explicit = np.zeros(len(gt_mat_explicit), dtype=np.int64)
    is_syncope_explicit[4] = 1  # 'a' is syncope token

    timings_ex, char_probs_ex, states_ex = ctc_segmentation(
        config, lpz, gt_mat_explicit, is_syncope_token=is_syncope_explicit
    )
    assert timings_ex[3] == 0.0  # leading blank skipped
    assert timings_ex[4] == 0.0  # syncope vowel 'a' skipped
    assert timings_ex[5] == 0.0  # trailing blank skipped
    assert timings_ex[2] > 0.0   # 'k' aligned
    assert timings_ex[6] > 0.0   # 't' aligned


def test_blank_constrained_word_boundary_syncope_case_b():
    """Test 7: Verify Case B blank-mediated syncope transitions at word boundaries."""
    char_list = ["•", "a", "d", "h", "i", "l", "o", "t"]
    config = CtcSegmentationParameters(
        char_list=char_list,
        blank=0,
        syncope_tokens=["i"],
        syncope_penalty=0.25,
        min_window_size=50,
        score_min_mean_over_L=2,
    )

    spoken_chars = list("dothla")  # 'doti hla' with dropped final 'i'
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=3, blank_frames=2)

    # 1. Test 1-step Case B skip (c - 3 -> c):
    # Word 1: 'doti', Word 2: 'hla' (separated by space/blank)
    # c=0: #, c=1: BLANK, c=2: d, c=3: o, c=4: t, c=5: i (syncope), c=6: BLANK, c=7: h, c=8: l, c=9: a, c=10: BLANK
    # Arriving at c=7 ('h'): c-1 is BLANK (c=6), c-2 is syncope 'i' (c=5).
    # 1-step skip jumps from c-3=4 ('t') to c=7 ('h').
    gt_mat_1step = np.array([
        [-1],  # 0: #
        [0],   # 1: BLANK
        [2],   # 2: d
        [6],   # 3: o
        [7],   # 4: t
        [4],   # 5: i (syncope)
        [0],   # 6: BLANK
        [3],   # 7: h
        [5],   # 8: l
        [1],   # 9: a
        [0],   # 10: BLANK
    ], dtype=np.int64)
    is_syncope_1step = np.zeros(len(gt_mat_1step), dtype=np.int64)
    is_syncope_1step[5] = 1

    timings_1, char_probs_1, states_1 = ctc_segmentation(
        config, lpz, gt_mat_1step, is_syncope_token=is_syncope_1step
    )
    assert timings_1[5] == 0.0, f"Expected final vowel 'i' at c=5 to be skipped, got {timings_1[5]}"
    assert timings_1[6] == 0.0, f"Expected word-boundary blank at c=6 to be skipped, got {timings_1[6]}"
    assert timings_1[4] > 0.0, f"Expected 't' at c=4 to be aligned, got {timings_1[4]}"
    assert timings_1[7] > 0.0, f"Expected 'h' at c=7 to be aligned, got {timings_1[7]}"

    # 2. Test 2-step Case B skip (c - 4 -> c):
    # Same word boundary with leading blank before final vowel:
    # c=6: 't', c=7: BLANK (leading), c=8: 'i' (syncope), c=9: BLANK (trailing), c=10: 'h'
    # Arriving at c=10 ('h'): c-1=9 (BLANK), c-2=8 ('i', syncope), c-3=7 (BLANK).
    # 2-step skip jumps from c-4=6 ('t') to c=10 ('h').
    gt_mat_2step = np.array([
        [-1],  # 0: #
        [0],   # 1: BLANK
        [2],   # 2: d
        [0],   # 3: BLANK
        [6],   # 4: o
        [0],   # 5: BLANK
        [7],   # 6: t
        [0],   # 7: BLANK (leading blank)
        [4],   # 8: i (syncope)
        [0],   # 9: BLANK (trailing blank)
        [3],   # 10: h
        [0],   # 11: BLANK
        [5],   # 12: l
        [0],   # 13: BLANK
        [1],   # 14: a
        [0],   # 15: BLANK
    ], dtype=np.int64)
    is_syncope_2step = np.zeros(len(gt_mat_2step), dtype=np.int64)
    is_syncope_2step[8] = 1

    timings_2, char_probs_2, states_2 = ctc_segmentation(
        config, lpz, gt_mat_2step, is_syncope_token=is_syncope_2step
    )
    assert timings_2[7] == 0.0, f"Expected leading blank at c=7 to be skipped, got {timings_2[7]}"
    assert timings_2[8] == 0.0, f"Expected vowel 'i' at c=8 to be skipped, got {timings_2[8]}"
    assert timings_2[9] == 0.0, f"Expected trailing blank at c=9 to be skipped, got {timings_2[9]}"
    assert timings_2[6] > 0.0, f"Expected 't' at c=6 to be aligned, got {timings_2[6]}"
    assert timings_2[10] > 0.0, f"Expected 'h' at c=10 to be aligned, got {timings_2[10]}"



