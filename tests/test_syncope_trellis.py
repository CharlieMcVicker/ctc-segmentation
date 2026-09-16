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
    typo_word = "yihstv"
    spoken_typo = list("ohsta")
    typo_char_list = ["•"] + sorted(list(set(typo_word + "ohsta")))
    config_typo = CtcSegmentationParameters(
        char_list=typo_char_list,
        syncope_tokens=["i", "v"],
        min_window_size=max(50, len(typo_word) * 4 + 10),
        score_min_mean_over_L=2,
    )
    gt_mat_t, utt_indices_t = prepare_text(config_typo, [typo_word], typo_char_list)
    lpz_t = make_emissions(spoken_typo, typo_char_list, frames_per_char=3, blank_frames=2)
    timings_t, char_probs_t, states_t = ctc_segmentation(config_typo, lpz_t, gt_mat_t)
    segs_t = determine_utterance_segments(config_typo, utt_indices_t, char_probs_t, timings_t, [typo_word])

    # Verify consonant 'y' is not skipped via a syncope jump
    assert segs_t[0][2] < -2.0, f"Expected severely degraded confidence for consonant mismatch, got {segs_t[0][2]}"
    assert "y" in states_t or -10.0 in char_probs_t, "Consonant 'y' should not be silently bypassed"


def test_blank_constrained_inter_character_blanks_compatibility():
    """Test 6: Verify blank-mediated skips work seamlessly when intermediate states are blanks."""
    char_list = ["•", "k", "a", "t"]
    config = CtcSegmentationParameters(
        char_list=char_list,
        blank=0,
        syncope_tokens=["a"],
        replace_spaces_with_blanks=True,
        min_window_size=50,
        score_min_mean_over_L=2,
    )

    # 1. Using prepare_tokenized_text with replace_spaces_with_blanks=True
    gt_mat, utt_indices = prepare_tokenized_text(config, ["k a t"])
    spoken = ["k", "t"]
    lpz = make_emissions(spoken, char_list, frames_per_char=3, blank_frames=4)
    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)
    segs = determine_utterance_segments(config, utt_indices, char_probs, timings, ["k a t"])

    assert timings[4] == 0.0, f"Expected intermediate blank at index 4 to have 0.0s timing, got {timings[4]}"
    assert timings[5] == 0.0, f"Expected syncope vowel 'a' at index 5 to have 0.0s timing, got {timings[5]}"
    assert timings[6] == 0.0, f"Expected intermediate blank at index 6 to have 0.0s timing, got {timings[6]}"
    assert timings[3] > 0.0, f"Expected 'k' at index 3 to have positive timing, got {timings[3]}"
    assert timings[7] > 0.0, f"Expected 't' at index 7 to have positive timing, got {timings[7]}"
    assert segs[0][2] > -0.5, f"Expected high confidence for valid syncope with inter-character blanks, got {segs[0][2]}"

    # 2. Explicit ground truth token array with C1 - [PAD] - V - [PAD] - C2
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
        min_window_size=50,
        score_min_mean_over_L=2,
    )

    spoken_chars = list("dothla")  # 'doti hla' with dropped final 'i'
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=3, blank_frames=2)

    # 1. Test 1-step Case B skip (c - 3 -> c):
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


def test_relative_contrastive_syncope_gating_invariants():
    """Test 8: Verify Relative Contrastive Syncope Gating invariants.
    
    1. Vowel skip is strictly blocked when P(vowel) >= P(anchor) at the arrival frame.
    2. Vowel skip is open and evaluated without penalty when P(anchor) > P(vowel).
    """
    char_list = ["•", "k", "a", "t"]  # 0: blank, 1: k, 2: a (syncope), 3: t
    config = CtcSegmentationParameters(
        char_list=char_list,
        blank=0,
        syncope_tokens=["a"],
        min_window_size=50,
        score_min_mean_over_L=2,
    )
    gt_mat, utt_indices = prepare_text(config, ["kat"], char_list)

    # Scenario A: Spoken 'k a t' where vowel 'a' is spoken and favored during its frames
    lpz_spoken = make_emissions(["k", "a", "t"], char_list, frames_per_char=3, blank_frames=2)
    timings_a, char_probs_a, states_a = ctc_segmentation(config, lpz_spoken, gt_mat)
    # Vowel 'a' at index 3 must NOT be skipped (has non-zero timing)
    assert timings_a[3] > 0.0, f"Expected vowel 'a' to NOT be skipped when spoken, got {timings_a[3]}"
    assert timings_a[2] > 0.0, f"Expected 'k' to be aligned, got {timings_a[2]}"
    assert timings_a[4] > 0.0, f"Expected 't' to be aligned, got {timings_a[4]}"
    assert "a" in states_a

    # Scenario B: Spoken 'k t' where vowel 'a' is dropped and anchor 't' is spoken immediately after 'k'
    lpz_sync = make_emissions(["k", "t"], char_list, frames_per_char=3, blank_frames=2)
    timings_b, char_probs_b, states_b = ctc_segmentation(config, lpz_sync, gt_mat)
    # Vowel 'a' at index 3 MUST be skipped (0.0s timing)
    assert timings_b[3] == 0.0, f"Expected vowel 'a' to be skipped when omitted, got {timings_b[3]}"
    assert timings_b[2] > 0.0, f"Expected 'k' to be aligned, got {timings_b[2]}"
    assert timings_b[4] > 0.0, f"Expected 't' to be aligned, got {timings_b[4]}"


def test_inter_word_syncope_pronounced():
    """Test 9: Inter-word syncope pronounced: vowel before word boundary is preserved."""
    words = ["kanohetv", "tsisa"]
    char_list = ["•"] + sorted(list(set("kanohetvtsisa")))
    config = CtcSegmentationParameters(
        char_list=char_list,
        blank=0,
        syncope_tokens=["v"],
        min_window_size=100,
        score_min_mean_over_L=2,
    )
    gt_mat, utt_indices = prepare_text(config, words, char_list)

    # Audio contains full pronunciation: 'kanohetv' -> inter-word blank -> 'tsisa'
    V = len(char_list)
    frames_per_char = 3
    blank_frames = 2
    chars_w1 = list("kanohetv")
    chars_w2 = list("tsisa")
    total_frames = blank_frames + len(chars_w1) * frames_per_char + blank_frames + len(chars_w2) * frames_per_char + blank_frames
    lpz = np.full((total_frames, V), -10.0, dtype=np.float32)

    # Initial blank
    lpz[0:blank_frames, 0] = 0.0
    t = blank_frames
    for ch in chars_w1:
        lpz[t : t + frames_per_char, char_list.index(ch)] = 0.0
        t += frames_per_char
    # Inter-word blank
    lpz[t : t + blank_frames, 0] = 0.0
    t += blank_frames
    for ch in chars_w2:
        lpz[t : t + frames_per_char, char_list.index(ch)] = 0.0
        t += frames_per_char
    # Final blank
    lpz[t : t + blank_frames, 0] = 0.0

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)
    segs = determine_utterance_segments(config, utt_indices, char_probs, timings, words)

    # In explicit pronunciation, 'v' must not be skipped
    v_pos = [i for i in range(len(gt_mat)) if config.is_syncope_token[i] == 1]
    assert len(v_pos) > 0
    assert any(timings[p] > 0.0 for p in v_pos), "Expected word-final 'v' to have positive timing when pronounced"
    assert "v" in states
    assert segs[0][2] > -0.5
    assert segs[1][2] > -0.5


def test_inter_word_syncope_dropped():
    """Test 10: Inter-word syncope dropped: final vowel and boundary PAD skipped cleanly to C_next."""
    words = ["kanohetv", "tsisa"]
    char_list = ["•"] + sorted(list(set("kanohetvtsisa")))
    config = CtcSegmentationParameters(
        char_list=char_list,
        blank=0,
        syncope_tokens=["v"],
        min_window_size=100,
        score_min_mean_over_L=2,
    )
    gt_mat, utt_indices = prepare_text(config, words, char_list)

    # Audio drops 'v': 'kanohet' -> 'tsisa' (direct transition to onset 'ts')
    spoken_chars = list("kanohettsisa")
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=3, blank_frames=2)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)
    segs = determine_utterance_segments(config, utt_indices, char_probs, timings, words)

    # 'v' at end of kanohetv must receive 0.0s timing
    v_pos = [i for i in range(len(gt_mat)) if config.is_syncope_token[i] == 1]
    for p in v_pos:
        assert timings[p] == 0.0, f"Expected dropped word-final 'v' at pos {p} to have 0.0s timing, got {timings[p]}"

    # Surrounding consonants 't' and 't' (onset of tsisa) receive valid positive timings
    assert timings[8] > 0.0, "Expected 't' before dropped 'v' to have positive timing"
    assert timings[11] > 0.0, "Expected 't' onset of 'tsisa' to have positive timing"
    assert segs[0][2] > -0.5
    assert segs[1][2] > -0.5


def test_phrase_terminal_vowel_pronounced():
    """Test 11: Phrase-terminal vowel pronounced: vowel before trailing silence is preserved."""
    word = "atalenihskv"
    char_list = ["•"] + sorted(list(set(word)))
    config = CtcSegmentationParameters(
        char_list=char_list,
        blank=0,
        syncope_tokens=["v"],
        min_window_size=60,
        score_min_mean_over_L=2,
    )
    gt_mat, utt_indices = prepare_text(config, [word], char_list)

    # Fully articulated with 6 trailing blank frames
    spoken_chars = list(word)
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=3, blank_frames=6)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)
    segs = determine_utterance_segments(config, utt_indices, char_probs, timings, [word])

    # Terminal vowel 'v' is at the end of the word
    v_pos = [i for i in range(len(gt_mat)) if config.is_syncope_token[i] == 1]
    assert len(v_pos) > 0
    assert timings[v_pos[0]] > 0.0, f"Expected terminal 'v' to have positive timing, got {timings[v_pos[0]]}"
    assert "v" in states
    assert segs[0][2] > -0.5


def test_phrase_terminal_vowel_dropped():
    """Test 12: Phrase-terminal vowel dropped: final vowel skipped into terminal silence."""
    word = "atalenihskv"
    char_list = ["•"] + sorted(list(set(word)))
    config = CtcSegmentationParameters(
        char_list=char_list,
        blank=0,
        syncope_tokens=["v"],
        min_window_size=60,
        score_min_mean_over_L=2,
    )
    gt_mat, utt_indices = prepare_text(config, [word], char_list)

    # Final 'v' is dropped, followed by 6 trailing blank frames
    spoken_chars = list("atalenihsk")
    lpz = make_emissions(spoken_chars, char_list, frames_per_char=3, blank_frames=6)

    timings, char_probs, states = ctc_segmentation(config, lpz, gt_mat)
    segs = determine_utterance_segments(config, utt_indices, char_probs, timings, [word])

    # Terminal vowel 'v' must have 0.0s timing
    v_pos = [i for i in range(len(gt_mat)) if config.is_syncope_token[i] == 1]
    assert len(v_pos) > 0
    assert timings[v_pos[0]] == 0.0, f"Expected dropped terminal 'v' to have 0.0s timing, got {timings[v_pos[0]]}"
    assert segs[0][2] > -0.5




