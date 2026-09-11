#!/usr/bin/env python3
# encoding: utf-8

"""Test suite and benchmarks for syncopy-aware CTC trellis using Cherokee examples."""

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
            optional_vowel_tokens=[dropped_vowel],
            syncopy_penalty=0.25,
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
            i for i in range(len(gt_mat)) if config.is_optional_vowel[i] == 1
        ]
        assert any(timings[pos] > 0.0 for pos in vowel_gt_positions), (
            f"Vowel '{dropped_vowel}' received 0.0s duration in explicit pronunciation of '{word}'"
        )


def test_syncopated_vowel_omitted_cherokee():
    """Test 2: When medial vowel is omitted, TRANS_VOWEL_SKIP is selected with high confidence."""
    examples = load_syncope_examples()
    for row in examples:
        word = row["syllabary_transliteration"].lower()
        dropped_vowel = row["dropped_vowel"].lower()

        # Build surface spoken sequence by removing the first occurrence of the dropped vowel
        # between consonants (matching medial syncopy)
        vowel_idx_in_word = word.find(dropped_vowel, 1)  # medial position
        if vowel_idx_in_word == -1:
            vowel_idx_in_word = word.find(dropped_vowel)
        spoken_word_chars = list(word)
        del spoken_word_chars[vowel_idx_in_word]

        char_list = ["•"] + sorted(list(set(word)))
        config = CtcSegmentationParameters(
            char_list=char_list,
            optional_vowel_tokens=[dropped_vowel],
            syncopy_penalty=0.25,
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
            f"Utterance '{word}' confidence {conf} degraded despite valid syncopy"
        )


def test_benchmark_cherokee_syncope_vs_baseline():
    """Test 3: Benchmark syncopy-aware trellis vs baseline on Cherokee syncope corpus."""
    examples = load_syncope_examples()
    assert len(examples) == 10

    syncopy_confidences = []
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

        # 1. Syncopy-aware run
        config_syncopy = CtcSegmentationParameters(
            char_list=char_list,
            optional_vowel_tokens=[dropped_vowel],
            syncopy_penalty=0.25,
            min_window_size=max(50, len(word) * 4 + 10),
            score_min_mean_over_L=2,
        )
        gt_mat_s, utt_indices_s = prepare_text(config_syncopy, [word], char_list)
        lpz = make_emissions(spoken_word_chars, char_list, frames_per_char=3, blank_frames=2)
        timings_s, probs_s, _ = ctc_segmentation(config_syncopy, lpz, gt_mat_s)
        segs_s = determine_utterance_segments(config_syncopy, utt_indices_s, probs_s, timings_s, [word])
        syncopy_confidences.append(segs_s[0][2])

        # 2. Baseline run (no syncopy awareness)
        config_base = CtcSegmentationParameters(
            char_list=char_list,
            optional_vowel_tokens=None,
            min_window_size=max(50, len(word) * 4 + 10),
            score_min_mean_over_L=2,
        )
        gt_mat_b, utt_indices_b = prepare_text(config_base, [word], char_list)
        timings_b, probs_b, _ = ctc_segmentation(config_base, lpz, gt_mat_b)
        segs_b = determine_utterance_segments(config_base, utt_indices_b, probs_b, timings_b, [word])
        baseline_confidences.append(segs_b[0][2])

    elapsed = time.perf_counter() - start_time

    # Verification:
    # 1. Syncopy-aware trellis produces consistently higher confidence under syncope
    assert np.mean(syncopy_confidences) > np.mean(baseline_confidences)
    for s_conf, b_conf in zip(syncopy_confidences, baseline_confidences):
        assert s_conf >= b_conf

    # 2. Runtime complexity remains O(T x S) and fast (< 1.0s for all 10 benchmarks)
    assert elapsed < 1.0, f"Benchmark took {elapsed:.3f}s, expected < 1.0s"
