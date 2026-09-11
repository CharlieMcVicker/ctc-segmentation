#!/usr/bin/env false
# encoding: utf-8

# Copyright 2020, Technische Universität München; Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test functions for CTC segmentation."""
import numpy as np
import pytest

from ctc_segmentation import ctc_segmentation
from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import determine_utterance_segments
from ctc_segmentation import prepare_text
from ctc_segmentation import prepare_tokenized_text
from ctc_segmentation import prepare_token_list


def test_ctcsegmentationparameters():
    """Test the configuration object.
    Test repr and init.
    """
    config = CtcSegmentationParameters()
    config = eval(str(config))
    assert config.index_duration == 0.025
    config.index_duration = 0.030
    assert config.index_duration == 0.030
    # test excluded parameters and update procedure
    config.set(char_list=["a", "»"])
    config.update_excluded_characters()
    assert "»" not in config.excluded_characters


def test_legacy_timing_parameters_deprecation():
    """Test deprecation warnings and backward compatibility for legacy timing parameters."""
    config = CtcSegmentationParameters()

    # Accessing index_duration_in_seconds emits DeprecationWarning
    with pytest.deprecated_call():
        _ = config.index_duration_in_seconds

    # Setting index_duration_in_seconds emits DeprecationWarning
    with pytest.deprecated_call():
        config.index_duration_in_seconds = 0.040
    assert config.index_duration == 0.040

    # Setting subsampling_factor emits DeprecationWarning
    with pytest.deprecated_call():
        config.subsampling_factor = 4

    # Accessing subsampling_factor emits DeprecationWarning
    with pytest.deprecated_call():
        assert config.subsampling_factor == 4

    # Setting frame_duration_ms emits DeprecationWarning and updates index_duration
    with pytest.deprecated_call():
        config.frame_duration_ms = 30
    assert config.index_duration == (30 * 4) / 1000.0

    # Accessing frame_duration_ms emits DeprecationWarning
    with pytest.deprecated_call():
        assert config.frame_duration_ms == 30

    # Test initializing via kwargs with legacy parameters
    with pytest.deprecated_call():
        config_legacy = CtcSegmentationParameters(subsampling_factor=2, frame_duration_ms=20)
    assert config_legacy.index_duration == (20 * 2) / 1000.0


def test_ctc_segmentation():
    """Test CTC segmentation.

    This is a minimal example for the function.
    Only executes CTC segmentation, does not check its result.
    """
    config = CtcSegmentationParameters()
    config.min_window_size = 20
    config.max_window_size = 50
    char_list = ["•", "a", "c", "d", "g", "o", "s", "t"]
    text = ["catzz#\n", "dogs!!\n"]
    # lpz = torch.nn.functional.log_softmax(torch.rand(30, 8) * 10, dim=0).numpy()
    ground_truth_mat, utt_begin_indices = prepare_text(config, text, char_list)
    timings, char_probs, state_list = ctc_segmentation(config, lpz, ground_truth_mat)


def test_determine_utterance_segments():
    """Test the generation of segments from aligned utterances.

    This is a function that is used after a completed CTC segmentation.
    Results are checked and compared with test vectors.
    """
    config = CtcSegmentationParameters()
    frame_duration_ms = 1000
    config.index_duration = frame_duration_ms / 1000.0
    config.score_min_mean_over_L = 2
    utt_begin_indices = [1, 4, 9]
    text = ["catzz#\n", "dogs!!\n"]
    char_probs = np.array([-0.5] * 10)
    timings = np.array(list(range(10))) + 0.5
    segments = determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, text
    )
    correct_segments = [(2.0, 4.0, -0.5), (5.0, 9.0, -0.5)]
    for i in [0, 1]:
        for j in [0, 1, 2]:
            assert segments[i][j] == correct_segments[i][j]


def test_prepare_text():
    """Test the prepare_text function for CTC segmentation.

    Results are checked and compared with test vectors.
    """
    config = CtcSegmentationParameters()
    text = ["catzz#\n", "dogs!!\n"]
    char_list = ["•", "a", "c", "d", "g", "o", "s", "t"]
    ground_truth_mat, utt_begin_indices = prepare_text(config, text, char_list)
    correct_begin_indices = np.array([1, 5, 10])
    assert (utt_begin_indices == correct_begin_indices).all()
    gtm = list(ground_truth_mat.shape)
    assert gtm[0] == 11
    assert gtm[1] == 1
    # test scan for longer tokens:
    text = ["cat"]
    char_list = ["•", "UNK", "a", "c", "t", "cat"]
    ground_truth_mat, utt_begin_indices = prepare_text(config, text, char_list)
    gtm = list(ground_truth_mat.shape)
    assert gtm[0] == 6
    assert gtm[1] == 3


def test_prepare_tokenized_text():
    """Test the prepare_tokenized_text function for CTC segmentation.

    Results are checked and compared with test vectors.
    """
    text = ["c a t", "d ▁o ▁g ▁s"]
    char_list = ["•", "a", "c", "d", "▁g", "▁o", "▁s", "t"]
    config = CtcSegmentationParameters(char_list=char_list)
    ground_truth_mat, utt_begin_indices = prepare_tokenized_text(config, text)
    correct_begin_indices = np.array([1, 5, 10])
    assert (utt_begin_indices == correct_begin_indices).all()
    gtm = list(ground_truth_mat.shape)
    assert gtm[0] == 11
    assert gtm[1] == 1


def test_prepare_token_list():
    """Test the prepare_token_list function for CTC segmentation.

    Results are checked and compared with test vectors.
    """
    text = [np.array([2, 1, 7]), np.array([3, 5, 4, 6])]
    char_list = ["•", "a", "c", "d", "▁g", "▁o", "▁s", "t"]
    config = CtcSegmentationParameters(char_list=char_list)
    ground_truth_mat, utt_begin_indices = prepare_token_list(config, text)
    correct_begin_indices = np.array([1, 5, 10])
    assert (utt_begin_indices == correct_begin_indices).all()
    gtm = list(ground_truth_mat.shape)
    assert gtm[0] == 11
    assert gtm[1] == 1


# pre-generated test vectors
lpz = np.array(
    [
        [
            -1.9890659,
            -6.910831,
            -5.693124,
            -2.8735375,
            -2.5746322,
            -3.7570968,
            -6.505041,
            -7.800645,
        ],
        [
            -1.7459257,
            -8.443403,
            -9.054435,
            -6.091851,
            -1.1048597,
            -4.3298893,
            -3.6350899,
            -4.132761,
        ],
        [
            -1.9080026,
            -7.994824,
            -9.81665,
            -2.3486533,
            -5.144716,
            -3.9509172,
            -3.4352026,
            -1.2714918,
        ],
        [
            -6.0218654,
            -2.3527913,
            -2.2818222,
            -4.691431,
            -8.936862,
            -6.176718,
            -9.35063,
            -3.822922,
        ],
        [
            -6.7574806,
            -4.8557367,
            -7.597179,
            -6.810881,
            -7.2958636,
            -2.3951168,
            -7.7496943,
            -2.4941995,
        ],
        [
            -4.045436,
            -1.1840547,
            -2.3596387,
            -6.391866,
            -9.6217985,
            -7.970184,
            -2.97404,
            -1.4489534,
        ],
        [
            -8.723544,
            -9.255755,
            -4.9860573,
            -5.4689684,
            -4.178754,
            -4.4266634,
            -1.6171856,
            -6.532046,
        ],
        [
            -5.7916913,
            -8.874264,
            -8.35385,
            -7.554833,
            -2.7915673,
            -6.53148,
            -7.262638,
            -4.068927,
        ],
        [
            -1.9035804,
            -8.733719,
            -3.5118732,
            -9.5878725,
            -2.337254,
            -5.6119165,
            -9.185156,
            -10.189388,
        ],
        [
            -2.9709957,
            -11.0104,
            -5.8517113,
            -4.0744276,
            -5.278929,
            -4.3865757,
            -7.6332912,
            -6.560225,
        ],
        [
            -8.324375,
            -7.9097023,
            -4.4599323,
            -7.7892103,
            -9.1231165,
            -2.0423908,
            -4.377398,
            -10.835497,
        ],
        [
            -10.399205,
            -7.0444527,
            -5.371065,
            -1.2489381,
            -5.8032174,
            -2.7301397,
            -8.445712,
            -3.8961184,
        ],
        [
            -2.0746524,
            -4.541919,
            -8.762662,
            -9.938227,
            -3.8826694,
            -5.6540346,
            -8.945148,
            -3.1916835,
        ],
        [
            -5.8310924,
            -3.471004,
            -5.153735,
            -2.415791,
            -5.1635947,
            -9.231514,
            -4.1059637,
            -2.7528045,
        ],
        [
            -5.7406664,
            -1.8533367,
            -5.225171,
            -6.8159046,
            -5.9029193,
            -6.623233,
            -4.1038485,
            -9.242478,
        ],
        [
            -3.882025,
            -7.318694,
            -8.598673,
            -8.664008,
            -8.898863,
            -4.3000784,
            -9.741696,
            -2.5367324,
        ],
        [
            -8.534433,
            -6.4304566,
            -1.5769805,
            -8.969663,
            -3.539075,
            -0.91964996,
            -6.275173,
            -2.4531362,
        ],
        [
            -10.100832,
            -1.9878258,
            -9.781347,
            -2.4888206,
            -6.2522135,
            -6.343619,
            -7.033285,
            -3.0782526,
        ],
        [
            -5.0670514,
            -3.3480282,
            -2.4745665,
            -3.039238,
            -10.691722,
            -9.94559,
            -7.566962,
            -9.439356,
        ],
        [
            -2.5350397,
            -9.904655,
            -3.815092,
            -6.5622272,
            -4.3727484,
            -4.5448284,
            -7.3634896,
            -8.524196,
        ],
        [
            -6.907628,
            -4.4899416,
            -1.2235631,
            -3.7986655,
            -6.103579,
            -6.596727,
            -11.327395,
            -6.719469,
        ],
        [
            -10.498164,
            -6.086135,
            -5.3307266,
            -2.8573642,
            -1.9187597,
            -7.7122536,
            -9.413016,
            -10.007352,
        ],
        [
            -4.31647,
            -2.97263,
            -5.1576066,
            -5.9061184,
            -4.530726,
            -10.311597,
            -2.7961264,
            -6.780219,
        ],
        [
            -10.060461,
            -6.929871,
            -4.6684146,
            -2.2593799,
            -2.1629434,
            -8.561601,
            -1.3917265,
            -5.724318,
        ],
        [
            -8.468343,
            -3.0233464,
            -5.2083797,
            -6.3359613,
            -7.7919903,
            -6.32028,
            -11.001884,
            -10.480761,
        ],
        [
            -8.077727,
            -9.722239,
            -4.501517,
            -4.7871294,
            -5.916735,
            -2.1889973,
            -2.3767185,
            -7.748427,
        ],
        [
            -4.550388,
            -8.701884,
            -5.8193216,
            -10.3321705,
            -3.7262502,
            -8.329333,
            -5.845203,
            -9.304822,
        ],
        [
            -5.4920406,
            -3.4807057,
            -7.677996,
            -2.2778478,
            -4.0280805,
            -2.5542955,
            -1.5931826,
            -9.432675,
        ],
        [
            -7.612656,
            -2.683886,
            -9.083887,
            -7.212092,
            -4.4599934,
            -5.9059615,
            -3.591928,
            -9.783908,
        ],
        [
            -3.0536897,
            -6.5981,
            -2.4680572,
            -6.5821176,
            -8.253022,
            -9.725112,
            -6.0701623,
            -7.134845,
        ],
    ],
    dtype=np.float32,
)


def test_optional_vowel_mask():
    """Test the generation of is_optional_vowel mask across preparation methods."""
    char_list = ["•", "a", "c", "d", "g", "o", "s", "t"]
    config = CtcSegmentationParameters(
        char_list=char_list,
        optional_vowel_tokens=["a", "o"],
        syncopy_penalty=0.3,
    )
    # Test prepare_text
    text = ["cat", "dog"]
    gt_mat, utt_indices = prepare_text(config, text, char_list)
    assert config.is_optional_vowel is not None
    assert len(config.is_optional_vowel) == gt_mat.shape[0]
    # Ground truth: '#', '·', 'c', 'a', 't', '·', 'd', 'o', 'g', '·'
    # 'a' is at index 3, 'o' is at index 7
    assert config.is_optional_vowel[3] == 1
    assert config.is_optional_vowel[7] == 1
    assert config.is_optional_vowel[2] == 0  # 'c'
    assert config.is_optional_vowel[4] == 0  # 't'

    # Test prepare_token_list
    tok_list = [np.array([2, 1, 7]), np.array([3, 5, 4])]  # [c, a, t], [d, o, g]
    gt_mat, utt_indices = prepare_token_list(config, tok_list)
    assert config.is_optional_vowel is not None
    assert len(config.is_optional_vowel) == gt_mat.shape[0]
    # Ground truth: [-1, 0, 2(c), 1(a), 7(t), 0, 3(d), 5(o), 4(g), 0]
    # 1(a) is at index 3, 5(o) is at index 7
    assert config.is_optional_vowel[3] == 1
    assert config.is_optional_vowel[7] == 1
    assert config.is_optional_vowel[2] == 0  # 2(c)
    assert config.is_optional_vowel[4] == 0  # 7(t)


def test_syncopy_trellis_with_vowel_skip():
    """Test syncopy trellis when vowel is spoken vs omitted."""
    char_list = ["•", "c1", "v1", "c2"]
    # Blank is 0, c1 is 1, v1 is 2, c2 is 3
    config = CtcSegmentationParameters(
        char_list=char_list,
        optional_vowel_tokens=["v1"],
        syncopy_penalty=0.25,
        min_window_size=30,
        score_min_mean_over_L=2,
    )
    tokens = [np.array([1, 2, 3])]  # c1, v1, c2
    gt_mat, utt_indices = prepare_token_list(config, tokens)
    # gt_mat: [-1, 0(blank), 1(c1), 2(v1), 3(c2), 0(blank)]
    # indices:   0,        1,     2,     3,     4,        5
    assert config.is_optional_vowel[3] == 1

    # Case 1: Vowel is spoken in audio (c1 -> v1 -> c2)
    # Timesteps: 0..2 blank, 3..5 c1, 6..8 v1, 9..11 c2, 12..14 blank
    T = 15
    V = len(char_list)
    lpz_spoken = np.full((T, V), -10.0, dtype=np.float32)
    lpz_spoken[0:3, 0] = 0.0   # blank
    lpz_spoken[3:6, 1] = 0.0   # c1
    lpz_spoken[6:9, 2] = 0.0   # v1
    lpz_spoken[9:12, 3] = 0.0  # c2
    lpz_spoken[12:15, 0] = 0.0 # blank

    timings_1, probs_1, states_1 = ctc_segmentation(config, lpz_spoken, gt_mat)
    # In spoken case, v1 (state 3) should have non-zero timing and appear in states
    assert "v1" in states_1
    assert timings_1[3] > 0.0

    # Case 2: Vowel is omitted/syncope'd in audio (c1 -> c2 directly, v1 skipped)
    # Timesteps: 0..2 blank, 3..6 c1, 7..10 c2, 11..13 blank
    T2 = 14
    lpz_omitted = np.full((T2, V), -10.0, dtype=np.float32)
    lpz_omitted[0:3, 0] = 0.0   # blank
    lpz_omitted[3:7, 1] = 0.0   # c1
    lpz_omitted[7:11, 3] = 0.0  # c2 (v1 omitted, prob is -10)
    lpz_omitted[11:14, 0] = 0.0 # blank

    timings_2, probs_2, states_2 = ctc_segmentation(config, lpz_omitted, gt_mat)
    # In syncopated case, v1 (state 3) should be skipped (duration 0.0, not in states)
    assert "v1" not in states_2
    assert "c1" in states_2
    assert "c2" in states_2
    assert timings_2[3] == 0.0  # skipped vowel has 0 timing

    # Utterance segment confidence should remain high despite dropped vowel
    segments = determine_utterance_segments(config, utt_indices, probs_2, timings_2, ["c1 v1 c2"])
    start, end, conf = segments[0]
    assert conf > -1.0  # High confidence


def test_syncopy_trellis_text_preparation():
    """Test syncopy trellis using prepare_text with character sequences."""
    char_list = ["•", "c", "a", "t"]
    config = CtcSegmentationParameters(
        char_list=char_list,
        optional_vowel_tokens=["a"],
        syncopy_penalty=0.3,
        min_window_size=30,
        score_min_mean_over_L=2,
    )
    text = ["cat"]
    gt_mat, utt_indices = prepare_text(config, text, char_list)
    # ground_truth: '#', '·', 'c', 'a', 't', '·'
    # indices:       0,   1,   2,   3,   4,   5
    assert config.is_optional_vowel[3] == 1  # 'a' is optional

    # Audio emits 'c' -> 't' (skipping 'a')
    T = 12
    V = len(char_list)
    lpz = np.full((T, V), -10.0, dtype=np.float32)
    lpz[0:2, 0] = 0.0   # blank
    lpz[2:6, 1] = 0.0   # 'c'
    lpz[6:10, 3] = 0.0  # 't' ('a' is skipped)
    lpz[10:12, 0] = 0.0 # blank

    timings, probs, states = ctc_segmentation(config, lpz, gt_mat)
    assert "a" not in states
    assert "c" in states
    assert "t" in states
    assert timings[3] == 0.0  # 'a' skipped

    segments = determine_utterance_segments(config, utt_indices, probs, timings, text)
    start, end, conf = segments[0]
    assert conf > -1.0



