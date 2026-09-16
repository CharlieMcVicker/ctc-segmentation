# CTC segmentation

<!-- Badges -->
[![build status](https://github.com/lumaku/ctc-segmentation/actions/workflows/python-package.yml/badge.svg)](https://github.com/lumaku/ctc-segmentation/actions/workflows/python-package.yml)
[![version](https://img.shields.io/pypi/v/ctc-segmentation)](https://pypi.org/project/ctc-segmentation/)
[![AUR](https://img.shields.io/aur/version/python-ctc-segmentation-git)](https://aur.archlinux.org/packages/python-ctc-segmentation-git)
[![downloads](https://img.shields.io/pypi/dm/ctc-segmentation)](https://pypi.org/project/ctc-segmentation/)

CTC segmentation can be used to find utterance alignments within large audio files.

* This repository contains the `ctc-segmentation` python package.
* A description of the algorithm is in the CTC segmentation paper (on [Springer Link](https://link.springer.com/chapter/10.1007%2F978-3-030-60276-5_27), on [ArXiv](https://arxiv.org/abs/2007.09127))


# Usage

The CTC segmentation package is not standalone, as it needs a neural network with CTC output. It is integrated in these frameworks:

* In ESPnet 1 as corpus recipe: [Alignment script](https://github.com/espnet/espnet/blob/master/espnet/bin/asr_align.py), [Example recipe](https://github.com/espnet/espnet/tree/master/egs/tedlium2/align1), [Demo](https://github.com/espnet/espnet#ctc-segmentation-demo )
* In ESPnet 2, as script or directly as python interface: [Alignment script](https://github.com/espnet/espnet/blob/master/espnet2/bin/asr_align.py), [Demo](https://github.com/espnet/espnet#ctc-segmentation-demo )
* In Nvidia NeMo as dataset creation tool: [Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/ctc_segmentation.html), [Example](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/CTC_Segmentation_Tutorial.ipynb)
* In Speechbrain, as python interface: [Alignment module](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/alignment/ctc_segmentation.py), [Examples](https://gist.github.com/lumaku/75eca1c86d9467a54888d149dc7b84f1)

It can also be used with other frameworks:

<details><summary>Wav2vec2 example code</summary><div>

```python
import torch
import numpy as np
from typing import List
import ctc_segmentation
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

# load model, processor and tokenizer
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# load dummy dataset and read soundfiles
SAMPLERATE = 16000
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
audio = ds[0]["audio"]["array"]
transcripts = ["A MAN SAID TO THE UNIVERSE", "SIR I EXIST"]

def align_with_transcript(
    audio : np.ndarray,
    transcripts : List[str],
    samplerate : int = SAMPLERATE,
    model : Wav2Vec2ForCTC = model,
    processor : Wav2Vec2Processor = processor,
    tokenizer : Wav2Vec2CTCTokenizer = tokenizer
):
    assert audio.ndim == 1
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits,dim=-1)
    
    # Tokenize transcripts
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    unk_id = vocab["<unk>"]
    
    tokens = []
    for transcript in transcripts:
        assert len(transcript) > 0
        tok_ids = tokenizer(transcript.replace("\n"," ").lower())['input_ids']
        tok_ids = np.array(tok_ids,dtype=np.int)
        tokens.append(tok_ids[tok_ids != unk_id])
    
    # Align
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcripts)
    return [{"text" : t, "start" : p[0], "end" : p[1], "conf" : p[2]} for t,p in zip(transcripts, segments)]
    
def get_word_timestamps(
    audio : np.ndarray,
    samplerate : int = SAMPLERATE,
    model : Wav2Vec2ForCTC = model,
    processor : Wav2Vec2Processor = processor,
    tokenizer : Wav2Vec2CTCTokenizer = tokenizer
):
    assert audio.ndim == 1
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits,dim=-1)
        
    predicted_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.decode(predicted_ids)
    
    # Split the transcription into words
    words = pred_transcript.split(" ")
    
    # Align
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, words)
    return [{"text" : w, "start" : p[0], "end" : p[1], "conf" : p[2]} for w,p in zip(words, segments)]

print(align_with_transcript(audio,transcripts))
# [{'text': 'A MAN SAID TO THE UNIVERSE', 'start': 0.08124999999999993, 'end': 2.034375, 'conf': 0.0}, 
#  {'text': 'SIR I EXIST', 'start': 2.3260775862068965, 'end': 4.078771551724138, 'conf': 0.0}]

print(get_word_timestamps(audio))
# [{'text': 'a', 'start': 0.08124999999999993, 'end': 0.5912715517241378, 'conf': 0.9999501323699951}, 
# {'text': 'man', 'start': 0.5912715517241378, 'end': 0.9219827586206896, 'conf': 0.9409108982174931}, 
# {'text': 'said', 'start': 0.9219827586206896, 'end': 1.2326508620689656, 'conf': 0.7700278702302796}, 
# {'text': 'to', 'start': 1.2326508620689656, 'end': 1.3529094827586206, 'conf': 0.5094435178226225}, 
# {'text': 'the', 'start': 1.3529094827586206, 'end': 1.4831896551724135, 'conf': 0.4580493446392211}, 
# {'text': 'universe', 'start': 1.4831896551724135, 'end': 2.034375, 'conf': 0.9285054256219009}, 
# {'text': 'sir', 'start': 2.3260775862068965, 'end': 3.036530172413793, 'conf': 0.0}, 
# {'text': 'i', 'start': 3.036530172413793, 'end': 3.347198275862069, 'conf': 0.7995760873559864}, 
# {'text': 'exist', 'start': 3.347198275862069, 'end': 4.078771551724138, 'conf': 0.0}]
```

</div></details>

<details><summary>Syncope-aware alignment example (syncope / phonetic reduction / token dropping)</summary><div>

To align text containing optional syncopated tokens (e.g. Cherokee medial vowel syncope, consonant cluster reduction) without combinatorial string expansion:

```python
import numpy as np
import ctc_segmentation

# Configure parameters with optional syncope tokens
char_list = ["•", "a", "e", "i", "o", "u", "v", "ts", "l", "g"]
config = ctc_segmentation.CtcSegmentationParameters(
    char_list=char_list,
    syncope_tokens=["a", "e", "i", "o", "u", "v"],  # tokens that can be skipped via syncope
)

# Prepare ground truth and run alignment
# If 'a' in 'tsalagi' is omitted in speech ('tsalgi'), relative contrastive gating allows the skip when acoustics favor 'l' over 'a'
text = ["tsalagi"]
ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, text)
timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs, ground_truth_mat)
segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
```

</div></details>

<details><summary>Intrusive token alignment example (aspiration, glottal stops, phonetic insertion)</summary><div>

To align text when speakers insert unwritten surface sounds (e.g. pre-/post-aspiration `h`, glottal stop `'`, epenthetic consonants) absent from citation transcripts:

```python
import numpy as np
import ctc_segmentation

# Configure parameters with candidate intrusive tokens and maximum blank frame stride
char_list = ["•", "a", "k", "e", "y", "h", "'"]
config = ctc_segmentation.CtcSegmentationParameters(
    char_list=char_list,
    intrusive_tokens=["h", "'"],  # intrusive tokens permitted between ground-truth states
    intrusive_max_stride=4,       # maximum blank frame stride bridging intrusive tokens (default: 4)
)

# Prepare ground truth and run alignment
# If citation 'akeya' is spoken as surface 'akeyha', relative contrastive gating takes a detour through 'h'
text = ["akeya"]
ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, text)

# Optional: Apply phonotactic site masking to restrict intrusions only to specific positions (e.g. post-vocalic)
# is_intrusive_site is a 1D int8/bool array matching len(ground_truth_mat)
config.is_intrusive_site = np.zeros(len(ground_truth_mat), dtype=np.int8)
config.is_intrusive_site[3] = 1  # only permit detour transition leading into state index 3

timings, char_probs, state_list = ctc_segmentation(config, probs, ground_truth_mat)
segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
```

</div></details>



# Installation

* With `pip`:
```sh
pip install ctc-segmentation
```

* From the Arch Linux AUR as `python-ctc-segmentation-git` using your favourite AUR helper.

* From source:
```sh
git clone https://github.com/lumaku/ctc-segmentation
cd ctc-segmentation
cythonize -3 ctc_segmentation/ctc_segmentation_dyn.pyx
python setup.py build
python setup.py install --optimize=1 --skip-build
```

# How it works

### 1. Forward propagation

Character probabilites from each time step are obtained from a CTC-based network.
With these, transition probabilities are mapped into a trellis diagram.
To account for preambles or unrelated segments in audio files, the transition cost are set to zero for the start-of-sentence or blank token.

![Forward trellis](doc/1_forward.png)

#### Syncope-Aware Trellis ($\epsilon$-Skip Transitions with Relative Contrastive Gating)

For languages with phonetic reductions or token syncope (e.g. Cherokee medial vowel syncope, unstressed sound deletion), sounds written in canonical text are often dropped in natural speech. Standard CTC segmentation forces alignment of every character, causing misalignments or requiring intractable $O(2^n)$ string expansions.

To support syncope in $O(T \times S)$ polynomial time while strictly protecting non-syncope characters (such as preceding consonants) from accidental omission, the trellis evaluates blank-constrained syncope-skip transitions governed by two core invariants:

1. **Consonant Preservation Invariant**: A syncope transition **MUST NOT** omit any character token whose `is_syncope_token == 0`.
2. **Blank-Only Stride Invariant**: Multi-column skips across optional syncope tokens are valid **if and only if** any additional intermediate columns are CTC blank / PAD tokens ($L = \text{blank}$ or $L = -1$).
3. **Non-Acoustic Blank Rejection Invariant**: An intermediate inter-word `PAD` (blank token) must **never** serve as a contrastive gating anchor. Inter-word syncope skips across $(V_{\text{final}} + \text{PAD})$ are evaluated when column $c$ reaches the true acoustic onset anchor of the subsequent word ($C_{\text{next}}$).
4. **Phrase-Terminal Direct Likelihood Competition**: At phrase/utterance-final boundaries ($c = \text{table.shape}[1] - 1$), transitions into terminal silence (`PAD_end`) are strictly anchored to the consonant offset boundary ($t_{\text{prev}} = t - 1$) without intermediate blank-farming, allowing pure likelihood competition between Path A ($C \to V \to \text{PAD}$) and Path B ($C \to \text{PAD}$).

##### Relative Contrastive Syncope Gate & Recurrence

Let $\text{is\_syncope\_token}[c] \in \{0, 1\}$ indicate if column $c$ is marked as an eligible syncope token. Rather than subtracting an arbitrary penalty, syncope skips are gated by local **Relative Contrastive Acoustic Evidence**: frame $t$ must exhibit stronger acoustic evidence for the incoming anchor token than for the skipped vowel:

$$\text{Gate}_{\text{syncope}}(t, c) = \begin{cases} \text{OPEN}, & \text{if } \text{lpz}[t + \text{offset\_sum}, L(c, s)] > \text{lpz}[t + \text{offset\_sum}, L(c_{\text{vowel}}, 0)] \\ \text{CLOSED}, & \text{otherwise} \end{cases}$$

When open, evaluate the transition with zero penalty ($\lambda_{\text{syncope}} = 0.0$):

* **Case A: Immediate Syncope Vowel ($c - 1$ is syncope token, $c_{\text{vowel}} = c - 1$, anchor is non-blank $L(c, s) \ne \text{blank}$)**
  1. *Direct 1-token skip ($c - 2 \to c$):*
     $$P_{\text{sync}, 1}(t, c) = \text{table}[t - 1 + \text{offset}(c, c - 2), c - 2] + \text{lpz}[t + \text{offset\_sum}, L(c, s)]$$
  2. *Blank-mediated 2-token skip ($c - 3 \to c$):*
     Permitted **only if** $c \ge 3$ and state $c - 2$ is a CTC blank/PAD token ($L(c - 2, 0) \in \{\text{blank}, -1\}$):
     $$P_{\text{sync}, 2}(t, c) = \begin{cases}
     \text{table}[t - 1 + \text{offset}(c, c - 3), c - 3] + \text{lpz}[t + \text{offset\_sum}, L(c, s)] & \text{if } L(c - 2, 0) \in \{\text{blank}, -1\} \\
     -\infty & \text{otherwise}
     \end{cases}$$

* **Case B: Inter-Word Syncope Across Blank ($c - 2$ is syncope token, $c - 1$ is blank, anchor is onset consonant $C_{\text{next}} = L(c, s) \ne \text{blank}$)**
  1. *Skip vowel + trailing blank ($c - 3 \to c$):*
     $$P_{\text{sync}, 3}(t, c) = \text{table}[t - 1 + \text{offset}(c, c - 3), c - 3] + \text{lpz}[t + \text{offset\_sum}, L(c, s)]$$
  2. *Skip leading blank + vowel + trailing blank ($c - 4 \to c$):*
     Permitted **only if** $c \ge 4$ and state $c - 3$ is a CTC blank/PAD token ($L(c - 3, 0) \in \{\text{blank}, -1\}$):
     $$P_{\text{sync}, 4}(t, c) = \begin{cases}
     \text{table}[t - 1 + \text{offset}(c, c - 4), c - 4] + \text{lpz}[t + \text{offset\_sum}, L(c, s)] & \text{if } L(c - 3, 0) \in \{\text{blank}, -1\} \\
     -\infty & \text{otherwise}
     \end{cases}$$

* **Case C: Syncope Directly into Blank ($L(c, s) = \text{blank}$, Phrase-Terminal or Inter-Word Pause)**
  When transitioning into a blank state $c$, the skip is gated at the **departure frame** $t_{\text{dep}} = t_{\text{prev}} + 1 + \text{offset}(c_{\text{prev}})$ (the immediate frame where $C_{\text{prev}}$ was exited):
  $$\text{Gate}_{\text{dep}}(t, c) = \begin{cases} \text{OPEN}, & \text{if } \text{lpz}[t_{\text{dep}}, \text{blank}] > \text{lpz}[t_{\text{dep}}, V_{\text{final}}] \\ \text{CLOSED}, & \text{otherwise} \end{cases}$$
  1. *Direct skip over final vowel into blank ($c - 2 \to c$):*
     $$P_{\text{sync}, 5}(t, c) = \text{table}[t - 1 + \text{offset}(c, c - 2), c - 2] + \text{lpz}[t + \text{offset\_sum}, \text{blank}]$$
  2. *Blank-mediated skip ($c - 3 \to c$):*
     Permitted **only if** $c \ge 3$ and state $c - 2$ is a CTC blank/PAD token ($L(c - 2, 0) \in \{\text{blank}, -1\}$):
     $$P_{\text{sync}, 6}(t, c) = \begin{cases}
     \text{table}[t - 1 + \text{offset}(c, c - 3), c - 3] + \text{lpz}[t + \text{offset\_sum}, \text{blank}] & \text{if } L(c - 2, 0) \in \{\text{blank}, -1\} \\
     -\infty & \text{otherwise}
     \end{cases}$$

#### Intrusive Token Transitions (Relative Contrastive Detours)

In many languages and dialects, speakers pronounce intrusive sounds (e.g. pre-/post-aspiration `/h/`, glottal stop `/'/`, or epenthetic consonants) that are absent from canonical citation transcripts.

##### The CTC Blank Gap Problem & Unified Relative Contrastive Gating

CTC acoustic models emit discrete posterior peaks separated by blank (`[PAD]` / $\epsilon$) frames:
```text
Frame t - 1 - δ:        'h'   (intrusive token peak)
Frames t - δ .. t - 1:  [PAD] (intervening CTC blank frames)
Frame t:                'u'   (ground truth token state c)
```

Instead of requiring fixed log-probability thresholds or hand-tuned penalties, candidate intrusive tokens $j \in J$ at frame $t_{\text{detour}} = t - 1 - \delta + \text{offset\_sum}$ are evaluated by **Relative Contrastive Acoustic Evidence**: acoustics must favor the intrusive candidate over the expected target anchor token $L(c, s)$:

$$\text{Gate}_{\text{intrusive}}(t_{\text{detour}}, c, j) = \begin{cases} \text{OPEN}, & \text{if } \text{lpz}[t_{\text{detour}}, j] > \text{lpz}[t_{\text{detour}}, L(c, s)] \\ \text{CLOSED}, & \text{otherwise} \end{cases}$$

When open, evaluate the path with zero penalty:
$$P_{\text{intrusive}}(t, c) = \begin{cases}
\max_{j \in J, 0 \le \delta \le W} \left[ \begin{aligned}
&\text{table}[t - 2 - \delta + \text{offset}, c - 1 - s] \\
&+ \text{lpz}[t_{\text{detour}}, j] \\
&+ \sum_{b=t-\delta}^{t-1} \text{lpz}[b + \text{offset\_sum}, \text{blank}] \\
&+ \text{lpz}[t + \text{offset\_sum}, L(c, s)]
\end{aligned} \right] & \text{if } M_{\text{site}}[c] = 1 \text{ and } \text{Gate}_{\text{intrusive}} \text{ is OPEN} \\
-\infty & \text{otherwise}
\end{cases}$$

where $W \ge 0$ is the maximum blank stride (`intrusive_max_stride`, default: `4`), $M_{\text{site}}[c] \in \{0, 1\}$ is the site mask (`is_intrusive_site`), and $\text{blank}$ is the CTC blank index.

* **Single-Slot Mutual Exclusivity**: Between any two ground-truth states, at most one intrusion is permitted, preventing ungrammatical stacking while cleanly reconstructing surface phonetic realisations.
* **Phonotactic Site Masking ($M_{\text{site}}$)**: Passing `is_intrusive_site` as a 1D `int8` mask array matching `len(ground_truth)` restricts detour transitions to phonotactically licensed positions ($M_{\text{site}}[c] = 1$).

##### Architecture: Userspace Config & TrellisRuntimeContext

CTC segmentation employs a clean architectural separation:
1. **Userspace Configuration (`CtcSegmentationParameters`)**: Accepts Python structures (token lists, masks, stride constraints).
2. **Compiled Runtime Arrays (`TrellisRuntimeContext`)**: Compiles configuration into C-contiguous 1D NumPy arrays (`int64`, `int8`, `int`).
3. **Zero-Overhead DP Core**: The Cython engine (`cython_fill_table`) and backtracking operate in log space with zero `exp()` calls and zero penalties.

### 2. Backtracking

Starting from the time step with the highest probability for the last character, backtracking determines the most probable path of characters through all time steps.

![Backward path](doc/2_backtracking.png)

When a syncope-skip transition was selected, backtracking recovers the jump across the omitted token, assigns `0.0s` duration to the skipped token, and does not consume audio frames for it.

When an intrusive detour transition was selected, backtracking emits the intrusive token $j^*$ at frame $t-1-\delta^*$, fills any intervening frames $[t-\delta^*, t-1]$ with blank/stay tokens ($\epsilon$), and emits the ground-truth token $L(c)$ at frame $t$, seamlessly reconstructing the surface phonetic sequence without requiring heuristic post-processing regexes.

### 3. Confidence score

As this method generates a probability for each aligned character, a confidence score for each utterance can be derived.
For example, if a word within an utterance is missing, this value is low.

![Confidence score](doc/3_scoring.png)

The confidence score helps to detect and filter-out bad utterances. For syncopated utterances, confidence is calculated over realized (spoken) frames, ensuring natural syncope does not artificially degrade utterance confidence.


# Parameters

There are several notable parameters to adjust the working of the algorithm that can be found in the class `CtcSegmentationParameters`:


### Data preparation parameters

* Localization: The character set is taken from the model dict, i.e., usually are generated with SentencePiece. An ASR model trained in the corresponding language and character set is needed. For asian languages, no changes to the CTC segmentation parameters should be necessary. One exception: If the character set contains any punctuation characters, "#", or the Greek char "ε", adapt the setting in an instance of `CtcSegmentationParameters` in `segmentation.py`.

* `CtcSegmentationParameters` includes a blank character. Copy over the Blank character from the dictionary to the configuration, if in the model dictionary e.g. "\<blank>" instead of the default "_" is used. If the Blank in the configuration and in the dictionary mismatch, the algorithm raises an IndexError at backtracking.

* If `replace_spaces_with_blanks` is True, then spaces in the ground truth sequence are replaces by blanks. This option is enabled by default and improves compability with dictionaries with unknown space characters.

### Alignment parameters

* `min_window_size`: Minimum window size considered for a single utterance. The current default value should be OK in most cases.

* To align utterances with longer unkown audio sections between them, use `blank_transition_cost_zero` (default: False). With this option, the stay transition in the blank state is free. A transition to the next character is only consumed if the probability to switch is higher. In this way, more time steps can be skipped between utterances. Caution: in combination with `replace_spaces_with_blanks == True`, this may lead to misaligned segments.

### Syncope & optional token parameters

* `syncope_tokens` (default: `None`): List of character strings or token IDs (e.g. `["a", "e", "i", "o", "u", "v"]`) that can be optionally skipped in speech via relative contrastive acoustic gating. When set, `prepare_text`, `prepare_token_list`, and `prepare_tokenized_text` automatically build the `is_syncope_token` mask. (Legacy alias: `optional_vowel_tokens`).
* `is_syncope_token` (default: `None`): 1D `int8` mask array across the interleaved state sequence marking optional syncope token positions. Can be explicitly passed or generated automatically. (Legacy alias: `is_optional_vowel`).

### Intrusive token parameters

* `intrusive_tokens` (default: `None`): Sequence of character strings or token IDs (e.g. `["h", "'"]`) eligible for intrusive detour insertion between ground-truth states when favored by relative contrastive acoustic gating.
* `intrusive_max_stride` (default: `4`): Maximum number of intervening CTC blank frames permitted between the intrusive token and the next ground-truth token.
* `is_intrusive_site` (default: `None`): 1D `int8` mask array of size `len(ground_truth)` restricting intrusive detours strictly to licensed positions.
* `is_intrusive_token` (default: `None`): 1D `int8` mask array of size `len(char_list)` marking eligible intrusive tokens in the vocabulary.

### Time stamp parameters

Directly set the parameter `index_duration` to give the corresponding time duration of one CTC output index (in seconds).

**Example:** For a given sample rate, say, 16kHz, `fs=16000`. Then, how many sample points correspond to one ctc output index? In some ASR systems, this can be calculated from the hop length of the windowing times encoder subsampling factor. For example, if the hop length of the frontend windowing is 128, and the subsampling factor in the encoder is 4, totalling 512 sample points for one CTC index. Then `index_duration = 512 / 16000`.

**Note:** In earlier versions, `index_duration` was not used and the time stamps were determined from the values of `subsampling_factor` and `frame_duration_ms`. To derive `index_duration` from these values, calculate`frame_duration_ms * subsampling_factor / 1000`.

### Confidence score parameters

Character probabilities over each L frames are accumulated to calculate the confidence score. The L value can be adapted with with `score_min_mean_over_L` . A lower L makes the score more sensitive to error in the transcription, but also errors in the ASR model.


# Toolkit Integration

CTC segmentation requires CTC activations of an already trained CTC-based network. Example code can be found in the alignment scripts `asr_align.py` of ESPnet 1 or ESpnet 2.

### Steps to alignment for regular ASR

1. First, the ground truth text need to be converted into a matrix: Use `prepare_token_list` from a ground truth sequence that was already coverted to a sequence of tokens (recommended). Alternatively, use `prepare_text` on raw text, this method filters characters not in the dictionary and can break longer tokens into smaller tokens.
2. `ctc_segmentation` computes character-wise alignments from the CTC log posterior probabilites.
3. `determine_utterance_segments` converts char-wise alignments to utterance-wise alignments.
4. In a post-processing step, segments may be filtered by their confidence value.

### Steps to alignment for different use-cases

Sometimes the ground truth data is not text, but a sequence of tokens, or a list of protein segments.
In this case, use either `prepare_token_list` or replace it with a function that suits better for your data.
For examples, see the `prepare_*` functions in `ctc_segmentation.py`, or the example included in the NeMo toolkit.

### Segments clean-up

Segments that were written to a `segments` file can be filtered using the confidence score. This is the minium confidence score in log space as described in the paper. 

Utterances with a low confidence score are discarded in a data clean-up. This parameter may need adjustment depending on dataset, ASR model and used text conversion.

```bash
min_confidence_score=1.5
awk -v ms=${min_confidence_score} '{ if ($5 > ms) {print} }' ${unfiltered} > ${filtered}
```


# FAQ

* *How do I split a large text into multiple utterances?* This can be done automatically, e.g. in our paper, the text of this book/chapter was split into utterances at sentence endings to derive utterances.

* *What if there are unrelated parts within the audio file without transcription?* Unrelated segments can be skipped with the `gratis_blank` parameter. Larger unrelated segments may deteriorate the results, try to increase the minimum window size. Partially repeating segments have a high chance to disarrange the alignments, remove them if possible. These segments can be detected with the confidence score. Use the `state_list` to see how well the unrelated part was "ignored".

* *How fast is CTC segmentation?* On a modern computer (64GB RAM, Nvidia RTX 2080 ti, AMD Ryzen 7 2700X), it takes around 400 ms to align 500s of audio with tokenized text and the default window size (8000~250s). In comparison, inference of CTC posteriors on CPU takes some time from 4s to 20s; GPU inference takes roughly 500 - 1000 ms, but often fails at such long audio files because of excessive memory consumption (tested with Transformer model on Espnet 2; this strongly depends on model architecture and used toolkit). A few factors influence CTC segmentation speed: Window size, length of audio, length of text, how well the text fits to audio and the preprocessing function. Aligning from tokenized text is faster because the alignment with `prepare_text` additionally includes transition probabilities from partial tokens; this increases the complexity by the length of the longest token in the character dictionary (including the blank token).

* *The inference of the model is too slow and uses too much memory.* Use RNN-based ASR networks that grow linearly in complexity with longer audio files. Inference complexity increases on long audio files quadratically for Transformer-based architectures. To solve this, it is possible to partition the audio into several parts. CTC segmentation includes an example partitioning function in `ctc_segmentation.get_partitions`. [Example Code in JtubeSpeech](https://github.com/sarulab-speech/jtubespeech/blob/a16cffc1d38ac23f43a230c68ea0927ed9b6ea9f/scripts/align.py#L339-L387)

* *How can I improve the alignment speed of CTC segmentation?* The alignment algorithm is not parallelizable for batch processing, so use a CPU with a good single-thread performance. It's possible to align multiple files in parallel, if the computer has enough temporary memory. The alignment is faster with shorter max token length, if text is aligned - or directly align from a token list.

* *How do I get word-based alignments instead of full utterance segments?* Use an ASR model with character tokens to improve the time-resolution. Then handle each word as single utterance.

* *How can I improve the accuracy of the generated alignments?* Be aware that depending on the ASR performance of network and other factors, CTC activations are not always accurate, sometimes shifted by a few frames. To get a better time resolution, use a dictionary with characters! Also, the `prepare_text` function tries to break down long tokens into smaller tokens.

* *What is the difference between `prepare_token_list` and `prepare_text`?* Explained in examples:

<details><summary>Example for `prepare_token_list`</summary><div>

Let's say we have a text `text = ["cat"]` and a dictionary that includes the word cat as well as its parts: `char_list = ["•", "UNK", "a", "c", "t", "cat"]`.
The "tokenize" method that uses the `preprocess_fn` will produce:

```python
text = ["cat"]
char_list = ["•", "UNK", "a", "c", "t", "cat"]
token_list = [tokenize(utt) for utt in text]
token_list
# [array([5])]
ground_truth_mat, utt_begin_indices = prepare_token_list(config, text)
ground_truth_mat
# array([[-1],
#        [ 0],
#        [ 5],
#        [ 0]])
```

</div></details>

<details><summary>Example for `prepare_text`</summary><div>
Toy example:

```python
text = ["cat"]
char_list = ["•", "UNK", "a", "c", "t", "cat"]
ground_truth_mat, utt_begin_indices = prepare_text(config, text, char_list)
# array([[-1, -1, -1],
#        [ 0, -1, -1],
#        [ 3, -1, -1],
#        [ 2, -1, -1],
#        [ 4, -1,  5],
#        [ 0, -1, -1]])
```

Here, the partial characters are detected (3,2,4), as well as the full "cat" token (5).
This is done to have a better time resolution for the alignment.

Full example with a bpe 500 model char list from Tedlium 2:

```python

from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import prepare_text

char_list = [ "<unk>", "'", "a", "ab", "able", "ace", "ach", "ack",
"act","ad","ag", "age", "ain", "al", "alk", "all", "ally", "am",
"ame", "an","and", "ans", "ant", "ap", "ar", "ard", "are",
"art","as", "ase","ass", "ast", "at", "ate", "ated", "ater", "ation",
"ations", "ause","ay", "b", "ber", "ble", "c", "ce", "cent",
"ces","ch", "ci", "ck","co", "ct", "d", "de", "du", "e", "ear",
"ect", "ed", "een", "el","ell", "em", "en", "ence", "ens",
"ent","enty", "ep", "er", "ere","ers", "es", "ess", "est", "et", "f",
"fe", "ff", "g", "ge", "gh","ght", "h", "her", "hing", "ht",
"i","ia", "ial", "ib", "ic","ical", "ice", "ich", "ict", "id", "ide",
"ie", "ies", "if","iff","ig", "ight", "ign", "il", "ild", "ill","im",
"in", "ind","ine", "ing", "ink", "int", "ion", "ions", "ip",
"ir","ire","is","ish", "ist", "it", "ite", "ith", "itt", "ittle",
"ity", "iv","ive", "ix", "iz", "j", "k", "ke", "king", "l",
"ld","le","ll","ly", "m", "ment", "ms", "n", "nd", "nder", "nt", "o",
"od","ody", "og", "ol", "olog", "om", "ome", "on",
"one","ong","oo","ood", "ook", "op", "or", "ore", "orm", "ort",
"ory", "os","ose", "ot", "other", "ou", "ould", "ound",
"ount","our","ous","ousand", "out", "ow", "own", "p", "ph", "ple",
"pp", "pt","q", "qu", "r", "ra", "rain", "re", "reat", "red",
"ree","res","ro","rou", "rough", "round", "ru", "ry", "s", "se",
"sel", "so","st","t", "ter", "th", "ther", "ty", "u", "ually",
"ud","ue", "ul","ult", "um", "un", "und", "ur", "ure", "us", "use",
"ust", "ut","v","ve", "vel", "ven", "ver", "very", "ves", "ving","w",
"way","x", "y", "z", "ăť", "ō", "▁", "▁a", "▁ab",
"▁about","▁ac","▁act","▁actually", "▁ad", "▁af", "▁ag", "▁al",
"▁all", "▁also","▁am", "▁an", "▁and", "▁any", "▁ar",
"▁are","▁around", "▁as","▁at","▁b", "▁back", "▁be", "▁bec",
"▁because", "▁been", "▁being","▁bet", "▁bl", "▁br", "▁bu",
"▁but","▁by", "▁c", "▁call","▁can","▁ch", "▁chan", "▁cl", "▁co",
"▁com", "▁comm", "▁comp","▁con", "▁cont", "▁could", "▁d",
"▁day","▁de", "▁des","▁did","▁diff", "▁differe", "▁different",
"▁dis", "▁do", "▁does","▁don", "▁down", "▁e", "▁en",
"▁even","▁every", "▁ex", "▁exp","▁f","▁fe", "▁fir", "▁first",
"▁five", "▁for", "▁fr", "▁from", "▁g","▁get", "▁go",
"▁going","▁good", "▁got", "▁h", "▁ha","▁had","▁happ", "▁has",
"▁have", "▁he", "▁her", "▁here", "▁his","▁how", "▁hum",
"▁hundred","▁i", "▁ide", "▁if", "▁im", "▁imp","▁in","▁ind", "▁int",
"▁inter", "▁into", "▁is", "▁it", "▁j", "▁just","▁k","▁kind",
"▁kn","▁know", "▁l", "▁le", "▁let", "▁li", "▁life","▁like",
"▁little", "▁lo", "▁look", "▁lot", "▁m",
"▁ma","▁make","▁man","▁many", "▁may", "▁me", "▁mo", "▁more",
"▁most","▁mu", "▁much", "▁my", "▁n", "▁ne", "▁need", "▁new",
"▁no","▁not","▁now", "▁o", "▁of", "▁on", "▁one","▁only", "▁or",
"▁other", "▁our","▁out", "▁over", "▁p", "▁part", "▁pe",
"▁peop","▁people","▁per","▁ph", "▁pl", "▁po", "▁pr", "▁pre", "▁pro",
"▁put", "▁qu", "▁r","▁re", "▁real", "▁really", "▁res",
"▁right","▁ro", "▁s","▁sa","▁said", "▁say", "▁sc", "▁se", "▁see",
"▁sh", "▁she", "▁show", "▁so","▁som", "▁some", "▁somet","▁something",
"▁sp","▁spe", "▁st","▁start", "▁su", "▁sy", "▁t", "▁ta",
"▁take","▁talk", "▁te","▁th","▁than", "▁that", "▁the", "▁their",
"▁them", "▁then", "▁there","▁these", "▁they", "▁thing",
"▁things","▁think", "▁this","▁those","▁thousand", "▁three",
"▁through", "▁tim", "▁time", "▁to", "▁tr","▁tw", "▁two", "▁u",
"▁un","▁under", "▁up", "▁us","▁v", "▁very","▁w", "▁want", "▁was",
"▁way", "▁we", "▁well", "▁were","▁wh","▁what", "▁when",
"▁where","▁which", "▁who", "▁why", "▁will","▁with", "▁wor", "▁work",
"▁world", "▁would", "▁y","▁year","▁years", "▁you", "▁your"]

text = ["I ▁really ▁like ▁technology",
 "The ▁quick ▁brown ▁fox ▁jumps ▁over ▁the ▁lazy ▁dog.",
 "unknown chars äüößß-!$ "]
config = CtcSegmentationParameters()
config.char_list = char_list
ground_truth_mat, utt_begin_indices = prepare_text(config, text)

# ground_truth_mat
# array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [190, 410,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55, 193, 411,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [137,  13,  -1,  -1, 412,  -1,  -1,  -1,  -1,  -1],
#        [137, 140,  15,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [240, 141,  -1,  16,  -1,  -1, 413,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [137, 356,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 87,  -1, 359,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [134,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55, 135,  -1,  -1, 361,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [209, 438,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55,  -1, 442,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 43,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 83,  47,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [145,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [137, 153,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 79, 152,  -1, 154,  -1,  -1,  -1,  -1,  -1,  -1],
#        [240,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 83,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [188,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [214, 189, 409,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 87,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 43,  91,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [134,  49,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 40, 266,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [190,  -1, 275,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149, 198,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [237, 181,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [145,  -1, 182,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 76, 311,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [239,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [133, 350,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [214,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [142, 220,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [183,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [204,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149, 386,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [229,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55, 230,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [190,  69, 233,  -1, 395,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [209, 438,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 83, 211, 443,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55,  -1,  -1, 446,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [137, 356,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [241,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [240,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 52, 292,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149,  -1, 301,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 79, 152,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [214,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [145, 221,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [134,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [145,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [237, 181,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [145,  -1, 182,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 43,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 83,  47,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [190,  24,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [204,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])
```

In the example, parts of the word "▁really" were separated into the token ids: [244, 410, 411, 412, 413]
This corresponds to `['▁', '▁r', '▁re', '▁real', '▁really']`

The CTC segmentation algorithm then iterates over these tokens in the ground truth, calculates the transition probabilities for each token from `lpz` and decides for the transition(s) with the token combination that has the highest accumulated transition probability.

</div></details>


* *Sometimes the end of the last utterance is cut short. How do I solve this?* This is a known issue and strongly depends on used ASR model. A possible solution might be to just add a few milliseconds to the end of the last utterance. It's also practical to apply a threshold on the mean absolute (MA) signal, as described by [Bakhturina et al.](https://arxiv.org/abs/2104.04896).


# Reference

The full paper can be found in the preprint https://arxiv.org/abs/2007.09127 or published at <https://doi.org/10.1007/978-3-030-60276-5_27>. The code used in the paper is archived in <https://github.com/cornerfarmer/ctc_segmentation>. To cite this work:

```
@InProceedings{ctcsegmentation,
author="K{\"u}rzinger, Ludwig
and Winkelbauer, Dominik
and Li, Lujun
and Watzel, Tobias
and Rigoll, Gerhard",
editor="Karpov, Alexey
and Potapova, Rodmonga",
title="CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition",
booktitle="Speech and Computer",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="267--278",
abstract="Recent end-to-end Automatic Speech Recognition (ASR) systems demonstrated the ability to outperform conventional hybrid DNN/HMM ASR. Aside from architectural improvements in those systems, those models grew in terms of depth, parameters and model capacity. However, these models also require more training data to achieve comparable performance.",
isbn="978-3-030-60276-5"
}
```
