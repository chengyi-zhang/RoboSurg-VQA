# Reproduction

The repository supports two workflows: recomputing numerical results from
frozen predictions, and extracting features and fitting the reference models.
Only the latter needs the original EndoVis images and masks.

## Environment

Use Python 3.12; the shared-model experiment used Python 3.12.9.
Run commands from the repository root.

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS,
or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

For numerical analyses and tests, NumPy is the only third-party dependency:

```bash
python -m pip install numpy==2.5.2
python scripts/validate_release.py
python -m unittest discover -s tests
```

For feature extraction and training, install the full dependencies:

```bash
python -m pip install -r requirements.txt
```

BiomedCLIP extraction and shared-head training require a CUDA-enabled PyTorch
build. Weights are downloaded from the official
[BiomedCLIP checkpoint](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
and the torchvision ResNet-18 provider. Downloads remain under `.cache/`.

## Statistical analyses

The following commands use the packaged predictions and annotations. They do
not train models or call an API. Commands without an explicit output option
write under `results/`, so run them in a separate working copy to preserve the
packaged reference outputs and their checksums.

| Analysis | Command | Output directory |
| --- | --- | --- |
| Taskwise metrics and bootstrap comparisons | `python scripts/analyse_taskwise_baselines.py` | `results/taskwise/` |
| Mask targets and question-prior comparisons | `python scripts/analyse_mask_and_question_prior.py` | `results/mask_question_analysis/` |
| Independent annotator agreement | `python scripts/analyse_human_audit.py` | `results/human_audit/` |
| Candidate and taskwise human-reference comparisons | `python scripts/evaluate_human_reference.py` | `results/human_reference/` |
| Shared-model human-reference comparisons | `python scripts/evaluate_shared_vqa.py` | `results/shared_vqa_human_reference/` |
| Task-group sensitivity | `python scripts/analyse_task_groups.py` | `results/task_groups/` |
| Audit coverage and label distributions | `python scripts/analyse_audit_sample.py` | `results/audit_sample/` |

Macro-F1 keeps each task's fixed class list during resampling, including classes
absent from an individual replicate. `tests/test_evaluation.py` checks this
behaviour and recomputes all 12 taskwise human-reference results and intervals.
It also checks the shared-model aggregate scores against the frozen predictions.

## Shared question-conditioned model

First follow [Data preparation](data_preparation.md) and validate the prepared arrays:

```bash
python scripts/validate_release.py --source-arrays
python scripts/extract_biomedclip_features.py
python scripts/train_shared_vqa.py
```

To use only an existing weight cache, add `--offline` to the extraction command.
The model and tokenizer revisions are recorded in [shared_vqa.json](../configs/shared_vqa.json):

| Component | Revision |
| --- | --- |
| BiomedCLIP | `9f341de24bfb00180f1b847274256e9b65a3a32e` |
| BiomedBERT configuration and tokenizer | `d673b8835373c6fa116d6d8006b33d48734e305d` |

Each training record is presented with its canonical question and two training
paraphrases. The held-out paraphrase is reserved for evaluation. Image and text
encoders remain frozen; the shared head is fitted with within-task inverse
answer-frequency weights.

Epoch selection uses four source sequences: EndoVis 2017 video 3 and EndoVis 2018
videos 2, 11 and 12. The selected head is refitted on all 23 training sequences.
Seeds 20260831, 20260832 and 20260833 are used, and the final prediction averages
their probabilities. Optimisation settings are in the same configuration file;
record-level assignments are in [vqa_records.jsonl](../data/vqa_records.jsonl).

## Taskwise baselines

The taskwise RGB and mask-geometry classifiers used scikit-learn 1.7.2, while
the shared-model environment used 1.9.0. For taskwise fitting, create a separate
environment, install `requirements.txt`, then select scikit-learn 1.7.2:

```bash
python -m pip install scikit-learn==1.7.2
python scripts/extract_taskwise_features.py
python scripts/train_taskwise_baselines.py
python scripts/evaluate_cross_source.py
```

The ResNet-18 encoder is frozen. The classifiers use the supplied split, mask
targets and task-applicability records. Device and library differences can
affect fitted predictions; the numerical analyses use the archived predictions
to reproduce the reported statistics.

## Dataset assembly

The supplied records are ready for analysis. These scripts rebuild individual
parts of the dataset when needed:

| Script | Required inputs | Output |
| --- | --- | --- |
| `build_manifest.py` | Prepared EndoVis NumPy arrays | Frame manifest |
| `build_mask_targets.py` | Extracted mask features, manifest and annotations | Instrument-mask targets |
| `build_vqa_records.py` | Manifest, annotations, mask targets and question configuration | VQA records and summary |
| `sample_human_audit.py` | Manifest, candidate answers and mask targets | Audit sample manifest |

Use a working copy when rebuilding these files. Preserve the released questions,
labels and source-sequence assignments when reproducing the paper.

## Candidate-label generation

The original request settings and prompt are in
[candidate_generation.json](../configs/candidate_generation.json) and
[candidate_generation_prompt.txt](../configs/candidate_generation_prompt.txt).
The generated labels are already included in the repository.

For a new annotation run, prepare the source arrays and set `OPENAI_API_KEY` in
your environment. A small run can be started with:

```bash
python scripts/generate_candidate_labels.py --limit 10
```

This optional command incurs API charges. It resumes from
`outputs/candidate_annotations.jsonl`; omit `--limit` to process the full
manifest. The historical `gpt-4o-mini` alias was not snapshot-pinned, so a new
API run need not reproduce the archived labels exactly.
