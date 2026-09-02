# Data preparation

The numerical analyses use the packaged predictions and do not require images.
Feature extraction uses the original left-camera frames and instrument-part
masks, stored as NumPy arrays. Obtain the media and annotations through the
[EndoVis 2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Data/)
and [EndoVis 2018](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Data/)
challenge providers under their access terms.

## Retained frames

`data/manifest.jsonl` gives every retained frame, its output path, shape, and dtype.
Frame indices are zero-based. Keep the original numeric index; do not renumber
frames after sorting or filtering.

| Dataset | Split | Source sequences | Frames | RGB shape |
| --- | --- | --- | ---: | --- |
| EndoVis 2017 (`site1`) | train | 1-8 | 1,800 | 1024 x 1280 x 3 |
| EndoVis 2017 (`site1`) | test | 9-10 | 600 | 1080 x 1920 x 3 |
| EndoVis 2018 (`site2`) | train | 1-7, 9-16 | 2,235 | 1024 x 1280 x 3 |
| EndoVis 2018 (`site2`) | test | 1-4 | 997 | 1024 x 1280 x 3 |

For 2017, `instrument_dataset_N/left_frames/frameFFF.png` corresponds to
`site1/<split>/image/videoNframeFFF.npy`. For 2018,
`seq_N/left_frames/frameFFF.png` corresponds to
`site2/<split>/image/videoNframeFFF.npy`. The archive's outer folder may differ
between releases. The 2018 training release has no sequence 8. Its test sequence
1 contains 250 labelled frames and sequences 2-4 contain 249 each.

The 2018 training and test sequence numbers overlap: `train/site2/video1` and
`test/site2/video1` are different sequences. Use the full `source_sequence_id`
in `data/vqa_records.jsonl` for grouped analyses, not `video_id` alone.

## Array encoding

- Images: `uint8`, RGB channel order, values 0-255, original spatial resolution.
- Masks: `uint8`, height x width; 0 = background, 1 = shaft, 2 = wrist,
  3 = clasper/jaws. Image and mask must share the same spatial dimensions.
- Store part IDs, not RGB palette colours, instance IDs, or instrument-type IDs.
- Do not resize arrays to the encoder input size. Each extractor applies its
  own pretrained image transform.

For EndoVis 2018, use the release's `labels.json` to map `instrument-shaft`,
`instrument-wrist`, and `instrument-clasper` to 1, 2, and 3; all other classes
become 0. The feature masks use the original release maps. The official seven-frame
repair set is used in the separate anatomical reference layer, whose derived Q2
answers are already included in `data/frame_annotations.jsonl`. Keep those
answers when reproducing the reported experiment.

For EndoVis 2017, combine the source instrument-part annotations into one part
map using the provider's part definitions. Conversion of these ground-truth
files depends on the downloaded release layout and is a separate preparation
step; `build_manifest.py` reads prepared arrays rather than converting PNGs.

An RGB conversion for one record is:

```python
from pathlib import Path
import numpy as np
from PIL import Image

source = Path("downloads/instrument_dataset_9/left_frames/frame000.png")
target = Path("external_data/site1/test/image/video9frame000.npy")
target.parent.mkdir(parents=True, exist_ok=True)
with Image.open(source) as image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
np.save(target, rgb, allow_pickle=False)
```

Save each aligned part map at the corresponding `mask_path` from the manifest.
Then check the presence, shape, dtype, and mask-value range of all 5,632 pairs:

```bash
python scripts/validate_release.py --source-arrays
```

An existing array collection can be checked without copying it:

```bash
python scripts/validate_release.py --source-arrays /path/to/array/root
```

That directory should contain `site1/` and `site2/`. For extraction, the paths in
the supplied manifest must resolve under the repository's `external_data/`.
