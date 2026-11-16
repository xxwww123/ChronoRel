# ChronoRel

Model components for **Bronze Ornament Inpainting via Observation Guided Diffusion and Constrained Expert Adaptation**.

This release contains the MEA, DSEA, and MADC modules and their feature interfaces. The MTADiffusion backbone, training pipeline, evaluation scripts, datasets, and checkpoints are not included.

## Components

| Module | Implementation |
| --- | --- |
| Mask-aware observation encoding | `chronorel/encoding.py` |
| MEA and rotation consistency | `chronorel/mea.py` |
| Candidate enumeration and joint decision | `chronorel/routing.py` |
| DSEA and sparse expert dispatch | `chronorel/dsea.py` |
| Conditional amplitude and spatial-detail paths | `chronorel/madc.py` |
| Orthonormal Haar detail cues | `chronorel/wavelets.py` |
| Component-level integration | `chronorel/model.py` |
| Local response targets and fitting loss | `chronorel/objectives.py` |

## Installation

```bash
pip install -e .
```

The paper uses Python 3.10 and PyTorch 2.1.2. The modules use native PyTorch operators without custom CUDA extensions.

## Feature interfaces

Feature tensors use `B × C × H × W`. Time embeddings use `B × T`. Missing masks use `B × 1 × H × W`, with one for missing pixels. Observation support is the valid receptive-field fraction. Boundary distance is positive inside the hole and negative outside, normalized by the original image diagonal before resizing.

`ObservationEncoder` returns a dictionary indexed by image-space stride. Each value contains an observation feature and its support map. Its widths, stage strides, and output projections are explicit constructor arguments. MEA and MADC use independent encoder instances.

MEA operates at the 32×32 and 16×16 Brush attention grids for a 512×512 input. Its `base_attention` argument wraps the existing Brush self-attention and returns an NCHW feature. The wrapper retains the backbone's attention weights and positional conventions. MEA samples four locations at each scale in `{1, 2, 4}`. Both paths share offsets and use independent projections and normalization. `rotation_consistency_loss` takes a quarter-turn count from `{1, 2, 3}`; the caller samples it uniformly during training.

DSEA receives the pre-injection U-Net feature, multiscale Brush features, the original Zero Conv output, observation support, normalized boundary distance, and the time embedding. It returns the updated feature, correction, response coefficients, routing decision, and load-balancing loss. The update is `H + ZeroConv(F) + gamma * correction`.

MADC receives the decoder feature, visible-image condition, support, mask, normalized boundary distance, and one Haar detail level. The first Haar level supplies the higher-resolution decoder adapter. The second supplies the lower-resolution adapter. `HaarDetailPyramid` is computed from the visible RGB image before compositing.

`ChronoRelAdapters` collects two MEA/DSEA interfaces and two MADC levels. `prepare_conditions` encodes the visible image once. `aggregate`, `adapt`, and `decode` are called at their corresponding backbone interfaces. Dictionary keys are image-space strides. The lower stride identifies the higher-resolution MADC level.

## Joint decision

The default bank contains four experts and eleven candidates: the empty candidate, four singletons, and six pairs. Candidates are ordered by cardinality and then lexicographically. Expert indices in code are zero-based.

The default perturbation budget is `epsilon=1e-4`; the strength regularizer is `eta=0.1`. The closed-form strength is

```text
a_C = min(1, relu(g_C) / (h_C + eta), sqrt(epsilon / d_C))
```

The perturbation cap is inactive when `d_C=0`. The empty candidate has zero strength and score. Candidates are ranked by `g_C*a_C - 0.5*(h_C+eta)*a_C**2`. Equal scores favor fewer experts, then the fixed candidate order.

`ranking="gain"` uses the same strengths and predicted feasible set but ranks unregularized gain. `ranking="random"` samples uniformly from that feasible set, including the empty candidate. Setting `epsilon=float("inf")` removes the perturbation cap while retaining `eta`. Setting `fixed_strength=0.50` fixes nonempty strengths and retains feasibility filtering and regularized ranking.

Each expert has independent bottleneck weights, timestep modulation, and output gates. Only selected experts with nonzero strength process a token. Shared corrections apply to all tokens. Load balancing uses the soft candidate probabilities and within-candidate expert weights before hard selection.

## Decoder details

The amplitude path uses 16×16 windows, stride 8, a Hann window, and eight pixels of reflection padding. Analysis and synthesis use the same window. Weighted overlap-add divides by the accumulated window products. Conjugate-frequency symmetrization preserves real-valued inverse transforms. Each level has a positive, learnable amplitude bound initialized to `0.10`.

Custom Hann windows require `1 <= stride <= window_size - 2` and `padding >= stride` to keep nonzero coverage at every output position. The padded decoder grid must fit at least one full window on each axis.

Haar filters are orthonormal. Detail bands are ordered as LH, HL, and HH within each image channel. A cue is valid only when its entire image-space support is visible. Resampling normalizes by interpolated validity. The spatial path also processes generated decoder features in unsupported regions.

`response_targets` takes raw RGB predictions in `[0, 1]`. Gain is 100 times the missing-region MSE reduction; perturbation is the unscaled observed-region MSE relative to the shared-only prediction. These tensors are computed before compositing and detached. The full diffusion trajectory and matched-intervention scheduler belong to the training pipeline.

## Configuration

Channel widths, attention dimensions, and per-layer projection sizes must be matched to the connected backbone. Parameter counts depend on these settings. Numerical stabilizers and near-zero residual initializations are specified in the source. The initial deformable sampling stencil places four points on the coordinate axes at each scale.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover tensor shapes, closed-form optimality, feasibility, sparse dispatch, gradient flow at zero support, Haar validity, overlap-add identity and coverage, and observed-pixel compositing. They do not run a restoration benchmark.
