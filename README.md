# MilliTransNet

## 1. Introduction

This project addresses the reconstruction of human body joints from reflected mmWave radar signals. The objective is to estimate a sequence of human skeletal joints from processed radar measurements, using synchronized Kinect V2 joint annotations as supervision during training.

## 2. Output

The model predicts 17 human body joints. These are derived from the 25-joint Kinect V2 skeleton, while the joints in `JOINTS_IGNORE_IDX = [6, 10, 15, 19, 21, 23, 22, 24]` are excluded.

The 17 retained joints are:

- `0`: spine base
- `1`: spine mid
- `2`: neck
- `3`: head
- `4`: shoulder left
- `5`: elbow left
- `7`: hand left
- `8`: shoulder right
- `9`: elbow right
- `11`: hand right
- `12`: hip left
- `13`: knee left
- `14`: ankle left
- `16`: hip right
- `17`: knee right
- `18`: ankle right
- `20`: spine shoulder

The ignored Kinect V2 joints are:

- `6`: wrist left
- `10`: wrist right
- `15`: foot left
- `19`: foot right
- `21`: hand tip left
- `22`: thumb left
- `23`: hand tip right
- `24`: thumb right

## 3. Input Signal

The input is a parsed and processed radar cube represented as a range-doppler-angle sequence. The sequence length typically corresponds to about 1-2 seconds of motion. Radar signal parsing and preprocessing are handled in a separate radar-processing codebase.

## 4. Training Ground Truth

The training ground truth is the synchronized Kinect V2 joint sequence corresponding to the radar measurements.

## 5. TODO

- To train MilliTransNet effectively, a much larger scale dataset is needed for good convergence.
- Replace SPL (structured prediction layer, which estimates child joints conditioned on estimated parent joints) with an SMPL-based output model.
- Train with dropped or missing frames, or with a sparser set of frames, to support continued inference under opportunistic sensing.
- Explore multi-person joint reconstruction using filtered heatmaps or a smarter model that identifies multiple people.
