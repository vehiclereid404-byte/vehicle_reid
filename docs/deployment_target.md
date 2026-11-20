# Deployment target: Raspberry Pi 5 + Hailo-8L

Every model we pick must run on a **Raspberry Pi 5 with a Hailo-8L** (13 TOPS NPU,
the Pi AI Kit or AI HAT+ 13 TOPS). Training happens on a GPU machine. Only inference runs on the Pi.

## Toolchain

```
PyTorch (fp32) → ONNX (fixed input size) → Hailo Dataflow Compiler (DFC):
    parse → optimize (int8, calibration images) → compile --hw-arch hailo8l → model.hef
model.hef → Pi 5, run with HailoRT (Python: hailo_platform)
```

- **Where the DFC runs:** x86 Ubuntu only (22.04 is safest), or WSL2 on Windows. On WSL2 the
  GPU-based optimization levels don't work. It doesn't run on the Pi, and there's no native
  Windows build. Download it from the Hailo Developer Zone (free account).
  Use a lab Linux machine with an NVIDIA GPU if there is one.
- **No CPU fallback:** a single unsupported layer means the model doesn't compile.
  `GridSample` is not supported, and the deformable attention in RT-DETR, RF-DETR and D-FINE uses it.
  Run a parse check **before** you spend time training a model.
- **int8 accuracy:** models run quantized to int8. Always report both the fp32 (PyTorch) score
  and the HEF score (DFC emulator or device), because the gap can be large.
- **Fixed, small input:** for example 640×640 for detection and 256×128 for ReID.
  The 4K frames are resized or tiled on the Pi CPU, so measure accuracy at the deployed input size.
- **Several models on one chip:** the Hailo-8L can switch between models (detector + ReID),
  but they share its throughput.

## Pi 5 CPU side
- **Cores:** 4× Cortex-A76, used for decoding, resizing, NMS, tracking and matching.
  Keep that code numpy-light.
- **Video decode:** the Pi 5 hardware decoder handles **HEVC only**. The dataset videos are
  MPEG-4 Part 2 (`FMP4`), 3840×2160 at 15 fps, which the Pi would decode in software (slow).
  For on-device benchmarks, re-encode to HEVC or feed pre-decoded frames, and say which you used.

## Per stage

| Stage | Runs on | Deployable choices | Needs proof first |
|---|---|---|---|
| Detection | Hailo-8L | YOLO family (Hailo Model Zoo has compiled YOLOv5/v8/X and DAMO-YOLO; check YOLOv9 in the current zoo) | RT-DETR and RF-DETR (`GridSample`); Faster R-CNN (two-stage RoI ops, no zoo path found) |
| Tracking | Pi CPU | ByteTrack; BoT-SORT with camera motion compensation **off** (the cameras are static) | appearance trackers: one ReID call per box on the chip |
| ReID | Hailo-8L | small CNNs at 256×128. Hailo's RepVGG-A0 ReID (about 0.9 GMACs, 512/2048-d) is the reference, with a torchreid retraining guide in the Model Zoo | OSNet, ResNet50-IBN (InstanceNorm reported supported), TransReID and CLIP-ReID (ViT, heavy) |
| Matching, OD | Pi CPU or server | numpy/scipy only | – |

A model that doesn't compile can stay in the report as a **GPU accuracy reference**,
labelled as not deployable.

## Every model reports
- fp32 score and HEF int8 score (same scorer from `src/evaluation/`)
- FPS and latency on the Pi 5 + Hailo-8L. Use the DFC emulator until the hardware is
  available, and label which one was used.
- HEF size, and the compile status: parses, compiles, or fails at layer X
- `deployable: yes/no`

## Code layout
- `src/<stage>/<name>_export.py`: PyTorch → ONNX at the deployed input size.
- Inference on the Pi goes through the same interfaces as on the GPU. For example,
  `src/detection/hailo_yolo.py:HailoYOLODetector` implements `Detector` with HailoRT,
  so switching between GPU and Pi is a one-line change in `params.yaml`.
- Calibration images for quantization come from the **train split only** (data-eval provides them).

## Open questions (team)
1. One Pi per camera, or one Pi for all four cameras? (Four 4K streams at 15 fps is 60 frames/s of detection.)
2. Must it run live at 15 fps, or is offline processing on the Pi enough?
3. Is a Pi 5 + Hailo-8L available now for benchmarks?
