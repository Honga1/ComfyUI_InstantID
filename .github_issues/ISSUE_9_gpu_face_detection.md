# [Performance] Enable GPU-Accelerated Face Detection with InsightFace

## 🎯 Priority: HIGH
## ⏱️ Estimated Impact: 50-100% faster face detection (doubles speed)
## 🔧 Difficulty: Hard (requires ONNX Runtime GPU setup)
## ⚠️ Platform: CUDA, ROCm, or alternative solutions

---

## Problem Description

Face detection with InsightFace is currently **CPU-bound**, creating a major bottleneck:

**Location:** `InstantID.py:238` (InstantIDFaceAnalysis.load_insight_face)

```python
def load_insight_face(self, provider):
    model = FaceAnalysis(
        name="antelopev2",
        root=INSIGHTFACE_DIR,
        providers=[provider + 'ExecutionProvider',]
    )
    model.prepare(ctx_id=0, det_size=(640, 640))
```

### Current Execution Flow

```
┌─────────────┐
│ Image (GPU) │
└──────┬──────┘
       │ tensor_to_image() - GPU→CPU transfer
       ▼
┌─────────────┐
│ NumPy (CPU) │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│ InsightFace on CPU   │  ◄─── BOTTLENECK (100-300ms per image)
│ - Face detection     │
│ - Face recognition   │
│ - Keypoint extraction│
└──────┬───────────────┘
       │
       ▼
┌──────────────┐
│ Embeddings   │
│ (CPU→GPU)    │
└──────────────┘
```

### Performance Impact

**Per image:**
- GPU→CPU transfer: 5-10ms
- Face detection on CPU: **100-300ms** ⚠️
- CPU→GPU transfer: 5-10ms
- **Total: 110-320ms per image**

**For batch of 6 images (typical workflow):**
- Sequential processing: 6 × 200ms = **1,200ms (1.2 seconds)**
- GPU sits idle during this entire time!

**Workflow impact:**
- Face detection often takes **30-50% of total workflow time**
- GPU utilization drops to 0% during face detection
- Creates stuttering in progress bars (CPU-bound phase)

## Root Cause Analysis

### Why CPU-Only?

InsightFace uses ONNX Runtime with these execution providers:

1. **CPUExecutionProvider** (default) ✅ Always available
2. **CUDAExecutionProvider** ❌ Requires:
   - ONNX Runtime GPU build
   - CUDA Toolkit
   - cuDNN libraries
3. **ROCMExecutionProvider** ❌ Requires:
   - ONNX Runtime ROCm build
   - ROCm stack

**Current code:** Line 238 accepts provider parameter but uses string concatenation:
```python
providers=[provider + 'ExecutionProvider',]
```

**Problem:** Even if user selects "CUDA", it may fall back to CPU silently if ONNX Runtime GPU isn't installed.

### GPU→CPU→GPU Roundtrip Cost

```python
# utils.py:16-19
def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()  # Force CPU
    image = image[..., [2, 1, 0]].numpy()               # Convert to NumPy
    return image
```

This forces CPU transfer before InsightFace even runs. If InsightFace could accept GPU tensors, we'd eliminate both transfers.

## Proposed Solutions

### Option 1: Enable ONNX Runtime GPU Support (Recommended)

**Goal:** Use existing InsightFace with GPU acceleration

**Requirements:**
```bash
# Install ONNX Runtime GPU
pip install onnxruntime-gpu  # For NVIDIA CUDA
# OR
pip install onnxruntime-rocm  # For AMD ROCm
```

**Implementation:**

```python
def load_insight_face(self, provider):
    global _faceanalysis_cache

    cache_key = f"antelopev2_{provider}"
    if cache_key in _faceanalysis_cache:
        return (_faceanalysis_cache[cache_key],)

    # Verify GPU provider is available
    available_providers = onnxruntime.get_available_providers()
    requested_provider = provider + 'ExecutionProvider'

    if requested_provider not in available_providers:
        print(f"\033[33mWARNING: {requested_provider} not available!")
        print(f"Available providers: {available_providers}")
        print(f"Falling back to CPUExecutionProvider\033[0m")
        providers = ['CPUExecutionProvider']
    else:
        providers = [requested_provider]
        print(f"\033[32mINFO: Using {requested_provider} for face detection\033[0m")

    model = FaceAnalysis(
        name="antelopev2",
        root=INSIGHTFACE_DIR,
        providers=providers
    )
    model.prepare(ctx_id=0, det_size=(640, 640))

    _faceanalysis_cache[cache_key] = model
    return (model,)
```

**Benefits:**
- ✅ Drop-in improvement (no major code changes)
- ✅ 50-100% faster face detection
- ✅ Eliminates GPU idle time
- ✅ Better GPU utilization

**Challenges:**
- ❌ Requires users to install onnxruntime-gpu
- ❌ CUDA/cuDNN dependencies
- ❌ May have version compatibility issues

**Expected Impact:**
- Face detection: 100-300ms → **50-150ms** (2× faster)
- Workflow speedup: **10-20%** (if face detection is 30% of time)

### Option 2: PyTorch-Native Face Detection (Long-term)

**Goal:** Replace InsightFace with pure PyTorch implementation

**Options:**
1. **FaceNet PyTorch** - https://github.com/timesler/facenet-pytorch
2. **RetinaFace PyTorch** - https://github.com/biubug6/Pytorch_Retinaface
3. **YOLO Face** - YOLOv8-face for detection
4. **Custom implementation** - Port InsightFace models to PyTorch

**Example with FaceNet:**

```python
from facenet_pytorch import MTCNN, InceptionResnetV1

class PyTorchFaceAnalysis:
    def __init__(self, device='cuda'):
        self.device = device
        # Face detection
        self.detector = MTCNN(
            device=self.device,
            post_process=False
        )
        # Face recognition
        self.recognizer = InceptionResnetV1(
            pretrained='vggface2'
        ).eval().to(self.device)

    def get(self, image_tensor):
        """
        Process image tensor directly on GPU.
        No CPU transfer needed!
        """
        with torch.no_grad():
            # Detect faces
            boxes, probs, landmarks = self.detector.detect(
                image_tensor, landmarks=True
            )

            if boxes is None:
                return []

            # Get embeddings
            faces = self.detector.extract(image_tensor, boxes)
            embeddings = self.recognizer(faces)

            # Format results compatible with InsightFace
            results = []
            for box, landmark, embedding in zip(boxes, landmarks, embeddings):
                results.append({
                    'bbox': box,
                    'kps': landmark,
                    'embedding': embedding.cpu().numpy(),
                    'det_score': probs[0]
                })

            return results
```

**Benefits:**
- ✅ Native GPU execution (no ONNX Runtime)
- ✅ No CPU transfers needed
- ✅ Better integration with PyTorch ecosystem
- ✅ Potentially faster (optimized CUDA kernels)
- ✅ Easier debugging and customization

**Challenges:**
- ❌ Major refactoring required
- ❌ Need to match InsightFace quality
- ❌ May require different models (compatibility)
- ❌ 2-3 weeks implementation time

**Expected Impact:**
- Face detection: 100-300ms → **30-80ms** (3-4× faster)
- Eliminates ALL CPU transfers
- Workflow speedup: **15-25%**

### Option 3: TensorRT Optimization (Advanced)

**Goal:** Convert ONNX models to TensorRT for maximum speed

**Implementation:**

```python
import tensorrt as trt

def convert_to_tensorrt(onnx_model_path):
    """
    Convert InsightFace ONNX models to TensorRT for maximum performance.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_model_path, 'rb') as model:
        parser.parse(model.read())

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # Optimize for FP16 if supported
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)
    return engine

# Use TensorRT engine for inference
```

**Benefits:**
- ✅ Fastest possible execution on NVIDIA GPUs
- ✅ 2-3× faster than ONNX Runtime GPU
- ✅ Lower latency

**Challenges:**
- ❌ NVIDIA-only (no AMD support)
- ❌ Complex setup and conversion
- ❌ Model conversion can fail
- ❌ Maintenance burden

**Expected Impact:**
- Face detection: 100-300ms → **20-50ms** (4-6× faster)
- Only works on NVIDIA GPUs

## Implementation Plan

### Phase 1: Quick Win - Enable GPU Provider (1 week)
1. Update `load_insight_face` with provider verification
2. Add installation instructions for onnxruntime-gpu
3. Test with CUDA and ROCm
4. Document setup requirements
5. Measure performance improvement

**Deliverables:**
- ✅ GPU-enabled face detection
- ✅ Installation guide
- ✅ Fallback to CPU if GPU unavailable

### Phase 2: Optimize Data Flow (1 week)
1. Investigate keeping tensors on GPU longer
2. Minimize CPU transfers in tensor_to_image
3. Batch processing improvements
4. Parallel face detection for multiple images

**Deliverables:**
- ✅ Reduced transfer overhead
- ✅ Better GPU utilization

### Phase 3: Long-term Solution (Optional, 3-4 weeks)
1. Research PyTorch face detection alternatives
2. Prototype replacement
3. Quality comparison with InsightFace
4. Performance benchmarking
5. Migration guide

**Deliverables:**
- ✅ Pure PyTorch face detection
- ✅ Elimination of ONNX dependency
- ✅ Maximum performance

## Installation & Setup Guide

### For CUDA Users (NVIDIA GPUs)

```bash
# Uninstall CPU-only version
pip uninstall onnxruntime

# Install GPU version
pip install onnxruntime-gpu

# Verify CUDA support
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include: 'CUDAExecutionProvider'
```

**Requirements:**
- CUDA Toolkit 11.x or 12.x
- cuDNN 8.x
- Compatible GPU (Compute Capability 3.5+)

### For ROCm Users (AMD GPUs)

```bash
# Install ROCm version
pip install onnxruntime-rocm

# Verify ROCm support
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include: 'ROCMExecutionProvider'
```

**Requirements:**
- ROCm 5.x or 6.x
- Compatible AMD GPU

### Compatibility Matrix

| Platform | Provider | Speed | Setup Difficulty |
|----------|----------|-------|------------------|
| CPU (any) | CPUExecutionProvider | Baseline (1×) | ✅ Easy (default) |
| NVIDIA GPU | CUDAExecutionProvider | 2-3× faster | ⚠️ Medium (CUDA setup) |
| AMD GPU | ROCMExecutionProvider | 2-3× faster | ⚠️ Medium (ROCm setup) |
| NVIDIA GPU | TensorRT | 4-6× faster | ❌ Hard (expert) |

## Testing Strategy

### Performance Benchmarks

```python
import time

def benchmark_face_detection(image_batch, provider):
    """
    Benchmark face detection performance.
    """
    model = load_insight_face(provider)

    # Warmup
    for _ in range(3):
        extractFeatures(model, image_batch)

    # Measure
    times = []
    for _ in range(10):
        start = time.perf_counter()
        results = extractFeatures(model, image_batch)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    print(f"{provider}: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
    return avg_time

# Run benchmarks
for provider in ['CPU', 'CUDA', 'ROCM']:
    benchmark_face_detection(test_images, provider)
```

### Quality Validation

1. **Embedding Consistency**
   - Compare embeddings from CPU vs GPU
   - Cosine similarity should be > 0.999
   - Small numerical differences acceptable

2. **Detection Consistency**
   - Same faces detected in same order
   - Bounding boxes within ±2 pixels
   - Keypoints within ±1 pixel

3. **Visual Inspection**
   - Generate outputs with CPU and GPU
   - Verify identical quality
   - No visual artifacts

## Risks and Mitigation

### Risk 1: ONNX Runtime GPU Installation Issues
**Impact:** Users can't install onnxruntime-gpu due to CUDA mismatches

**Mitigation:**
- Provide detailed installation guide
- Automatic fallback to CPU
- Check provider availability at runtime
- Document known issues and workarounds

### Risk 2: Numerical Differences
**Impact:** GPU results slightly different from CPU (floating point)

**Mitigation:**
- Use FP32 precision (not FP16) for face detection
- Verify differences are within tolerance
- Document expected behavior

### Risk 3: Platform Fragmentation
**Impact:** Works on NVIDIA but not AMD, or vice versa

**Mitigation:**
- Test on multiple platforms
- Maintain CPU fallback
- Clear documentation of supported platforms

### Risk 4: Memory Usage
**Impact:** GPU face detection uses more VRAM

**Mitigation:**
- Profile VRAM usage
- Offload models when not in use
- Batch size limits for low-VRAM GPUs

## Success Metrics

### Performance Targets
- ✅ 2× faster face detection on GPU
- ✅ <5% CPU utilization during face detection
- ✅ GPU utilization > 80% throughout workflow

### Quality Targets
- ✅ Embedding similarity > 0.999 vs CPU
- ✅ Detection accuracy maintained
- ✅ Zero regression in visual quality

### Adoption Targets
- ✅ 80%+ GPU users enable GPU detection
- ✅ Installation success rate > 90%
- ✅ <10% support issues related to setup

## References

- **Performance Report:** `PERFORMANCE_REPORT.md` - Issue #9
- **Code Location:** `InstantID.py:238`, `utils.py:16-19`
- **InsightFace:** https://github.com/deepinsight/insightface
- **ONNX Runtime:** https://onnxruntime.ai/docs/execution-providers/
- **FaceNet PyTorch:** https://github.com/timesler/facenet-pytorch

## Related Issues

- Depends on: None (standalone)
- Blocks: None
- Related to: Memory optimization, GPU utilization, workflow speed

---

## Labels
`performance`, `gpu`, `optimization`, `hard`, `high-priority`, `infrastructure`

## Assignees
TBD

## Milestone
v2.0 Performance Improvements
