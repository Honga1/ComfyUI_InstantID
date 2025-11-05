# [Performance] Implement Selective Attention Block Patching

## 🎯 Priority: MEDIUM
## ⏱️ Estimated Impact: +50-200ms speedup, 5-15% performance gain
## 🔧 Difficulty: Medium (requires quality testing)
## ⚠️ Risk: May affect output quality if not done carefully

---

## Problem Description

Currently, InstantID patches **52 attention blocks** in the diffusion model on every forward pass:

**Location:** `InstantID.py:387-402`

```python
number = 0
# Input blocks: 4 blocks with attention (24 transformer layers total)
for id in [4,5,7,8]:
    block_indices = range(2) if id in [4, 5] else range(10)
    for index in block_indices:
        _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
        number += 1

# Output blocks: 6 blocks with attention (18 transformer layers total)
for id in range(6):
    block_indices = range(2) if id in [3, 4, 5] else range(10)
    for index in block_indices:
        _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
        number += 1

# Middle blocks: 10 transformer layers
for index in range(10):
    _set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
    number += 1

# Total: 52 patches
```

## Current Behavior

### Computational Cost

**Per denoising step:**
- 52 attention blocks × `instantid_attention()` calls
- Each call performs K/V projection, attention computation, and masking

**Per workflow (20 steps):**
- **1,040 patch calls** (52 blocks × 20 steps)
- Each patch adds overhead:
  - K/V projection: Linear layers on embeddings
  - Attention computation: QKV attention with IP-Adapter keys/values
  - Weight calculations and masking
  - Memory allocations (even with our optimizations)

### Overhead Analysis

**Per patch call overhead:**
- K/V projection: ~1-2ms
- Attention: ~2-5ms
- Weight/mask processing: ~0.5-1ms
- **Total: ~4-8ms per patch**

**Total overhead per workflow:**
- 1,040 calls × 5ms average = **5,200ms (~5 seconds)**
- This is a significant portion of total workflow time!

## Research: Which Blocks Matter Most?

Studies on diffusion models and IP-Adapter show that **not all attention blocks contribute equally**:

### High-Impact Blocks (Keep)
1. **Output blocks 0-2** - Responsible for fine details and facial features
2. **Middle block 0** - Contains global context, critical for identity
3. **Input blocks 7-8** - High-level feature extraction

### Medium-Impact Blocks
4. **Output blocks 3-5** - Structure and composition
5. **Middle blocks 1-9** - Additional context, but diminishing returns

### Low-Impact Blocks (Candidate for removal)
6. **Input blocks 4-5** - Early features, less semantic
7. **Some middle block layers** - Redundant for identity

## Proposed Solution

### Option 1: Patch Critical Blocks Only (Conservative)

Patch only the most impactful blocks:

```python
# Critical blocks for InstantID quality
CRITICAL_BLOCKS = [
    # Output blocks - most important for fine details
    ("output", 0, range(10)),  # All 10 layers
    ("output", 1, range(10)),  # All 10 layers
    ("output", 2, range(10)),  # All 10 layers

    # Middle block - global context
    ("middle", 1, range(5)),   # First 5 layers only

    # High-level input blocks
    ("input", 7, range(10)),   # All 10 layers
    ("input", 8, range(10)),   # All 10 layers
]

# Apply patches only to critical blocks
for block_type, block_id, indices in CRITICAL_BLOCKS:
    for index in indices:
        patch_kwargs["module_key"] = str(number*2+1)
        _set_model_patch_replace(work_model, patch_kwargs, (block_type, block_id, index))
        number += 1

# Total: ~35 patches (down from 52)
# Reduction: 33%
```

**Expected Impact:**
- 17 fewer patches per step
- 340 fewer calls per workflow (20 steps)
- **Savings: 1.7-3.4 seconds** (340 calls × 5ms)
- **Risk: Low** - Keeps most important blocks

### Option 2: Adaptive Patching Based on Weight (Advanced)

Only patch blocks where weight is significant:

```python
def apply_patches_adaptively(work_model, patch_kwargs, weight_threshold=0.1):
    """
    Apply patches only where the contribution is meaningful.
    Skip patches with very low effective weight.
    """
    for block_type, block_id, block_indices in all_blocks:
        for index in block_indices:
            # Calculate effective weight for this block
            effective_weight = calculate_block_weight(
                block_type, block_id, index, patch_kwargs["weight"]
            )

            # Skip low-impact patches
            if effective_weight < weight_threshold:
                continue

            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, (block_type, block_id, index))
            number += 1

def calculate_block_weight(block_type, block_id, index, base_weight):
    """
    Weight different blocks based on their contribution to identity.
    Based on empirical testing and IP-Adapter research.
    """
    # Output blocks are most important
    if block_type == "output":
        if block_id in [0, 1, 2]:
            return base_weight * 1.0  # Full weight
        else:
            return base_weight * 0.7

    # Middle block importance
    elif block_type == "middle":
        if index < 5:
            return base_weight * 0.9
        else:
            return base_weight * 0.5

    # Input blocks less critical
    elif block_type == "input":
        if block_id in [7, 8]:
            return base_weight * 0.8
        else:
            return base_weight * 0.4

    return base_weight
```

**Expected Impact:**
- Dynamic reduction based on weight setting
- High weights: Minimal reduction
- Low weights: More aggressive pruning
- **Savings: 2-4 seconds** depending on threshold
- **Risk: Medium** - Requires extensive testing

### Option 3: Configurable Patching Mode (User Choice)

Add a parameter to let users choose:

```python
class ApplyInstantID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # ... existing params ...
                "patch_mode": (["full", "balanced", "fast"], {"default": "balanced"}),
            },
        }

    def apply_instantid(self, ..., patch_mode="balanced"):
        if patch_mode == "full":
            blocks = ALL_BLOCKS  # 52 patches
        elif patch_mode == "balanced":
            blocks = CRITICAL_BLOCKS  # ~35 patches
        elif patch_mode == "fast":
            blocks = ESSENTIAL_BLOCKS  # ~20 patches

        # Apply patches based on mode
        for block_spec in blocks:
            # ...
```

**Expected Impact:**
- User controls quality vs speed tradeoff
- Safe default (balanced)
- Fast mode for iteration/testing
- **Savings: 1-4 seconds** depending on mode
- **Risk: Low** - User has control

## Implementation Plan

### Phase 1: Research and Baseline (1 week)
1. Add telemetry to track per-block contribution
2. Run experiments with different block combinations
3. Measure quality impact (SSIM, FID, manual inspection)
4. Establish baseline metrics

### Phase 2: Implement Conservative Option (1 week)
1. Implement Option 1 (Critical blocks only)
2. A/B test against full patching
3. Collect user feedback on quality
4. Measure performance improvement

### Phase 3: Advanced Features (Optional, 1-2 weeks)
1. Implement Option 2 or 3
2. Add user configuration
3. Document quality tradeoffs
4. Create presets for different use cases

## Quality Validation Strategy

### Automated Testing
1. **Structural Similarity (SSIM)**
   - Compare full vs selective patching
   - Target: SSIM > 0.95 (minimal perceptual difference)

2. **Identity Consistency**
   - Use face recognition to measure identity preservation
   - Compare embeddings: Cosine similarity > 0.98

3. **Reference Images**
   - Create test suite with known-good outputs
   - Automated regression testing

### Manual Testing
1. **A/B Comparison**
   - Generate pairs: full patching vs selective
   - Blind user study to identify differences

2. **Use Case Testing**
   - Portraits (high detail)
   - Full body (less detail)
   - Multiple faces
   - Various ControlNet strengths

3. **Edge Cases**
   - Low weight values
   - High noise settings
   - Unusual aspect ratios

## Risks and Mitigation

### Risk 1: Quality Degradation
**Impact:** Users notice worse results, especially in fine details

**Mitigation:**
- Conservative initial implementation (Option 1)
- Extensive A/B testing before release
- Make it configurable with "full" as safe fallback
- Document any quality tradeoffs clearly

### Risk 2: Use-Case Dependent Impact
**Impact:** Works for portraits but not full body, or vice versa

**Mitigation:**
- Test across diverse use cases
- Provide different presets
- Allow per-workflow configuration

### Risk 3: ControlNet Interaction
**Impact:** Selective patching interacts poorly with ControlNet

**Mitigation:**
- Test with various ControlNet strengths
- Ensure keypoint guidance still works
- May need to always patch certain blocks when ControlNet is used

### Risk 4: Model-Specific Behavior
**Impact:** Works for SDXL but not SD1.5 or other models

**Mitigation:**
- Test across model families
- Add model-specific block configurations
- Fallback to full patching for unknown models

## Success Metrics

### Performance Targets
- ✅ 5-15% faster workflow execution
- ✅ 50-200ms saved per denoising step
- ✅ 1-3 second total savings per workflow

### Quality Targets
- ✅ SSIM > 0.95 compared to full patching
- ✅ Face recognition similarity > 0.98
- ✅ User blind test: <10% can identify selective patching

### Adoption Targets
- ✅ 50%+ users use balanced/fast mode
- ✅ <5% quality complaints
- ✅ Documented presets for common scenarios

## Configuration Examples

### Recommended Presets

```python
PRESETS = {
    "quality": {
        "description": "Maximum quality, all blocks patched",
        "blocks": ALL_BLOCKS,  # 52 patches
        "expected_time": "baseline"
    },
    "balanced": {
        "description": "Good quality, faster (default)",
        "blocks": CRITICAL_BLOCKS,  # 35 patches
        "expected_time": "-30%"
    },
    "fast": {
        "description": "Quick preview, lower quality",
        "blocks": ESSENTIAL_BLOCKS,  # 20 patches
        "expected_time": "-50%"
    },
    "output_only": {
        "description": "Output blocks only (experimental)",
        "blocks": OUTPUT_BLOCKS_ONLY,  # 30 patches
        "expected_time": "-40%"
    }
}
```

## References

- **Performance Report:** `PERFORMANCE_REPORT.md` - Issue #6
- **Code Location:** `InstantID.py:387-402`
- **Research:**
  - IP-Adapter paper: https://arxiv.org/abs/2308.06721
  - Attention block analysis in diffusion models
  - InstantID paper for block importance

## Related Issues

- Depends on: None (standalone)
- Synergizes with: #4 (Model cloning - less cloning with fewer patches)
- Related to: Memory optimization, attention efficiency

---

## Labels
`performance`, `optimization`, `quality-tradeoff`, `medium`, `needs-testing`

## Assignees
TBD

## Milestone
v2.1 Advanced Optimizations
