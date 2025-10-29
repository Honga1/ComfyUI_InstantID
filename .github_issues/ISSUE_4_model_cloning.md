# [Performance] Eliminate Model Cloning Overhead in ApplyInstantID

## 🎯 Priority: HIGH
## ⏱️ Estimated Impact: +100-500ms speedup, +2-8GB VRAM savings
## 🔧 Difficulty: Hard (requires deep ComfyUI integration)

---

## Problem Description

Every time `ApplyInstantID` or `InstantIDAttentionPatch` is executed, the entire diffusion model is cloned:

```python
# InstantID.py:378 and InstantID.py:546
work_model = model.clone()
```

This operation:
- Creates a **deep copy** of the entire U-Net model (2-8GB depending on model)
- Copies all model parameters, state dictionaries, and options
- Takes **100-500ms** per execution
- Increases VRAM pressure and can trigger swapping
- Happens on **every workflow run**, even when parameters don't change

## Current Behavior

**Location:** `InstantID.py:378` (ApplyInstantID) and `InstantID.py:546` (InstantIDAttentionPatch)

**Why it happens:**
ComfyUI's architecture requires model cloning to apply patches without mutating the original model. This ensures nodes don't have side effects on shared models.

**Cost per workflow:**
- Time: 100-500ms
- VRAM: 2-8GB duplicate allocation
- Memory bus: Full model copy transfer

For workflows with multiple InstantID nodes or iterations, this compounds significantly.

## Proposed Solutions

### Option 1: Patch Caching with Hash Keys (Recommended)

Cache patched models based on a hash of patch parameters:

```python
# Global cache for patched models
_patched_model_cache = {}

def apply_instantid(self, ...):
    # Create hash from all parameters that affect patches
    patch_hash = hash((
        id(model),           # Original model identity
        ip_weight,
        cn_strength,
        start_at,
        end_at,
        str(image_prompt_embeds),  # Embedding fingerprint
        # ... other relevant params
    ))

    # Check cache first
    if patch_hash in _patched_model_cache:
        work_model = _patched_model_cache[patch_hash]
    else:
        work_model = model.clone()
        # Apply patches...
        _patched_model_cache[patch_hash] = work_model

    return work_model
```

**Pros:**
- Eliminates cloning for identical parameter sets
- Works within ComfyUI's model architecture
- No changes to ComfyUI core needed

**Cons:**
- Cache can grow large with many parameter variations
- Need careful cache invalidation strategy
- Hash computation has small overhead

**Expected Impact:**
- First run: No change
- Repeat runs with same params: **100-500ms faster, 2-8GB less VRAM**
- Cache hit rate: 60-80% for typical workflows

### Option 2: Lightweight Patch Application (Advanced)

Apply patches as dynamic hooks during forward pass instead of cloning:

```python
class InstantIDPatch:
    def __init__(self, model, patch_kwargs):
        self.model = model
        self.patch_kwargs = patch_kwargs
        self.hooks = []

    def __enter__(self):
        # Register forward hooks on attention blocks
        for block in self.model.blocks:
            hook = block.register_forward_hook(self._patch_attention)
            self.hooks.append(hook)
        return self.model

    def __exit__(self, *args):
        # Remove hooks
        for hook in self.hooks:
            hook.remove()

    def _patch_attention(self, module, input, output):
        # Apply InstantID modifications
        return modified_output

# Usage
with InstantIDPatch(model, patch_kwargs) as patched_model:
    # Use patched model
    output = patched_model(...)
# Model automatically unpacked after context exit
```

**Pros:**
- **Zero cloning overhead**
- **Zero VRAM duplication**
- Patches applied/removed dynamically

**Cons:**
- Requires significant refactoring
- May need ComfyUI core changes
- More complex error handling
- Potential compatibility issues

**Expected Impact:**
- **Eliminates 100-500ms overhead completely**
- **Saves 2-8GB VRAM**
- **10-15% total workflow speedup**

### Option 3: Model Clone-on-Write (Hybrid)

Implement shallow cloning with copy-on-write semantics:

```python
class LazyClonedModel:
    def __init__(self, original_model):
        self.original = original_model
        self.modifications = {}

    def __getattr__(self, name):
        if name in self.modifications:
            return self.modifications[name]
        return getattr(self.original, name)

    def patch_block(self, block_id, patch):
        # Only clone specific block being patched
        if block_id not in self.modifications:
            self.modifications[block_id] = self.original.blocks[block_id].clone()
        self.modifications[block_id].apply_patch(patch)
```

**Pros:**
- Reduces cloning to only modified components
- Better than full clone, not as complex as hooks
- Gradual migration path

**Cons:**
- Still some cloning overhead
- Requires model structure awareness
- Partial VRAM savings only

## Implementation Plan

### Phase 1: Proof of Concept (1-2 weeks)
1. Implement Option 1 (Patch Caching)
2. Test with various workflows
3. Measure cache hit rates and speedup
4. Identify edge cases

### Phase 2: Optimization (1 week)
1. Optimize hash function for speed
2. Implement cache eviction (LRU or size-based)
3. Add telemetry for cache effectiveness
4. Handle cache invalidation correctly

### Phase 3: Advanced (Optional, 2-3 weeks)
1. Prototype Option 2 (Hooks approach)
2. Evaluate compatibility with ComfyUI
3. Consider upstreaming to ComfyUI core
4. Comprehensive testing

## Risks and Considerations

### Cache Invalidation
- **Risk:** Stale cached models used with wrong parameters
- **Mitigation:** Comprehensive hash including all relevant state

### Memory Growth
- **Risk:** Cache grows unbounded
- **Mitigation:** Implement LRU eviction or max cache size (e.g., 5 models)

### Model Mutation
- **Risk:** Cached model accidentally mutated
- **Mitigation:** Deep freeze cached models or copy on retrieval

### Compatibility
- **Risk:** Breaks with ComfyUI updates
- **Mitigation:** Version checks, fallback to cloning

## Testing Strategy

### Correctness Tests
1. Verify identical output with and without caching
2. Test with varying parameters to ensure proper invalidation
3. Multi-node workflows with shared models
4. Edge case: Model changes between runs

### Performance Tests
1. Measure first run vs cached run time
2. Monitor VRAM usage with `nvidia-smi`
3. Test cache hit rates across typical workflows
4. Stress test with many parameter variations

### Regression Tests
1. Ensure no quality degradation in outputs
2. Visual comparison of generated images
3. Numerical comparison of embeddings

## Success Metrics

### Required
- ✅ No output quality degradation
- ✅ Cache hit rate > 50% for typical workflows
- ✅ No memory leaks over extended use

### Target
- 🎯 100-300ms speedup on cached runs
- 🎯 50% reduction in peak VRAM usage
- 🎯 10-15% total workflow speedup

### Stretch
- 🚀 Zero cloning overhead (Option 2 implemented)
- 🚀 Compatible with all ComfyUI model types
- 🚀 Upstreamed to ComfyUI core

## References

- **Performance Report:** `PERFORMANCE_REPORT.md` - Issue #4
- **Code Location:** `InstantID.py:378`, `InstantID.py:546`
- **Related:** ComfyUI model cloning mechanism in `comfy/model_patcher.py`

## Related Issues

- Depends on: None (standalone optimization)
- Blocks: None
- Related to: GPU memory optimization, VRAM swapping prevention

---

## Labels
`performance`, `optimization`, `memory`, `hard`, `high-priority`

## Assignees
TBD

## Milestone
v2.0 Performance Improvements
