#!/usr/bin/env python3
"""
Test script to verify the run caching implementation.
Tests the RunCache class and its methods.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the RunCache class
from InstantID import RunCache

def test_basic_caching():
    """Test basic cache operations."""
    print("Testing basic caching...")

    cache = RunCache(max_size=3)

    # Test 1: Cache miss
    key1 = cache.get_key(torch.tensor([1.0, 2.0, 3.0]))
    result1 = cache.get(key1)
    assert result1 is None, "Expected cache miss"
    print("  ✓ Cache miss works")

    # Test 2: Cache put and hit
    cache.put(key1, "value1")
    result2 = cache.get(key1)
    assert result2 == "value1", "Expected cache hit"
    print("  ✓ Cache put and hit work")

    # Test 3: Different tensors have different keys
    key2 = cache.get_key(torch.tensor([1.0, 2.0, 4.0]))
    assert key1 != key2, "Different tensors should have different keys"
    print("  ✓ Different tensors have different keys")

    # Test 4: Same tensors have same keys
    key3 = cache.get_key(torch.tensor([1.0, 2.0, 3.0]))
    assert key1 == key3, "Same tensors should have same keys"
    print("  ✓ Same tensors have same keys")

    print("✓ Basic caching tests passed!\n")

def test_lru_eviction():
    """Test LRU eviction when cache is full."""
    print("Testing LRU eviction...")

    cache = RunCache(max_size=3)

    # Fill cache to capacity
    key1 = cache.get_key(torch.tensor([1.0]))
    key2 = cache.get_key(torch.tensor([2.0]))
    key3 = cache.get_key(torch.tensor([3.0]))

    cache.put(key1, "value1")
    cache.put(key2, "value2")
    cache.put(key3, "value3")

    assert cache.get(key1) == "value1", "Key1 should be in cache"
    assert cache.get(key2) == "value2", "Key2 should be in cache"
    assert cache.get(key3) == "value3", "Key3 should be in cache"
    print("  ✓ Cache filled to capacity")

    # Add one more item, should evict oldest (key1)
    key4 = cache.get_key(torch.tensor([4.0]))
    cache.put(key4, "value4")

    assert cache.get(key1) is None, "Key1 should be evicted"
    assert cache.get(key2) == "value2", "Key2 should still be in cache"
    assert cache.get(key3) == "value3", "Key3 should still be in cache"
    assert cache.get(key4) == "value4", "Key4 should be in cache"
    print("  ✓ LRU eviction works correctly")

    print("✓ LRU eviction tests passed!\n")

def test_cache_stats():
    """Test cache statistics."""
    print("Testing cache statistics...")

    cache = RunCache(max_size=5)

    key1 = cache.get_key(torch.tensor([1.0]))
    key2 = cache.get_key(torch.tensor([2.0]))

    # 2 misses
    cache.get(key1)
    cache.get(key2)

    # Put values
    cache.put(key1, "value1")
    cache.put(key2, "value2")

    # 2 hits
    cache.get(key1)
    cache.get(key2)

    # 1 more miss
    cache.get(cache.get_key(torch.tensor([3.0])))

    stats = cache.get_stats()
    assert stats['hits'] == 2, f"Expected 2 hits, got {stats['hits']}"
    assert stats['misses'] == 3, f"Expected 3 misses, got {stats['misses']}"
    assert stats['size'] == 2, f"Expected size 2, got {stats['size']}"
    print(f"  ✓ Stats: {stats}")

    # Test clear
    cache.clear()
    stats = cache.get_stats()
    assert stats['hits'] == 0, "Hits should be 0 after clear"
    assert stats['misses'] == 0, "Misses should be 0 after clear"
    assert stats['size'] == 0, "Size should be 0 after clear"
    print("  ✓ Cache clear works")

    print("✓ Cache statistics tests passed!\n")

def test_tensor_hashing():
    """Test tensor hashing for different scenarios."""
    print("Testing tensor hashing...")

    cache = RunCache()

    # Test identical tensors
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([1.0, 2.0, 3.0])
    key1 = cache.get_key(t1)
    key2 = cache.get_key(t2)
    assert key1 == key2, "Identical tensors should have same hash"
    print("  ✓ Identical tensors have same hash")

    # Test different tensors
    t3 = torch.tensor([1.0, 2.0, 3.1])
    key3 = cache.get_key(t3)
    assert key1 != key3, "Different tensors should have different hashes"
    print("  ✓ Different tensors have different hashes")

    # Test with kwargs
    key4 = cache.get_key(t1, extract_kps=True)
    key5 = cache.get_key(t1, extract_kps=False)
    assert key4 != key5, "Same tensor with different kwargs should have different hashes"
    print("  ✓ Kwargs affect hash correctly")

    # Test large tensor (should use sampling)
    large_tensor = torch.randn(1000, 1000)
    key_large = cache.get_key(large_tensor)
    assert key_large is not None, "Large tensor should be hashable"
    print("  ✓ Large tensor hashing works")

    print("✓ Tensor hashing tests passed!\n")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Run Caching Implementation Tests")
    print("=" * 60 + "\n")

    try:
        test_basic_caching()
        test_lru_eviction()
        test_cache_stats()
        test_tensor_hashing()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
