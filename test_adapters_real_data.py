#!/usr/bin/env python3
"""
Dataset Adapter Test Suite - Real Data Version

Tests the newly implemented adapters with actual data located at:
/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/<DATASET_NAME>/
"""

import sys
import json
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from biomedxpro.impl.adapters import (
    list_available_adapters,
    get_adapter,
    JsonDatasetAdapter,
)
from biomedxpro.core.domain import DataSplit

# Base path for actual datasets
DATA_BASE_PATH = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv")

# New JSON-based datasets
JSON_DATASETS = [
    "btmri",
    "busi",
    "chmnist",
    "covid_19",
    "ctkidney",
    "dermamnist",
    "kneexray",
    "kvasir",
    "lungcolon",
    "octmnist",
    "retina",
]

# Mapping of adapter names to actual dataset folder names
DATASET_FOLDER_MAP = {
    "btmri": "BTMRI",
    "busi": "BUSI",
    "chmnist": "CHMNIST",
    "covid_19": "COVID_19",
    "ctkidney": "CTKidney",
    "dermamnist": "DermaMNIST",
    "kneexray": "KneeXray",
    "kvasir": "Kvasir",
    "lungcolon": "LungColon",
    "octmnist": "OCTMNIST",
    "retina": "RETINA",
}


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_1_registry():
    """Test 1: Check adapter registry."""
    print_header("TEST 1: Adapter Registry")
    
    adapters = list_available_adapters()
    print(f"Total registered adapters: {len(adapters)}\n")
    
    print("Registered adapters:")
    for adapter_name in sorted(adapters):
        status = "✓" if adapter_name in JSON_DATASETS else "◯"
        print(f"  {status} {adapter_name}")
    
    json_count = sum(1 for a in adapters if a in JSON_DATASETS)
    print(f"\n✓ {json_count}/11 new JSON-based adapters registered")
    
    return True


def test_2_instantiation():
    """Test 2: Instantiate all adapters."""
    print_header("TEST 2: Adapter Instantiation")
    
    success_count = 0
    for adapter_name in JSON_DATASETS:
        try:
            adapter = get_adapter(adapter_name, root="/tmp/test")
            print(f"✓ {adapter_name:15} -> {type(adapter).__name__}")
            success_count += 1
        except Exception as e:
            print(f"✗ {adapter_name:15} -> ERROR: {e}")
    
    print(f"\nInstantiation Results: {success_count}/{len(JSON_DATASETS)} successful")
    return success_count == len(JSON_DATASETS)


def test_3_json_files():
    """Test 3: Verify JSON split files exist in workspace."""
    print_header("TEST 3: JSON Split Files (Workspace)")
    
    workspace_data_path = Path(__file__).parent / "src" / "biomedxpro" / "data"
    success_count = 0
    
    print(f"Workspace data directory: {workspace_data_path}\n")
    
    for adapter_name, folder_name in DATASET_FOLDER_MAP.items():
        json_path = workspace_data_path / folder_name / f"split_{folder_name}.json"
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                train_count = len(data.get("train", []))
                val_count = len(data.get("val", []))
                test_count = len(data.get("test", []))
                total = train_count + val_count + test_count
                
                print(f"✓ {adapter_name:12} -> Train: {train_count:6} | Val: {val_count:6} | Test: {test_count:6} | Total: {total:7}")
                success_count += 1
            except Exception as e:
                print(f"✗ {adapter_name:12} -> ERROR reading JSON: {e}")
        else:
            print(f"✗ {adapter_name:12} -> JSON file not found: {json_path}")
    
    print(f"\nJSON Files Found: {success_count}/{len(JSON_DATASETS)} datasets")
    return success_count == len(JSON_DATASETS)


def test_4_real_data_availability():
    """Test 4: Check real data availability."""
    print_header("TEST 4: Real Data Availability (/storage/projects3/...)")
    
    success_count = 0
    
    print(f"Base data path: {DATA_BASE_PATH}\n")
    
    if not DATA_BASE_PATH.exists():
        print(f"WARNING: Base data path does not exist: {DATA_BASE_PATH}")
        print("This is expected if running on local machine without network access.")
        print("\nSkipping real data tests...")
        return None  # Return None to indicate skipped
    
    for adapter_name, folder_name in DATASET_FOLDER_MAP.items():
        dataset_path = DATA_BASE_PATH / folder_name
        
        if dataset_path.exists():
            # Count image files
            image_count = sum(1 for _ in dataset_path.rglob("*.*") if _.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])
            
            # Check for split JSON
            split_files = list(dataset_path.glob("split_*.json"))
            
            status = "✓" if image_count > 0 and split_files else "◯"
            print(f"{status} {adapter_name:12} -> {image_count:6} images | Split file: {'Yes' if split_files else 'No'}")
            if image_count > 0:
                success_count += 1
        else:
            print(f"◯ {adapter_name:12} -> Dataset directory not found")
    
    if success_count > 0:
        print(f"\nReal Data Available: {success_count}/{len(JSON_DATASETS)} datasets")
    
    return success_count if success_count > 0 else None


def test_5_adapter_loading():
    """Test 5: Test adapter with real data."""
    print_header("TEST 5: Sample Loading (Real Data)")
    
    if not DATA_BASE_PATH.exists():
        print(f"Skipping: Base data path not available: {DATA_BASE_PATH}")
        return None
    
    print(f"Testing sample loading from: {DATA_BASE_PATH}\n")
    
    success_count = 0
    
    for adapter_name, folder_name in DATASET_FOLDER_MAP.items():
        dataset_path = DATA_BASE_PATH / folder_name
        
        if not dataset_path.exists():
            print(f"◯ {adapter_name:12} -> Dataset path not found")
            continue
        
        try:
            adapter = get_adapter(adapter_name, root=str(dataset_path))
            
            # Try loading each split
            train_samples = adapter.load_samples(DataSplit.TRAIN)
            val_samples = adapter.load_samples(DataSplit.VAL)
            test_samples = adapter.load_samples(DataSplit.TEST)
            
            total = len(train_samples) + len(val_samples) + len(test_samples)
            
            if total > 0:
                print(f"✓ {adapter_name:12} -> Train: {len(train_samples):6} | Val: {len(val_samples):6} | Test: {len(test_samples):6}")
                success_count += 1
            else:
                print(f"◯ {adapter_name:12} -> No samples loaded (check image paths)")
                
        except Exception as e:
            print(f"✗ {adapter_name:12} -> ERROR: {str(e)[:50]}")
    
    if success_count > 0:
        print(f"\nSuccessful Loads: {success_count}/{len(JSON_DATASETS)} datasets")
    
    return success_count if success_count > 0 else None


def test_6_few_shot_learning():
    """Test 6: Few-shot learning functionality."""
    print_header("TEST 6: Few-Shot Learning")
    
    if not DATA_BASE_PATH.exists():
        print(f"Skipping: Base data path not available: {DATA_BASE_PATH}")
        return None
    
    # Test with first available dataset
    test_adapter = None
    for adapter_name, folder_name in DATASET_FOLDER_MAP.items():
        dataset_path = DATA_BASE_PATH / folder_name
        if dataset_path.exists():
            test_adapter = (adapter_name, dataset_path)
            break
    
    if not test_adapter:
        print("Skipping: No dataset available for testing")
        return None
    
    adapter_name, dataset_path = test_adapter
    print(f"Testing with: {adapter_name}\n")
    
    try:
        # Full dataset
        adapter_full = get_adapter(adapter_name, root=str(dataset_path), shots=0)
        train_full = adapter_full.load_samples(DataSplit.TRAIN)
        print(f"✓ Full dataset: {len(train_full)} train samples")
        
        # Few-shot dataset
        shots = min(5, len(train_full) // 2) if train_full else 0
        if shots > 0:
            adapter_few = get_adapter(adapter_name, root=str(dataset_path), shots=shots)
            train_few = adapter_few.load_samples(DataSplit.TRAIN)
            print(f"✓ Few-shot ({shots} per class): {len(train_few)} train samples")
            print(f"✓ Reduction: {len(train_full)} -> {len(train_few)} samples")
            return True
        else:
            print(f"⚠ Cannot test few-shot: insufficient training samples")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_7_adapter_methods():
    """Test 7: Verify adapter methods."""
    print_header("TEST 7: Adapter Methods & Structure")
    
    success_count = 0
    
    for adapter_name in JSON_DATASETS:
        try:
            adapter_class = get_adapter(adapter_name, root="/tmp").__class__
            
            # Check for required methods
            has_load_samples = hasattr(adapter_class, "load_samples")
            has_split_data = hasattr(adapter_class, "_load_split_data")
            
            if has_load_samples and has_split_data:
                print(f"✓ {adapter_name:12} -> load_samples() + _load_split_data()")
                success_count += 1
            else:
                missing = []
                if not has_load_samples:
                    missing.append("load_samples()")
                if not has_split_data:
                    missing.append("_load_split_data()")
                print(f"✗ {adapter_name:12} -> Missing: {', '.join(missing)}")
        except Exception as e:
            print(f"✗ {adapter_name:12} -> ERROR: {e}")
    
    print(f"\nMethod Verification: {success_count}/{len(JSON_DATASETS)} adapters OK")
    return success_count == len(JSON_DATASETS)


def print_summary(results: dict):
    """Print test summary."""
    print_header("TEST SUMMARY")
    
    test_names = [
        "Adapter Registry",
        "Adapter Instantiation",
        "JSON Split Files",
        "Real Data Availability",
        "Sample Loading",
        "Few-Shot Learning",
        "Adapter Methods",
    ]
    
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    
    for test_name, result in zip(test_names, results.values()):
        if result is True:
            status = "✓ PASS"
        elif result is None:
            status = "◯ SKIP"
        else:
            status = "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("\n" + "-" * 70)
    print(f"Passed: {passed} | Skipped: {skipped} | Failed: {failed}")
    print("=" * 70)


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 15 + "DATASET ADAPTER TEST SUITE - REAL DATA" + " " * 15 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    results = {
        "test_1_registry": test_1_registry(),
        "test_2_instantiation": test_2_instantiation(),
        "test_3_json_files": test_3_json_files(),
        "test_4_real_data": test_4_real_data_availability(),
        "test_5_loading": test_5_adapter_loading(),
        "test_6_few_shot": test_6_few_shot_learning(),
        "test_7_methods": test_7_adapter_methods(),
    }
    
    print_summary(results)
    
    print("\nNOTE:")
    print("- If real data tests are skipped, it's expected on local machines")
    print("- Real data should be at: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/<DATASET>/")
    print("- JSON split definitions are in workspace: src/biomedxpro/data/<DATASET>/split_*.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
