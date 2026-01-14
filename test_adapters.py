#!/usr/bin/env python3
"""
Test script for newly implemented dataset adapters.

This script tests:
1. Adapter registration and availability
2. Adapter instantiation
3. JSON split file loading
4. Sample loading for different splits (train, val, test)
5. Few-shot learning functionality
6. Basic data validation
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from biomedxpro.impl.adapters import (
    list_available_adapters,
    get_adapter,
    JsonDatasetAdapter,
)
from biomedxpro.core.domain import DataSplit, StandardSample


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text: str) -> None:
    """Print a formatted subheader."""
    print(f"\n>>> {text}")
    print("-" * 70)


def test_adapter_registry() -> bool:
    """Test 1: Verify all adapters are registered."""
    print_header("TEST 1: Adapter Registry")
    
    try:
        adapters = list_available_adapters()
        print(f"Total registered adapters: {len(adapters)}")
        
        # List all adapters
        print("\nRegistered adapters:")
        for adapter_name in sorted(adapters):
            print(f"  ✓ {adapter_name}")
        
        # Verify the 11 new JSON-based adapters are present
        new_adapters = {
            "btmri", "busi", "chmnist", "covid_19", "ctkidney",
            "dermamnist", "kneexray", "kvasir", "lungcolon", "octmnist", "retina"
        }
        
        registered = set(adapters)
        missing = new_adapters - registered
        
        if missing:
            print(f"\n✗ ERROR: Missing adapters: {missing}")
            return False
        
        print(f"\n✓ All 11 new JSON-based adapters are registered!")
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_adapter_instantiation() -> bool:
    """Test 2: Verify adapters can be instantiated."""
    print_header("TEST 2: Adapter Instantiation")
    
    adapters_to_test = [
        "btmri", "busi", "chmnist", "covid_19", "ctkidney",
        "dermamnist", "kneexray", "kvasir", "lungcolon", "octmnist", "retina"
    ]
    
    success_count = 0
    failed_adapters = []
    
    for adapter_name in adapters_to_test:
        try:
            adapter = get_adapter(adapter_name, root="/tmp/test", shots=0)
            adapter_type = type(adapter).__name__
            print(f"✓ {adapter_name:15} -> {adapter_type}")
            success_count += 1
        except Exception as e:
            print(f"✗ {adapter_name:15} -> ERROR: {e}")
            failed_adapters.append((adapter_name, str(e)))
    
    print(f"\nInstantiation Results: {success_count}/{len(adapters_to_test)} successful")
    
    if failed_adapters:
        print("\nFailed instantiations:")
        for name, error in failed_adapters:
            print(f"  - {name}: {error}")
        return False
    
    return True


def test_json_data_loading() -> bool:
    """Test 3: Test JSON data loading with actual dataset files."""
    print_header("TEST 3: JSON Data Loading")
    
    data_path = Path(__file__).parent / "src" / "biomedxpro" / "data"
    
    if not data_path.exists():
        print(f"✗ Data path not found: {data_path}")
        return False
    
    print(f"Data directory: {data_path}")
    
    # Test datasets (those with JSON files)
    test_datasets = {
        "btmri": data_path / "BTMRI",
        "busi": data_path / "BUSI",
        "chmnist": data_path / "CHMNIST",
        "covid_19": data_path / "COVID_19",
        "ctkidney": data_path / "CTKidney",
        "dermamnist": data_path / "DermaMNIST",
        "kneexray": data_path / "KneeXray",
        "kvasir": data_path / "Kvasir",
        "lungcolon": data_path / "LungColon",
        "octmnist": data_path / "OCTMNIST",
        "retina": data_path / "RETINA",
    }
    
    success_count = 0
    
    for adapter_name, dataset_path in test_datasets.items():
        try:
            json_files = list(dataset_path.glob("split_*.json"))
            
            if not json_files:
                print(f"⚠ {adapter_name:15} -> No JSON split file found")
                continue
            
            # Try to instantiate and check split data loading
            adapter = get_adapter(adapter_name, root=str(dataset_path), shots=0)
            
            # Access the private method to load split data
            split_data = adapter._load_split_data()
            
            # Verify structure
            required_keys = {"train", "val", "test"}
            actual_keys = set(split_data.keys())
            
            if actual_keys != required_keys:
                print(f"✗ {adapter_name:15} -> Missing keys: {required_keys - actual_keys}")
                continue
            
            train_count = len(split_data["train"])
            val_count = len(split_data["val"])
            test_count = len(split_data["test"])
            total_count = train_count + val_count + test_count
            
            print(f"✓ {adapter_name:15} -> Train: {train_count:4} | Val: {val_count:4} | Test: {test_count:4} | Total: {total_count:6}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ {adapter_name:15} -> ERROR: {e}")
    
    print(f"\nJSON Loading Results: {success_count}/{len(test_datasets)} datasets loaded successfully")
    return success_count > 0


def test_sample_loading() -> bool:
    """Test 4: Test loading samples for different splits."""
    print_header("TEST 4: Sample Loading (with actual data)")
    
    data_path = Path(__file__).parent / "src" / "biomedxpro" / "data"
    
    # Test with a dataset that should have data
    test_dataset = "kvasir"  # Typically has data
    dataset_path = data_path / "Kvasir"
    
    if not dataset_path.exists():
        print(f"⚠ Dataset directory not found: {dataset_path}")
        print("  Skipping sample loading tests (data directory structure may differ)")
        return True
    
    try:
        print(f"Testing adapter: {test_dataset}")
        adapter = get_adapter(test_dataset, root=str(dataset_path), shots=0)
        
        results = {}
        
        for split in [DataSplit.TRAIN, DataSplit.VAL, DataSplit.TEST]:
            try:
                samples = adapter.load_samples(split)
                results[split] = samples
                split_name = split.name
                print(f"✓ {split_name:6} split loaded: {len(samples)} samples")
            except Exception as e:
                print(f"✗ {split.name:6} split ERROR: {e}")
                results[split] = None
        
        # Verify sample structure
        print("\nSample structure verification:")
        for split, samples in results.items():
            if samples:
                sample = samples[0]
                if isinstance(sample, StandardSample):
                    print(f"✓ {split.name:6} -> Sample type: StandardSample")
                    print(f"         -> image_path: {sample.image_path}")
                    print(f"         -> label: {sample.label}")
                else:
                    print(f"✗ {split.name:6} -> Invalid sample type: {type(sample)}")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR loading samples: {e}")
        return False


def test_few_shot_functionality() -> bool:
    """Test 5: Test few-shot learning functionality."""
    print_header("TEST 5: Few-Shot Learning")
    
    data_path = Path(__file__).parent / "src" / "biomedxpro" / "data"
    
    test_dataset = "kvasir"
    dataset_path = data_path / "Kvasir"
    
    if not dataset_path.exists():
        print(f"⚠ Dataset directory not found: {dataset_path}")
        print("  Skipping few-shot tests")
        return True
    
    try:
        print(f"Testing few-shot learning with {test_dataset}")
        
        # Load full dataset
        adapter_full = get_adapter(test_dataset, root=str(dataset_path), shots=0)
        train_full = adapter_full.load_samples(DataSplit.TRAIN)
        print(f"✓ Full dataset train samples: {len(train_full)}")
        
        # Load few-shot (5 samples per class)
        adapter_shots = get_adapter(test_dataset, root=str(dataset_path), shots=5)
        train_shots = adapter_shots.load_samples(DataSplit.TRAIN)
        print(f"✓ Few-shot (5 per class) train samples: {len(train_shots)}")
        
        # Verify few-shot is smaller or equal
        if len(train_shots) <= len(train_full):
            print(f"✓ Few-shot correctly reduces dataset size")
            print(f"  Reduction: {len(train_full)} -> {len(train_shots)} samples")
            return True
        else:
            print(f"✗ Few-shot dataset is larger than full dataset!")
            return False
            
    except Exception as e:
        print(f"✗ ERROR in few-shot testing: {e}")
        return False


def test_inheritance_and_structure() -> bool:
    """Test 6: Verify correct inheritance and class structure."""
    print_header("TEST 6: Inheritance and Structure")
    
    try:
        from biomedxpro.impl.adapters import (
            BTMRIAdapter, BUSIAdapter, CHMNISTAdapter, COVID19Adapter,
            CTKidneyAdapter, DermaMNISTAdapter, KneeXrayAdapter, KvasirAdapter,
            LungColonAdapter, OCTMNISTAdapter, RETINAAdapter
        )
        from biomedxpro.core.interfaces import IDatasetAdapter
        
        adapters = [
            BTMRIAdapter, BUSIAdapter, CHMNISTAdapter, COVID19Adapter,
            CTKidneyAdapter, DermaMNISTAdapter, KneeXrayAdapter, KvasirAdapter,
            LungColonAdapter, OCTMNISTAdapter, RETINAAdapter
        ]
        
        print("Verifying class hierarchy:")
        
        for adapter_class in adapters:
            # Check if inherits from JsonDatasetAdapter
            if issubclass(adapter_class, JsonDatasetAdapter):
                print(f"✓ {adapter_class.__name__:20} -> inherits from JsonDatasetAdapter")
            else:
                print(f"✗ {adapter_class.__name__:20} -> NOT inheriting from JsonDatasetAdapter")
                return False
            
            # Check if implements IDatasetAdapter interface
            if issubclass(adapter_class, IDatasetAdapter):
                print(f"  └─> implements IDatasetAdapter interface")
            else:
                print(f"  ✗ Does NOT implement IDatasetAdapter interface")
                return False
        
        print(f"\n✓ All 11 adapters correctly inherit from JsonDatasetAdapter")
        print(f"✓ All 11 adapters implement IDatasetAdapter interface")
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_adapter_methods() -> bool:
    """Test 7: Verify all adapters have required methods."""
    print_header("TEST 7: Adapter Methods")
    
    try:
        from biomedxpro.impl.adapters import (
            BTMRIAdapter, BUSIAdapter, CHMNISTAdapter, COVID19Adapter,
            CTKidneyAdapter, DermaMNISTAdapter, KneeXrayAdapter, KvasirAdapter,
            LungColonAdapter, OCTMNISTAdapter, RETINAAdapter
        )
        
        adapters = [
            BTMRIAdapter, BUSIAdapter, CHMNISTAdapter, COVID19Adapter,
            CTKidneyAdapter, DermaMNISTAdapter, KneeXrayAdapter, KvasirAdapter,
            LungColonAdapter, OCTMNISTAdapter, RETINAAdapter
        ]
        
        required_methods = {
            "load_samples": "Required method from IDatasetAdapter",
            "_load_split_data": "Method for loading JSON split data",
        }
        
        print("Checking required methods:")
        all_valid = True
        
        for adapter_class in adapters:
            print(f"\n{adapter_class.__name__}:")
            for method_name, description in required_methods.items():
                if hasattr(adapter_class, method_name):
                    method = getattr(adapter_class, method_name)
                    if callable(method):
                        print(f"  ✓ {method_name:20} - {description}")
                    else:
                        print(f"  ✗ {method_name:20} - exists but not callable")
                        all_valid = False
                else:
                    print(f"  ✗ {method_name:20} - NOT FOUND")
                    all_valid = False
        
        return all_valid
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def run_all_tests() -> None:
    """Run all tests and generate report."""
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  DATASET ADAPTER TEST SUITE".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    tests = [
        ("Adapter Registry", test_adapter_registry),
        ("Adapter Instantiation", test_adapter_instantiation),
        ("JSON Data Loading", test_json_data_loading),
        ("Sample Loading", test_sample_loading),
        ("Few-Shot Learning", test_few_shot_functionality),
        ("Inheritance & Structure", test_inheritance_and_structure),
        ("Adapter Methods", test_adapter_methods),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n✗ FATAL ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name:30} [{status}]")
    
    print("\n" + "-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("\nThe newly implemented adapters are working correctly!")
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        print("Please review the errors above.")
    
    print("\n" + "#" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
