#!/usr/bin/env python3
"""
Simple test script to verify the improved U-Net works correctly.
"""

import sys
import os
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.setup import setup_device, create_unet_model


def test_original_unet():
    """Test original U-Net"""
    print("Testing Original U-Net...")
    
    try:
        device = setup_device('cpu')
        
        # Create original model
        model = create_unet_model(
            use_improved=False,
            use_attention=True,
            base_dim=32,  # Smaller for testing
            dim_mults=(1, 2, 4),  # Fewer levels
            cond_dim=32,  # Match the conditioning dim
            rrdb_num_blocks=8,  # Match the blocks count
        )
        model = model.to(device)
        
        # Create dummy inputs
        batch_size = 1
        img_size = 64  # Smaller for testing
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        time_steps = torch.randint(0, 1000, (batch_size,), device=device)
        
        # Create dummy conditioning
        cond_dim = 32
        lr_size = img_size // 4
        condition = []
        num_blocks = 8
        # Create all RRDB features (8 blocks + 1 final = 9 total)
        for i in range(num_blocks + 1):
            feat = torch.randn(batch_size, cond_dim, lr_size, lr_size, device=device)
            condition.append(feat)
        
        print(f"  Number of conditioning features: {len(condition)}")
        print(f"  Each feature shape: {condition[0].shape}")
        print(f"  Features for U-Net (indices 2,5,8): {[i for i in range(len(condition)) if i % 3 == 2]}")
        print(f"  Expected total channels: {len([i for i in range(len(condition)) if i % 3 == 2]) * cond_dim}")
        
        # Forward pass
        with torch.no_grad():
            output = model(x, time_steps, condition)
        
        print(f"✓ Original U-Net works! Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Original U-Net failed: {e}")
        return False


def test_improved_unet():
    """Test improved U-Net"""
    print("Testing Improved U-Net...")
    
    try:
        device = setup_device('cpu')
        
        # Test different conditioning strategies
        strategies = ['cross_attention', 'additive', 'concatenation', 'mixed']
        
        for strategy in strategies:
            print(f"  Testing {strategy}...")
            
            # Create improved model
            model = create_unet_model(
                use_improved=True,
                conditioning_strategy=strategy,
                attention_levels=[1, 2],
                attention_heads=4,
                base_dim=32,  # Smaller for testing
                dim_mults=(1, 2, 4),  # Fewer levels
                cond_dim=32,  # Match the conditioning dim
                rrdb_num_blocks=8,  # Match the blocks count
            )
            model = model.to(device)
            
            # Create dummy inputs
            batch_size = 1
            img_size = 64  # Smaller for testing
            x = torch.randn(batch_size, 3, img_size, img_size, device=device)
            time_steps = torch.randint(0, 1000, (batch_size,), device=device)
            
            # Create dummy conditioning
            cond_dim = 32
            lr_size = img_size // 4
            condition = []
            num_blocks = 8
            # Create all RRDB features (8 blocks + 1 final = 9 total)
            for i in range(num_blocks + 1):
                feat = torch.randn(batch_size, cond_dim, lr_size, lr_size, device=device)
                condition.append(feat)
            
            print(f"    Number of conditioning features: {len(condition)}")
            print(f"    Each feature shape: {condition[0].shape}")
            print(f"    Features for U-Net (indices 2,5,8): {[i for i in range(len(condition)) if i % 3 == 2]}")
            print(f"    Expected total channels: {len([i for i in range(len(condition)) if i % 3 == 2]) * cond_dim}")
            
            # Forward pass
            with torch.no_grad():
                output = model(x, time_steps, condition)
            
            print(f"    ✓ {strategy} works! Output shape: {output.shape}")
            
            del model
            torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        return True
        
    except Exception as e:
        print(f"  ✗ Improved U-Net failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING U-NET IMPLEMENTATIONS")
    print("=" * 60)
    
    # Test original
    original_success = test_original_unet()
    
    print()
    
    # Test improved
    improved_success = test_improved_unet()
    
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Original U-Net: {'✓ PASS' if original_success else '✗ FAIL'}")
    print(f"Improved U-Net: {'✓ PASS' if improved_success else '✗ FAIL'}")
    
    if original_success and improved_success:
        print("\n🎉 All tests passed! The improved U-Net is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
