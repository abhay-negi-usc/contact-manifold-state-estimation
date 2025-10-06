#!/usr/bin/env python3
"""
Helper script to modify training parameters when resuming from checkpoints
"""
import torch
import argparse
import os

def modify_checkpoint_lr(checkpoint_path, new_lr):
    """Modify the learning rate in a checkpoint file"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get old learning rate
    old_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    
    # Update learning rate in optimizer state
    checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] = new_lr
    
    # Create backup
    backup_path = checkpoint_path + '.backup'
    if not os.path.exists(backup_path):
        torch.save(checkpoint, backup_path)
        print(f"💾 Backup created: {backup_path}")
    
    # Save modified checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✅ Learning rate updated: {old_lr:.2e} → {new_lr:.2e}")
    return True

def print_checkpoint_info(checkpoint_path):
    """Print information about a checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"📊 Checkpoint Information: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Best Loss: {checkpoint.get('best_loss', 'Unknown')}")
    print(f"   Learning Rate: {checkpoint['optimizer_state_dict']['param_groups'][0]['lr']:.2e}")
    print(f"   Batch Size: {checkpoint.get('batch_size', 'Unknown')}")
    print(f"   Geometry: {checkpoint.get('geometry', 'Unknown')}")
    
    # Check for optimizations
    if 'optimizations' in checkpoint:
        opt = checkpoint['optimizations']
        print(f"   Optimizations:")
        print(f"     - KD-tree: {opt.get('kdtree', 'Unknown')}")
        print(f"     - Mixed Precision: {opt.get('mixed_precision', 'Unknown')}")
        print(f"     - Samples/Epoch: {opt.get('samples_per_epoch', 'Unknown')}")

def main():
    parser = argparse.ArgumentParser(description='Modify training checkpoint parameters')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--new-lr', type=float, help='New learning rate')
    parser.add_argument('--info', action='store_true', help='Just print checkpoint info')
    
    args = parser.parse_args()
    
    if args.info:
        print_checkpoint_info(args.checkpoint)
    elif args.new_lr:
        modify_checkpoint_lr(args.checkpoint, args.new_lr)
        print_checkpoint_info(args.checkpoint)
    else:
        print("Please specify --new-lr or --info")
        parser.print_help()

if __name__ == "__main__":
    main()
