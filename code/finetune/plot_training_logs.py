#!/usr/bin/env python3
"""
训练日志分析和可视化脚本
用于分析训练过程中的loss、learning rate等指标变化
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def plot_training_metrics(csv_file, output_dir=None):
    """
    绘制训练指标图表
    """
    # 读取日志文件
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 成功读取日志文件: {csv_file}")
        print(f"📈 训练记录数: {len(df)}")
    except Exception as e:
        print(f"❌ 读取日志文件失败: {e}")
        return
    
    # 设置中文字体和样式
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Dashboard', fontsize=16, fontweight='bold')
    
    # 1. 损失函数变化
    ax1 = axes[0, 0]
    if 'train_loss' in df.columns:
        train_mask = df['train_loss'] != 'N/A'
        if train_mask.any():
            ax1.plot(df[train_mask]['step'], pd.to_numeric(df[train_mask]['train_loss']), 
                    'b-', label='Train Loss', linewidth=2)
    
    if 'eval_loss' in df.columns:
        eval_mask = df['eval_loss'] != 'N/A'
        if eval_mask.any():
            ax1.plot(df[eval_mask]['step'], pd.to_numeric(df[eval_mask]['eval_loss']), 
                    'r-', label='Eval Loss', linewidth=2)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Training Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 学习率变化
    ax2 = axes[0, 1]
    if 'learning_rate' in df.columns:
        lr_mask = df['learning_rate'] != 'N/A'
        if lr_mask.any():
            ax2.plot(df[lr_mask]['step'], pd.to_numeric(df[lr_mask]['learning_rate']), 
                    'g-', label='Learning Rate', linewidth=2)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 3. Epoch vs Loss
    ax3 = axes[1, 0]
    if 'epoch' in df.columns and 'train_loss' in df.columns:
        train_mask = df['train_loss'] != 'N/A'
        if train_mask.any():
            ax3.plot(df[train_mask]['epoch'], pd.to_numeric(df[train_mask]['train_loss']), 
                    'b-', label='Train Loss', linewidth=2)
    
    if 'epoch' in df.columns and 'eval_loss' in df.columns:
        eval_mask = df['eval_loss'] != 'N/A'
        if eval_mask.any():
            ax3.plot(df[eval_mask]['epoch'], pd.to_numeric(df[eval_mask]['eval_loss']), 
                    'r-', label='Eval Loss', linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss vs Epoch')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 训练时间统计
    ax4 = axes[1, 1]
    if 'total_time' in df.columns:
        time_mask = df['total_time'] != 'N/A'
        if time_mask.any():
            total_times = pd.to_numeric(df[time_mask]['total_time'])
            ax4.plot(df[time_mask]['step'], total_times / 3600, 'purple', linewidth=2)  # 转换为小时
    
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Training Time (Hours)')
    ax4.set_title('Training Time Progress')
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📈 训练指标图表已保存: {plot_file}")
    
    # 显示统计信息
    print("\n📊 训练统计摘要:")
    print("-" * 50)
    
    if 'train_loss' in df.columns:
        train_losses = pd.to_numeric(df[df['train_loss'] != 'N/A']['train_loss'])
        if len(train_losses) > 0:
            print(f"🔥 训练损失: 初始={train_losses.iloc[0]:.4f}, 最终={train_losses.iloc[-1]:.4f}")
            print(f"📉 损失下降: {((train_losses.iloc[0] - train_losses.iloc[-1]) / train_losses.iloc[0] * 100):.2f}%")
    
    if 'eval_loss' in df.columns:
        eval_losses = pd.to_numeric(df[df['eval_loss'] != 'N/A']['eval_loss'])
        if len(eval_losses) > 0:
            print(f"✅ 验证损失: 最佳={eval_losses.min():.4f}")
    
    if 'total_time' in df.columns:
        total_times = pd.to_numeric(df[df['total_time'] != 'N/A']['total_time'])
        if len(total_times) > 0:
            print(f"⏰ 训练时间: {total_times.iloc[-1] / 3600:.2f} 小时")
    
    if 'epoch' in df.columns:
        epochs = pd.to_numeric(df[df['epoch'] != 'N/A']['epoch'])
        if len(epochs) > 0:
            print(f"📚 完成轮数: {epochs.iloc[-1]:.2f} epochs")
    
    return df

def compare_training_runs(csv_files, labels=None, output_dir="./comparison"):
    """
    对比多个训练运行的结果
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(csv_files))]
    
    plt.figure(figsize=(15, 5))
    
    # 对比训练损失
    plt.subplot(1, 3, 1)
    for csv_file, label in zip(csv_files, labels):
        try:
            df = pd.read_csv(csv_file)
            train_mask = df['train_loss'] != 'N/A'
            if train_mask.any():
                plt.plot(df[train_mask]['step'], pd.to_numeric(df[train_mask]['train_loss']), 
                        label=f'{label} - Train', linewidth=2)
        except Exception as e:
            print(f"⚠️ 无法读取 {csv_file}: {e}")
    
    plt.xlabel('Training Steps')
    plt.ylabel('Train Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 对比验证损失
    plt.subplot(1, 3, 2)
    for csv_file, label in zip(csv_files, labels):
        try:
            df = pd.read_csv(csv_file)
            eval_mask = df['eval_loss'] != 'N/A'
            if eval_mask.any():
                plt.plot(df[eval_mask]['step'], pd.to_numeric(df[eval_mask]['eval_loss']), 
                        label=f'{label} - Eval', linewidth=2)
        except Exception as e:
            print(f"⚠️ 无法读取 {csv_file}: {e}")
    
    plt.xlabel('Training Steps')
    plt.ylabel('Eval Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 对比学习率
    plt.subplot(1, 3, 3)
    for csv_file, label in zip(csv_files, labels):
        try:
            df = pd.read_csv(csv_file)
            lr_mask = df['learning_rate'] != 'N/A'
            if lr_mask.any():
                plt.plot(df[lr_mask]['step'], pd.to_numeric(df[lr_mask]['learning_rate']), 
                        label=f'{label} - LR', linewidth=2)
        except Exception as e:
            print(f"⚠️ 无法读取 {csv_file}: {e}")
    
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # 保存对比图
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, 'training_comparison.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"📊 训练对比图已保存: {comparison_file}")

def main():
    parser = argparse.ArgumentParser(description="训练日志分析和可视化")
    parser.add_argument("--log_file", type=str, required=True, help="训练日志CSV文件路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--compare", nargs="+", help="对比多个日志文件")
    parser.add_argument("--labels", nargs="+", help="对比图的标签")
    
    args = parser.parse_args()
    
    if args.compare:
        print("📊 开始对比分析多个训练运行...")
        compare_training_runs(args.compare, args.labels, args.output_dir or "./comparison")
    else:
        print("📈 开始分析单个训练运行...")
        plot_training_metrics(args.log_file, args.output_dir)

if __name__ == "__main__":
    main() 