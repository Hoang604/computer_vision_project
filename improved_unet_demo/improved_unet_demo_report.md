# Improved U-Net Demo Report

Generated on: 2025-07-15 17:15:54

## Overview

This report summarizes the performance and capabilities of the improved U-Net architecture with multi-level attention and alternative conditioning mechanisms.

## Conditioning Strategies Comparison

| Strategy | Parameters | Inference Time (ms) | Memory (MB) | Output Mean | Output Std |
|----------|------------|---------------------|-------------|-------------|------------|
| Cross Attention | 39.8M | 370.4 | 112.4 | 0.000 | 0.012 |
| Additive | 40.3M | 85.5 | 105.2 | -0.003 | 0.017 |
| Concatenation | 51.2M | 15.7 | 131.6 | -0.007 | 0.034 |
| Mixed | 45.6M | 11.3 | 121.0 | -0.000 | 0.029 |

## Key Findings

