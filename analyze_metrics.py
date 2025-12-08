import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import yaml # Import yaml library

def analyze_metrics(file_path, output_dir):
    """
    Analyzes metrics.csv containing image enhancement results.
    Handles two cases:
    1. With Ground Truth (GT): Analysis based on PSNR, SSIM, BRISQUE, and Runtime.
    2. Without Ground Truth (No GT): Analysis based on BRISQUE and Runtime.
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading metrics from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Identify columns
    # Expected: filename, method, runtime_sec, psnr, ssim, brisque
    required_cols = {'method', 'runtime_sec'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV must contain at least {required_cols}")
        return

    # Split data into GT and No-GT
    # We assume that if PSNR or SSIM is present (not NaN), it has GT.
    # Otherwise, it's No-GT.
    
    if 'psnr' in df.columns and 'ssim' in df.columns:
        # Check for NaN in psnr to identify No-GT rows
        # Some CSVs might have empty strings or specific markers, but standard pandas read handles empty as NaN
        gt_mask = df['psnr'].notna() & df['ssim'].notna()
        df_gt = df[gt_mask].copy()
        df_no_gt = df[~gt_mask].copy()
    else:
        # If psnr/ssim columns don't exist at all, assume all are No-GT
        df_gt = pd.DataFrame()
        df_no_gt = df.copy()

    # --- Analysis for Data WITH Ground Truth ---
    if not df_gt.empty:
        print("\n" + "="*50)
        print("ANALYSIS: Data WITH Ground Truth (GT)")
        print("="*50)
        print(f"Number of samples: {len(df_gt)}")
        
        # Aggregation
        # Added 'brisque' to gt_metrics
        gt_metrics = ['psnr', 'ssim', 'brisque', 'runtime_sec']
        # Filter metrics that actually exist in columns
        gt_metrics = [m for m in gt_metrics if m in df_gt.columns]
        
        # Calculate stats
        save_formatted_summary(df_gt, gt_metrics, output_dir, 'summary_with_gt.csv')
        
        # Visualizations
        plot_metrics(df_gt, gt_metrics, "With GT", output_dir)
        
        # Identify Best Methods
        print("\n--- Best Performing Methods (With GT) ---")
        if 'psnr' in df_gt.columns:
            best_psnr = df_gt.groupby('method')['psnr'].mean().idxmax()
            print(f"Highest Average PSNR: {best_psnr} ({df_gt.groupby('method')['psnr'].mean().max():.4f})")
        if 'ssim' in df_gt.columns:
            best_ssim = df_gt.groupby('method')['ssim'].mean().idxmax()
            print(f"Highest Average SSIM: {best_ssim} ({df_gt.groupby('method')['ssim'].mean().max():.4f})")
        if 'brisque' in df_gt.columns:
            best_brisque = df_gt.groupby('method')['brisque'].mean().idxmin()
            print(f"Lowest Average BRISQUE: {best_brisque} ({df_gt.groupby('method')['brisque'].mean().min():.4f})")
        
    else:
        print("\nNo data with Ground Truth found (based on 'psnr'/'ssim' columns).")

    # --- Analysis for Data WITHOUT Ground Truth ---
    if not df_no_gt.empty:
        print("\n" + "="*50)
        print("ANALYSIS: Data WITHOUT Ground Truth (No GT)")
        print("="*50)
        print(f"Number of samples: {len(df_no_gt)}")
        
        # Aggregation
        # For No-GT, we focus on BRISQUE (if available) and Runtime
        no_gt_metrics = ['brisque', 'runtime_sec']
        no_gt_metrics = [m for m in no_gt_metrics if m in df_no_gt.columns]
        
        if no_gt_metrics:
            save_formatted_summary(df_no_gt, no_gt_metrics, output_dir, 'summary_no_gt.csv')
            
            # Visualizations
            plot_metrics(df_no_gt, no_gt_metrics, "No GT", output_dir)
            
            # Identify Best Methods
            print("\n--- Best Performing Methods (No GT) ---")
            if 'brisque' in df_no_gt.columns:
                # Lower BRISQUE is better
                best_brisque = df_no_gt.groupby('method')['brisque'].mean().idxmin()
                print(f"Lowest Average BRISQUE: {best_brisque} ({df_no_gt.groupby('method')['brisque'].mean().min():.4f})")
    else:
        print("\nNo data without Ground Truth found.")

    print("\n" + "="*50)
    print(f"Analysis complete. Results saved to: {output_dir}")

def save_formatted_summary(df, metrics, output_dir, filename):
    """
    Calculates statistics and saves them in a vertical format.
    Row headers: method -> statistic (mean, std, median, min, max)
    Column headers: metrics (psnr, ssim, etc.)
    """
    # Calculate stats
    stats = ['mean', 'std', 'median', 'min', 'max']
    summary = df.groupby('method')[metrics].agg(stats)
    
    # Print summary to console (standard pandas format)
    print("\nSummary Statistics (Mean, Std, Median, Min, Max):")
    print(summary)
    
    # Reshape for CSV output
    # The current summary has MultiIndex columns: (metric, stat)
    # We want rows to be: method, stat
    # And columns to be: metric
    
    # Stack the metrics to move them to index, leaving stats as columns
    # summary.stack(level=0) -> Index: (method, metric), Cols: stats
    
    # Actually, let's just stack the stats level to get what user wants?
    # User wants:
    #          psnr   ssim
    # method1
    #   mean   ...    ...
    #   std    ...    ...
    
    # Current columns are MultiIndex: level 0 = metric, level 1 = stat
    # We want to swap levels so stat is inner index, and metric is column
    
    summary_stacked = summary.stack(level=1) # Index: (method, stat), Cols: metrics
    
    # Reorder stats in the index to match desired order
    # stack() sorts the inner level (stats) alphabetically by default (max, mean, median, min, std)
    # We want: mean, std, median, min, max
    
    # Create a custom sorter
    summary_stacked = summary_stacked.reset_index()
    summary_stacked['level_1'] = pd.Categorical(summary_stacked['level_1'], categories=stats, ordered=True)
    summary_stacked = summary_stacked.sort_values(['method', 'level_1'])
    summary_stacked = summary_stacked.set_index(['method', 'level_1'])
    
    # Rename index for clarity in CSV
    summary_stacked.index.names = ['Method', 'Statistic']
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    summary_stacked.to_csv(output_path)
    print(f"Saved formatted summary to: {output_path}")


def plot_metrics(df, metrics, suffix, output_dir):
    """
    Generates boxplots and bar charts for given metrics.
    """
    sns.set_style("whitegrid")
    
    # 1. Boxplots (Distribution)
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='method', y=metric, data=df)
        plt.title(f'{metric.upper()} Distribution by Method ({suffix})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_{metric}_{suffix.lower().replace(" ", "_")}.png'))
        plt.close()

    # 2. Average Bar Charts with Error Bars
    # Calculate mean and std for plotting
    summary = df.groupby('method')[metrics].agg(['mean', 'std'])
    
    for metric in metrics:
        means = summary[metric]['mean']
        errors = summary[metric]['std']
        
        plt.figure(figsize=(10, 6))
        means.plot(kind='bar', yerr=errors, capsize=4, rot=45, alpha=0.8)
        plt.title(f'Average {metric.upper()} by Method ({suffix})')
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'barplot_{metric}_{suffix.lower().replace(" ", "_")}.png'))
        plt.close()

    # 3. Trade-off Scatter Plot (Quality vs Time)
    # Only if we have both a quality metric and runtime
    quality_metric = None
    if 'psnr' in metrics:
        quality_metric = 'psnr'
    elif 'brisque' in metrics:
        quality_metric = 'brisque'
        
    if quality_metric and 'runtime_sec' in metrics:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='runtime_sec', y=quality_metric, hue='method', data=df, s=100, alpha=0.7)
        
        # Add mean points for each method to make it clearer
        method_means = df.groupby('method')[[ 'runtime_sec', quality_metric ]].mean()
        for method, row in method_means.iterrows():
            plt.text(row['runtime_sec'], row[quality_metric], f' {method}', fontsize=9, fontweight='bold')
            
        plt.title(f'{quality_metric.upper()} vs Runtime ({suffix})')
        plt.xlabel('Runtime (seconds)')
        plt.ylabel(quality_metric.upper())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'scatter_{quality_metric}_vs_runtime_{suffix.lower().replace(" ", "_")}.png'))
        plt.close()

if __name__ == "__main__":
    config_path = "config.yaml" # Default config file path

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: config.yaml not found at {config_path}")
        exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing config.yaml: {exc}")
        exit(1)
    except ModuleNotFoundError:
        print("Error: PyYAML library not found. Please install it using 'pip install PyYAML'")
        exit(1)

    if 'analysis' not in config:
        print("Error: 'analysis' section not found in config.yaml. Please add 'metrics_file' and 'output_dir' under 'analysis'.")
        exit(1)

    metrics_file = config['analysis'].get('metrics_file')
    output_dir = config['analysis'].get('output_dir')

    if not metrics_file:
        print("Error: 'metrics_file' not specified under 'analysis' in config.yaml.")
        exit(1)
    if not output_dir:
        print("Error: 'output_dir' not specified under 'analysis' in config.yaml.")
        exit(1)

    analyze_metrics(metrics_file, output_dir)
