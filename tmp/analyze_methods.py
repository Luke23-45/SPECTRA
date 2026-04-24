import pandas as pd
import os
from pathlib import Path
import numpy as np

# Define the methods and their paths
BASE_DIR = Path(r"C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\SPECTRA\outputs\logs")
METHODS = [
    "bpgs_alb_nyuv2_s42",
    "bpgs_nyuv2_s42",
    "gradnorm_proxy_nyuv2_s42",
    "kendall_nyuv2_s42",
    "static_nyuv2_s42",
    "uwso_nyuv2_s42"
]

# Key validation metrics to analyze
VAL_METRICS = [
    "val/miou",
    "val/depth_abs_rel",
    "val/depth_rmse",
    "val/segmentation_miou",
    "val/segmentation_pixel_acc",
    "val/normals_mean_angle",
    "val/normals_within_11_25",
    "val/total_loss"
]

def load_metrics(method_name):
    """Load metrics.csv for a given method."""
    # Find the metrics.csv file (path structure varies slightly)
    method_dir = BASE_DIR / method_name
    csv_dir = method_dir / "csv_logs"
    
    # Find the nested directory
    nested_dirs = [d for d in csv_dir.iterdir() if d.is_dir()]
    if not nested_dirs:
        print(f"Warning: No nested directory found for {method_name}")
        return None
    
    metrics_path = nested_dirs[0] / "metrics.csv"
    if not metrics_path.exists():
        print(f"Warning: metrics.csv not found for {method_name}")
        return None
    
    df = pd.read_csv(metrics_path)
    return df

def get_final_metrics(df):
    """Get the final (best) validation metrics from the dataframe."""
    if df is None or df.empty:
        return None
    
    # Filter rows that have validation metrics (not NaN)
    final_metrics = {}
    
    for metric in VAL_METRICS:
        if metric in df.columns:
            # Get non-NaN values
            valid_values = df[metric].dropna()
            if not valid_values.empty:
                # For loss metrics, lower is better
                # For accuracy/mIoU metrics, higher is better
                if "loss" in metric.lower() or "abs_rel" in metric.lower() or "rmse" in metric.lower() or "mean_angle" in metric.lower():
                    final_metrics[metric] = valid_values.min()  # Best (lowest) value
                else:
                    final_metrics[metric] = valid_values.max()  # Best (highest) value
    
    return final_metrics

def analyze_all_methods():
    """Analyze all methods and create a comparison table."""
    results = {}
    
    print("Loading metrics for all methods...")
    for method in METHODS:
        print(f"  Loading {method}...")
        df = load_metrics(method)
        final_metrics = get_final_metrics(df)
        if final_metrics:
            results[method] = final_metrics
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    
    # Rank methods for each metric
    print("\n" + "="*80)
    print("METHOD PERFORMANCE COMPARISON")
    print("="*80)
    
    rankings = {}
    for metric in VAL_METRICS:
        if metric in comparison_df.columns:
            # For loss/error metrics, lower is better (rank 1 = lowest)
            # For accuracy/mIoU metrics, higher is better (rank 1 = highest)
            if "loss" in metric.lower() or "abs_rel" in metric.lower() or "rmse" in metric.lower() or "mean_angle" in metric.lower():
                ranking = comparison_df[metric].rank(ascending=True)
            else:
                ranking = comparison_df[metric].rank(ascending=False)
            rankings[metric] = ranking
    
    ranking_df = pd.DataFrame(rankings).T
    
    # Print detailed comparison
    print("\nFinal Metrics:")
    print(comparison_df.round(4))
    
    print("\n" + "-"*80)
    print("Rankings (1 = best):")
    print(ranking_df.astype(int))
    
    # Calculate overall ranking (average rank across all metrics)
    overall_rank = ranking_df.mean(axis=0).sort_values()
    
    print("\n" + "="*80)
    print("OVERALL RANKING (lower average rank = better):")
    print("="*80)
    for i, (method, avg_rank) in enumerate(overall_rank.items(), 1):
        print(f"{i}. {method}: {avg_rank:.2f}")
    
    # Save results to CSV
    output_dir = Path(r"C:\Users\Hellx\Documents\Programming\python\Project\iron\bc\SPECTRA\tmp")
    comparison_df.to_csv(output_dir / "method_comparison.csv")
    ranking_df.to_csv(output_dir / "method_rankings.csv")
    
    print(f"\nResults saved to {output_dir}")
    print("  - method_comparison.csv")
    print("  - method_rankings.csv")
    
    return comparison_df, ranking_df, overall_rank

if __name__ == "__main__":
    analyze_all_methods()
