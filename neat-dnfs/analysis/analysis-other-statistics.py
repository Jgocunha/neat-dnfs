import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def parse_evolution_timestamps(file_path):
    """Parse the evolution_timestamps.txt file and extract metrics."""
    metrics = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract key information using regex
    num_generations = re.search(r"Number of generations: (\d+)", content)
    start_time = re.search(r"Evolution Start Time: ([\d\-: ]+)", content)
    end_time = re.search(r"Evolution End Time: ([\d\-: ]+)", content)
    duration_seconds = re.search(r"Duration \(seconds\): (\d+)", content)
    
    if num_generations and start_time and end_time and duration_seconds:
        metrics["num_generations"] = int(num_generations.group(1))
        metrics["start_time"] = start_time.group(1)
        metrics["end_time"] = end_time.group(1)
        metrics["duration_seconds"] = int(duration_seconds.group(1))
        
        # Calculate derived metrics
        metrics["seconds_per_generation"] = metrics["duration_seconds"] / metrics["num_generations"]
        
        # Parse datetime objects for additional calculations
        start_dt = datetime.datetime.strptime(metrics["start_time"], "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.datetime.strptime(metrics["end_time"], "%Y-%m-%d %H:%M:%S")
        metrics["duration_hours"] = (end_dt - start_dt).total_seconds() / 3600
    
    return metrics

def parse_field_gene_statistics(file_path):
    """Parse the field_gene_statistics_total.txt file and extract metrics."""
    metrics = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract all numerical values using regex
    pattern = r"Total (\w+(?:\s\w+)*) mutations: (\d+)"
    matches = re.findall(pattern, content)
    
    for mutation_type, count in matches:
        metrics[mutation_type.replace(" ", "_").lower()] = int(count)
    
    # Calculate derived metrics
    if 'gauss_kernel' in metrics and 'mexican_hat_kernel' in metrics and 'oscillatory_kernel' in metrics:
        total_kernel_specific = metrics['gauss_kernel'] + metrics['mexican_hat_kernel'] + metrics['oscillatory_kernel']
        if total_kernel_specific > 0:
            metrics['gauss_kernel_pct'] = metrics['gauss_kernel'] / total_kernel_specific * 100
            metrics['mexican_hat_kernel_pct'] = metrics['mexican_hat_kernel'] / total_kernel_specific * 100
            metrics['oscillatory_kernel_pct'] = metrics['oscillatory_kernel'] / total_kernel_specific * 100
    
    if 'kernel' in metrics and 'neural_field' in metrics:
        metrics['kernel_to_field_ratio'] = metrics['kernel'] / metrics['neural_field'] if metrics['neural_field'] > 0 else 0
    
    return metrics

def parse_genome_statistics(file_path):
    """Parse the genome_statistics_total.txt file and extract metrics."""
    metrics = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract all numerical values using regex
    pattern = r"(\w+(?:\s\w+)*) mutations total: (\d+)"
    matches = re.findall(pattern, content)
    
    for mutation_type, count in matches:
        metrics[mutation_type.replace(" ", "_").lower()] = int(count)
    
    # Calculate total mutations
    total_mutations = sum(metrics.values())
    metrics['total_mutations'] = total_mutations
    
    # Calculate percentages
    for key in list(metrics.keys()):
        if key != 'total_mutations':
            metrics[f"{key}_pct"] = (metrics[key] / total_mutations * 100) if total_mutations > 0 else 0
    
    # Calculate structural vs parametric ratio
    structural_mutations = metrics.get('add_connection_gene', 0) + metrics.get('add_field_gene', 0)
    parametric_mutations = metrics.get('mutate_field_gene', 0) + metrics.get('mutate_connection_gene', 0)
    metrics['structural_mutations'] = structural_mutations
    metrics['parametric_mutations'] = parametric_mutations
    metrics['structural_to_parametric_ratio'] = structural_mutations / parametric_mutations if parametric_mutations > 0 else 0
    
    return metrics

def analyze_run_directory(run_dir):
    """Analyze all statistics files for a specific run."""
    run_metrics = {"run_dir": os.path.basename(run_dir)}
    
    # Parse evolution timestamps
    timestamp_file = os.path.join(run_dir, "statistics", "evolution_timestamps.txt")
    if os.path.exists(timestamp_file):
        timestamp_metrics = parse_evolution_timestamps(timestamp_file)
        run_metrics.update(timestamp_metrics)
    
    # Parse field gene statistics
    field_gene_file = os.path.join(run_dir, "statistics", "field_gene_statistics_total.txt")
    if os.path.exists(field_gene_file):
        field_gene_metrics = parse_field_gene_statistics(field_gene_file)
        run_metrics.update(field_gene_metrics)
    
    # Parse genome statistics
    genome_file = os.path.join(run_dir, "statistics", "genome_statistics_total.txt")
    if os.path.exists(genome_file):
        genome_metrics = parse_genome_statistics(genome_file)
        run_metrics.update(genome_metrics)
    
    return run_metrics

def analyze_all_runs(root_dir):
    """Analyze all run directories."""
    root_path = Path(root_dir)
    
    # Get all run directories
    run_dirs = [d for d in root_path.iterdir() if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2} \d{2}h\d{2}m\d{2}s", d.name)]
    
    if not run_dirs:
        raise ValueError(f"No run directories found in {root_dir}")
    
    print(f"Found {len(run_dirs)} run directories")
    
    # Process each run directory
    all_run_metrics = []
    for run_dir in run_dirs:
        try:
            metrics = analyze_run_directory(str(run_dir))
            if metrics:
                all_run_metrics.append(metrics)
                print(f"Processed: {run_dir.name}")
        except Exception as e:
            print(f"Error processing {run_dir.name}: {e}")
    
    return all_run_metrics

def calculate_aggregate_metrics(all_run_metrics):
    """Calculate aggregate metrics across all runs."""
    # Skip if no metrics were collected
    if not all_run_metrics:
        return {}
    
    # Create a dataframe from all run metrics
    df = pd.DataFrame(all_run_metrics)
    
    # Calculate mean, median, min, max for numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    agg_metrics = {}
    for col in numeric_cols:
        if df[col].notna().any():  # Check if column has any non-NaN values
            agg_metrics[f"{col}_mean"] = df[col].mean()
            agg_metrics[f"{col}_median"] = df[col].median()
            agg_metrics[f"{col}_min"] = df[col].min()
            agg_metrics[f"{col}_max"] = df[col].max()
            agg_metrics[f"{col}_std"] = df[col].std()
    
    return agg_metrics, df

def visualize_metrics(df, output_dir=None):
    """Create visualizations for the metrics."""
    if df.empty:
        print("No data to visualize")
        return
    
    # Set the style
    sns.set(style="whitegrid")
    
    # 1. Distribution of run durations
    if 'duration_hours' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['duration_hours'], kde=True)
        plt.title('Distribution of Run Durations')
        plt.xlabel('Duration (hours)')
        plt.ylabel('Count')
        if output_dir:
            plt.savefig(os.path.join(output_dir, "run_durations.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    # 2. Time efficiency (seconds per generation)
    if 'seconds_per_generation' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['seconds_per_generation'], kde=True)
        plt.title('Time Efficiency (Seconds per Generation)')
        plt.xlabel('Seconds per Generation')
        plt.ylabel('Count')
        if output_dir:
            plt.savefig(os.path.join(output_dir, "time_efficiency.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    # 3. Mutation type breakdown
    mutation_cols = [col for col in df.columns if col.endswith('_pct') and 'kernel' not in col]
    if mutation_cols:
        mutation_data = df[mutation_cols].mean().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=mutation_data.values, y=mutation_data.index)
        plt.title('Average Mutation Type Distribution')
        plt.xlabel('Percentage of Total Mutations')
        plt.ylabel('Mutation Type')
        
        # Clean up y-axis labels
        labels = [label.replace('_pct', '').replace('_', ' ').title() for label in mutation_data.index]
        plt.yticks(range(len(labels)), labels)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "mutation_distribution.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    # 4. Kernel type distribution
    kernel_cols = ['gauss_kernel_pct', 'mexican_hat_kernel_pct', 'oscillatory_kernel_pct']
    if all(col in df.columns for col in kernel_cols):
        kernel_data = df[kernel_cols].mean()
        
        plt.figure(figsize=(10, 6))
        plt.pie(kernel_data, labels=[k.replace('_pct', '').replace('_', ' ').title() for k in kernel_cols], 
                autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
        plt.title('Kernel Type Distribution')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "kernel_distribution.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    # 5. Structural vs Parametric mutations
    if 'structural_mutations' in df.columns and 'parametric_mutations' in df.columns:
        struct_param_data = df[['structural_mutations', 'parametric_mutations']].mean()
        
        plt.figure(figsize=(8, 6))
        plt.pie(struct_param_data, labels=['Structural', 'Parametric'], 
                autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        plt.title('Structural vs Parametric Mutations')
        plt.axis('equal')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "structural_vs_parametric.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    # 6. Correlation heatmap for key metrics
    key_metrics = [
        'duration_hours', 'seconds_per_generation', 
        'structural_to_parametric_ratio', 'kernel_to_field_ratio',
        'gauss_kernel_pct', 'mexican_hat_kernel_pct',
        'toggle_connection_gene_pct'
    ]
    
    available_metrics = [col for col in key_metrics if col in df.columns]
    if len(available_metrics) > 1:
        plt.figure(figsize=(12, 10))
        correlation = df[available_metrics].corr()
        
        # Create a custom diverging colormap
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#6495ED', '#FFFFFF', '#F08080'])
        
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, fmt=".2f", square=True, linewidths=.5)
        plt.title('Correlation Between Key Metrics')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()

def print_summary(agg_metrics, df):
    """Print a summary of the metrics."""
    print("\n===== EXTENDED METRICS SUMMARY =====")
    
    # 1. Time & Performance Metrics
    print("\nTime & Performance Metrics:")
    if 'duration_hours_mean' in agg_metrics:
        print(f"Average Run Duration: {agg_metrics['duration_hours_mean']:.2f} hours")
        print(f"Median Run Duration: {agg_metrics['duration_hours_median']:.2f} hours")
    
    if 'seconds_per_generation_mean' in agg_metrics:
        print(f"Average Time per Generation: {agg_metrics['seconds_per_generation_mean']:.2f} seconds")
        print(f"Median Time per Generation: {agg_metrics['seconds_per_generation_median']:.2f} seconds")
    
    # 2. Mutation Statistics
    print("\nMutation Statistics:")
    
    # Find the most common mutation type
    mutation_cols = [col for col in df.columns if col.endswith('_pct') and 'kernel' not in col]
    if mutation_cols:
        mutation_means = {col: df[col].mean() for col in mutation_cols}
        most_common = max(mutation_means.items(), key=lambda x: x[1])
        print(f"Most Common Mutation Type: {most_common[0].replace('_pct', '').replace('_', ' ').title()} ({most_common[1]:.2f}%)")
    
    if 'structural_to_parametric_ratio_mean' in agg_metrics:
        print(f"Structural to Parametric Mutation Ratio: {agg_metrics['structural_to_parametric_ratio_mean']:.4f}")
        print(f"Average Toggle Mutations: {agg_metrics.get('toggle_connection_gene_pct_mean', 0):.2f}%")
    
    # 3. Neural Field Architecture
    print("\nNeural Field Architecture:")
    
    if 'kernel_to_field_ratio_mean' in agg_metrics:
        print(f"Kernel to Neural Field Mutation Ratio: {agg_metrics['kernel_to_field_ratio_mean']:.2f}")
    
    if 'gauss_kernel_pct_mean' in agg_metrics and 'mexican_hat_kernel_pct_mean' in agg_metrics:
        print("Kernel Type Distribution:")
        print(f"  - Gaussian Kernels: {agg_metrics['gauss_kernel_pct_mean']:.2f}%")
        print(f"  - Mexican Hat Kernels: {agg_metrics['mexican_hat_kernel_pct_mean']:.2f}%")
        print(f"  - Oscillatory Kernels: {agg_metrics.get('oscillatory_kernel_pct_mean', 0):.2f}%")
    
    # 4. Notable Runs
    print("\nNotable Runs:")
    
    if 'duration_hours' in df.columns:
        fastest_run = df.loc[df['duration_hours'].idxmin()]
        slowest_run = df.loc[df['duration_hours'].idxmax()]
        print(f"Fastest Run: {fastest_run['run_dir']} ({fastest_run['duration_hours']:.2f} hours)")
        print(f"Slowest Run: {slowest_run['run_dir']} ({slowest_run['duration_hours']:.2f} hours)")
    
    if 'seconds_per_generation' in df.columns:
        most_efficient = df.loc[df['seconds_per_generation'].idxmin()]
        print(f"Most Efficient Run: {most_efficient['run_dir']} ({most_efficient['seconds_per_generation']:.2f} sec/gen)")

def main():
    # Root directory containing all run folders
    #root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Single bump (self-stabilized)/"
    #root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Single bump (self-sustained)/"
    #root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Logic AND/"
    #root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Selective output/"
    #root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Action simulation/"   
    root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Action execution/"   
    
    # Output directory for plots
    output_dir = None  # Set to a path to save plots, or None to display
    
    # Analyze all runs
    all_run_metrics = analyze_all_runs(root_dir)
    
    # Calculate aggregate metrics
    agg_metrics, metrics_df = calculate_aggregate_metrics(all_run_metrics)
    
    # Print summary
    print_summary(agg_metrics, metrics_df)
    
    # Generate visualizations
    #visualize_metrics(metrics_df, output_dir)
    
if __name__ == "__main__":
    main()