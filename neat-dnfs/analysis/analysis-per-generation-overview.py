import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def analyze_single_run(file_path, fitness_threshold):
    """
    Extract convergence metrics from a single per_generation_overview.txt file.
    
    Parameters:
    file_path (str): Path to the per_generation_overview.txt file
    fitness_threshold (float): Fitness threshold to consider the run successful
    
    Returns:
    dict: Dictionary containing the convergence metrics
    """
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract data for each generation
    generation_pattern = r"Current generation: (\d+).*?Best solution: \[solution \d+ \[ fit\.: ([\d\.]+).*?genome \((.*?)\).*?field genes \{(.*?)\}.*?connection genes \{(.*?)\}"
    generations_data = re.findall(generation_pattern, content, re.DOTALL)
    
    if not generations_data:
        raise ValueError("Could not parse generation data from the file")
    
    # Convert to numeric values for standard metrics
    generations = [int(gen) for gen, _, _, _, _ in generations_data]
    fitness_values = [float(fit) for _, fit, _, _, _ in generations_data]
    
    # Extract genome info for the final generation
    final_gen_idx = generations.index(max(generations))
    final_gen_data = generations_data[final_gen_idx]
    
    # Parse field genes to count hidden fields
    field_genes_str = final_gen_data[3]
    field_gene_pattern = r"fg \(id: \d+, type: (INPUT|OUTPUT|HIDDEN)\)"
    field_types = re.findall(field_gene_pattern, field_genes_str)
    hidden_fields_count = field_types.count("HIDDEN")
    
    # Parse connection genes to count enabled connections
    connection_genes_str = final_gen_data[4]
    connection_gene_pattern = r"enabled: (true|false)"
    connection_states = re.findall(connection_gene_pattern, connection_genes_str)
    enabled_connections_count = connection_states.count("true")
    
    # Calculate fitness improvements between generations
    fitness_improvements = []
    for i in range(1, len(fitness_values)):
        improvement = max(0, fitness_values[i] - fitness_values[i-1])
        fitness_improvements.append(improvement)
    
    # Check if the run reached the fitness threshold
    max_fitness = max(fitness_values)
    success = max_fitness >= fitness_threshold
    
    # Calculate generations to threshold
    generation_to_threshold = None
    if success:
        for i, fitness in enumerate(fitness_values):
            if fitness >= fitness_threshold:
                generation_to_threshold = generations[i]
                break
    
    # Calculate average improvement per generation
    avg_improvement_per_gen = sum(fitness_improvements) / len(fitness_improvements) if fitness_improvements else 0
    
    # Create metrics dictionary
    metrics = {
        "success": success,
        "max_fitness": max_fitness,
        "total_generations": len(generations),
        "generation_to_threshold": generation_to_threshold,
        "avg_improvement_per_gen": avg_improvement_per_gen,
        "generations": generations,
        "fitness_values": fitness_values,
        "fitness_improvements": fitness_improvements,
        "run_dir": os.path.basename(os.path.dirname(os.path.dirname(file_path))),
        "hidden_fields_count": hidden_fields_count,
        "enabled_connections_count": enabled_connections_count
    }
    
    return metrics

def analyze_multiple_runs(root_dir, fitness_threshold):
    """
    Analyze multiple evolutionary runs by processing all per_generation_overview.txt files.
    
    Parameters:
    root_dir (str): Path to the root directory containing all run folders
    fitness_threshold (float): Fitness threshold to consider a run successful
    
    Returns:
    dict: Dictionary containing aggregated metrics
    """
    root_path = Path(root_dir)
    all_metrics = []
    
    # Get all run directories
    run_dirs = [d for d in root_path.iterdir() if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2} \d{2}h\d{2}m\d{2}s", d.name)]
    
    if not run_dirs:
        raise ValueError(f"No run directories found in {root_dir}")
    
    print(f"Found {len(run_dirs)} run directories")
    
    # Process each run directory
    for run_dir in run_dirs:
        stats_file = run_dir / "statistics" / "per_generation_overview.txt"
        
        if not stats_file.exists():
            print(f"Warning: No per_generation_overview.txt found in {run_dir}")
            continue
        
        try:
            metrics = analyze_single_run(str(stats_file), fitness_threshold)
            all_metrics.append(metrics)
            print(f"Processed: {run_dir.name} - Maximum fitness: {metrics['max_fitness']:.4f}")
        except Exception as e:
            print(f"Error processing {run_dir.name}: {e}")
    
    # Aggregate metrics across all runs
    return aggregate_metrics(all_metrics)

def aggregate_metrics(all_metrics):
    """
    Aggregate metrics across multiple runs.
    
    Parameters:
    all_metrics (list): List of metrics dictionaries from all runs
    
    Returns:
    dict: Dictionary containing aggregated metrics
    """
    total_runs = len(all_metrics)
    successful_runs = [m for m in all_metrics if m["success"]]
    num_successful = len(successful_runs)
    
    # Calculate metrics for successful runs
    if num_successful > 0:
        # Generations to threshold
        generations_to_threshold = [m["generation_to_threshold"] for m in successful_runs]
        mean_generations = np.mean(generations_to_threshold)
        median_generations = np.median(generations_to_threshold)
        std_generations = np.std(generations_to_threshold)
        
        # Architecture complexity metrics
        hidden_fields_counts = [m["hidden_fields_count"] for m in successful_runs]
        enabled_connections_counts = [m["enabled_connections_count"] for m in successful_runs]
        
        mean_hidden_fields = np.mean(hidden_fields_counts)
        median_hidden_fields = np.median(hidden_fields_counts)
        std_hidden_fields = np.std(hidden_fields_counts)
        
        mean_enabled_connections = np.mean(enabled_connections_counts)
        median_enabled_connections = np.median(enabled_connections_counts)
        std_enabled_connections = np.std(enabled_connections_counts)
        
        # Convergence rate
        convergence_rates = []
        for m in successful_runs:
            first_fitness = m["fitness_values"][0]
            threshold_idx = m["generations"].index(m["generation_to_threshold"])
            threshold_fitness = m["fitness_values"][threshold_idx]
            generations_taken = m["generation_to_threshold"]
            # Avoid division by zero
            if generations_taken > 0:
                convergence_rates.append((threshold_fitness - first_fitness) / generations_taken)
        
        mean_convergence_rate = np.mean(convergence_rates) if convergence_rates else 0
        
        # Average improvement per generation
        avg_improvements = [m["avg_improvement_per_gen"] for m in successful_runs]
        mean_improvement = np.mean(avg_improvements)
    else:
        mean_generations = median_generations = std_generations = 0
        mean_convergence_rate = mean_improvement = 0
        mean_hidden_fields = median_hidden_fields = std_hidden_fields = 0
        mean_enabled_connections = median_enabled_connections = std_enabled_connections = 0
    
    return {
        "total_runs": total_runs,
        "successful_runs": num_successful,
        "success_rate": num_successful / total_runs if total_runs > 0 else 0,
        "mean_generations_to_threshold": mean_generations,
        "median_generations_to_threshold": median_generations,
        "std_generations_to_threshold": std_generations,
        "mean_convergence_rate": mean_convergence_rate,
        "mean_improvement_per_gen": mean_improvement,
        "mean_hidden_fields": mean_hidden_fields,
        "median_hidden_fields": median_hidden_fields,
        "std_hidden_fields": std_hidden_fields,
        "mean_enabled_connections": mean_enabled_connections,
        "median_enabled_connections": median_enabled_connections,
        "std_enabled_connections": std_enabled_connections,
        "all_run_metrics": all_metrics,
    }

def plot_aggregated_results(aggregated_metrics, fitness_threshold, output_dir=None):
    """
    Generate plots for the aggregated results.
    
    Parameters:
    aggregated_metrics (dict): Dictionary containing aggregated metrics
    output_dir (str): Directory to save plots (if None, plots are displayed)
    """
    successful_runs = [m for m in aggregated_metrics["all_run_metrics"] if m["success"]]
    
    if not successful_runs:
        print("No successful runs to plot")
        return
    
    # Plot histogram of generations to threshold
    plt.figure(figsize=(10, 6))
    generations_to_threshold = [m["generation_to_threshold"] for m in successful_runs]
    plt.hist(generations_to_threshold, bins=min(20, len(generations_to_threshold)), alpha=0.7)
    
    plt.axvline(x=aggregated_metrics["mean_generations_to_threshold"], color='r', 
                linestyle='--', label=f'Mean: {aggregated_metrics["mean_generations_to_threshold"]:.2f}')
    plt.axvline(x=aggregated_metrics["median_generations_to_threshold"], color='g', 
                linestyle='--', label=f'Median: {aggregated_metrics["median_generations_to_threshold"]:.2f}')
    
    plt.title('Generations to Reach Fitness Threshold')
    plt.xlabel('Generations')
    plt.ylabel('Number of Runs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "generations_to_threshold.png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Plot fitness curves for all successful runs
    plt.figure(figsize=(12, 8))
    
    for i, metrics in enumerate(successful_runs):
        # Normalize to show only up to threshold
        if metrics["generation_to_threshold"] is not None:
            threshold_idx = metrics["generations"].index(metrics["generation_to_threshold"])
            gens = metrics["generations"][:threshold_idx+1]
            fits = metrics["fitness_values"][:threshold_idx+1]
        else:
            gens = metrics["generations"]
            fits = metrics["fitness_values"]
        
        # Plot with low alpha to handle multiple runs
        plt.plot(gens, fits, alpha=0.3, label=f"Run {metrics['run_dir']}" if i < 10 else None)
    
    plt.axhline(y=fitness_threshold, color='r', linestyle='--', label='Fitness Threshold')
    plt.title(f'Fitness Evolution (Successful Runs Only, n={len(successful_runs)})')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    
    # Only show first 10 runs in legend to avoid clutter
    if len(successful_runs) > 10:
        plt.legend(loc='lower right', fontsize='small', ncol=2)
    else:
        plt.legend(loc='lower right')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "fitness_evolution.png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Plot architecture complexity metrics
    if successful_runs:
        plt.figure(figsize=(12, 6))
        
        # Set up a 1x2 subplot grid
        plt.subplot(1, 2, 1)
        hidden_fields = [m["hidden_fields_count"] for m in successful_runs]
        plt.hist(hidden_fields, bins=range(min(hidden_fields), max(hidden_fields)+2), alpha=0.7)
        plt.axvline(x=aggregated_metrics["mean_hidden_fields"], color='r', 
                   linestyle='--', label=f'Mean: {aggregated_metrics["mean_hidden_fields"]:.2f}')
        plt.title('Hidden Fields in Successful Solutions')
        plt.xlabel('Number of Hidden Fields')
        plt.ylabel('Number of Runs')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        enabled_connections = [m["enabled_connections_count"] for m in successful_runs]
        plt.hist(enabled_connections, bins=range(min(enabled_connections), max(enabled_connections)+2), alpha=0.7)
        plt.axvline(x=aggregated_metrics["mean_enabled_connections"], color='r', 
                   linestyle='--', label=f'Mean: {aggregated_metrics["mean_enabled_connections"]:.2f}')
        plt.title('Enabled Connections in Successful Solutions')
        plt.xlabel('Number of Enabled Connections')
        plt.ylabel('Number of Runs')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "architecture_complexity.png"), dpi=300, bbox_inches='tight')
        else:
            plt.show()

def print_summary(aggregated_metrics):
    """
    Print a summary of the aggregated metrics.
    
    Parameters:
    aggregated_metrics (dict): Dictionary containing aggregated metrics
    """
    print("\n===== CONVERGENCE METRICS SUMMARY =====")
    print(f"Total Runs: {aggregated_metrics['total_runs']}")
    print(f"Successful Runs: {aggregated_metrics['successful_runs']} ({aggregated_metrics['success_rate']*100:.1f}%)")
    
    if aggregated_metrics['successful_runs'] > 0:
        print("\nFor Successful Runs:")
        print(f"Mean Generations to Threshold: {aggregated_metrics['mean_generations_to_threshold']:.2f}")
        print(f"Median Generations to Threshold: {aggregated_metrics['median_generations_to_threshold']:.2f}")
        print(f"Standard Deviation of Generations to Threshold: {aggregated_metrics['std_generations_to_threshold']:.2f}")
        print(f"Mean Convergence Rate (fitness gain per generation): {aggregated_metrics['mean_convergence_rate']:.4f}")
        print(f"Mean Fitness Improvement per Generation: {aggregated_metrics['mean_improvement_per_gen']:.4f}")
        
        print("\nArchitecture Complexity (Successful Solutions):")
        print(f"Hidden Fields Count: Mean: {aggregated_metrics['mean_hidden_fields']:.2f}, Median: {aggregated_metrics['median_hidden_fields']:.1f}, Std Dev: {aggregated_metrics['std_hidden_fields']:.2f}")
        print(f"Enabled Connections Count: Mean: {aggregated_metrics['mean_enabled_connections']:.2f}, Median: {aggregated_metrics['median_enabled_connections']:.1f}, Std Dev: {aggregated_metrics['std_enabled_connections']:.2f}")
    
    # Add additional run-specific info
    all_runs = aggregated_metrics["all_run_metrics"]
    max_fitness_run = max(all_runs, key=lambda x: x["max_fitness"])
    print(f"\nHighest Fitness Achieved: {max_fitness_run['max_fitness']:.4f} (Run: {max_fitness_run['run_dir']})")
    
    if aggregated_metrics['successful_runs'] > 0:
        successful_runs = [r for r in all_runs if r["success"]]
        
        # Fastest successful run
        fastest_run = min(successful_runs, key=lambda x: x["generation_to_threshold"])
        print(f"Fastest Successful Run: {fastest_run['generation_to_threshold']} generations (Run: {fastest_run['run_dir']})")
        
        # Slowest successful run
        slowest_run = max(successful_runs, key=lambda x: x["generation_to_threshold"])
        print(f"Slowest Successful Run: {slowest_run['generation_to_threshold']} generations (Run: {slowest_run['run_dir']})")
        
        # Most/least complex architectures
        most_hidden_run = max(successful_runs, key=lambda x: x["hidden_fields_count"])
        print(f"Most Hidden Fields: {most_hidden_run['hidden_fields_count']} (Run: {most_hidden_run['run_dir']})")
        
        most_connections_run = max(successful_runs, key=lambda x: x["enabled_connections_count"])
        print(f"Most Enabled Connections: {most_connections_run['enabled_connections_count']} (Run: {most_connections_run['run_dir']})")

def main():
    # Root directory containing all run folders
    #root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Single bump (self-stabilized)/"
    #root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Single bump (self-sustained)/"
    #root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Logic AND/"
    #root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Selective output/"
    #root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Action simulation/"
    root_dir = "C:/dev-files/neat-dnfs/neat-dnfs/data/Action execution/"   

    # Fitness threshold to consider a run successful
    fitness_threshold = 0.85  # Adjust based on your specific requirements
    
    # Output directory for plots
    output_dir = None  # Set to a path to save plots, or None to display
    
    # Analyze all runs
    aggregated_metrics = analyze_multiple_runs(root_dir, fitness_threshold)
    
    # Print summary
    print_summary(aggregated_metrics)
    
    # Generate plots
    #plot_aggregated_results(aggregated_metrics, fitness_threshold, output_dir)

if __name__ == "__main__":
    main()