import re
import numpy as np
import matplotlib.pyplot as plt

def analyze_single_run(file_path, fitness_threshold=0.8):
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
    generation_pattern = r"Current generation: (\d+).*?Best solution: \[solution \d+ \[ fit\.: ([\d\.]+)"
    generations_data = re.findall(generation_pattern, content, re.DOTALL)
    
    if not generations_data:
        raise ValueError("Could not parse generation data from the file")
    
    # Convert to numeric values
    generations = [int(gen) for gen, _ in generations_data]
    fitness_values = [float(fit) for _, fit in generations_data]
    
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
        "fitness_improvements": fitness_improvements
    }
    
    return metrics

def plot_convergence(metrics, output_path=None):
    """
    Generate convergence plots based on the metrics.
    
    Parameters:
    metrics (dict): Dictionary containing the metrics
    output_path (str): Path to save the plots (if None, plots are displayed)
    """
    # Plot fitness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["generations"], metrics["fitness_values"], marker='o', linestyle='-')
    
    if metrics["success"]:
        # Mark the generation where threshold was reached
        threshold_gen = metrics["generation_to_threshold"]
        threshold_idx = metrics["generations"].index(threshold_gen)
        plt.scatter([threshold_gen], [metrics["fitness_values"][threshold_idx]], 
                   color='red', s=100, zorder=5, label=f'Threshold reached at generation {threshold_gen}')
    
    plt.axhline(y=0.8, color='r', linestyle='--', label='Fitness Threshold')
    plt.title('Fitness Evolution Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_path:
        plt.savefig(f"{output_path}_fitness.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Plot fitness improvement per generation
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(metrics["fitness_improvements"])+1), metrics["fitness_improvements"])
    plt.axhline(y=metrics["avg_improvement_per_gen"], color='r', linestyle='--', 
               label=f'Avg. Improvement: {metrics["avg_improvement_per_gen"]:.4f}')
    plt.title('Fitness Improvement Per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Improvement')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    if output_path:
        plt.savefig(f"{output_path}_improvements.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()

def print_summary(metrics):
    """
    Print a summary of the metrics.
    
    Parameters:
    metrics (dict): Dictionary containing the metrics
    """
    print("\n===== CONVERGENCE METRICS SUMMARY =====")
    print(f"Total Generations: {metrics['total_generations']}")
    print(f"Maximum Fitness Achieved: {metrics['max_fitness']:.4f}")
    print(f"Average Fitness Improvement per Generation: {metrics['avg_improvement_per_gen']:.4f}")
    
    if metrics["success"]:
        print(f"Fitness Threshold Reached: Yes (at generation {metrics['generation_to_threshold']})")
    else:
        print("Fitness Threshold Reached: No")
    
    # Calculate convergence rate (fitness gain per generation)
    first_fitness = metrics["fitness_values"][0]
    last_fitness = metrics["fitness_values"][-1]
    total_improvement = last_fitness - first_fitness
    convergence_rate = total_improvement / (len(metrics["generations"]) - 1) if len(metrics["generations"]) > 1 else 0
    
    print(f"Overall Convergence Rate: {convergence_rate:.4f} fitness/generation")
    
    # Find largest fitness jumps
    if metrics["fitness_improvements"]:
        max_improvement_gen = np.argmax(metrics["fitness_improvements"]) + 1
        max_improvement = max(metrics["fitness_improvements"])
        print(f"Largest Fitness Jump: {max_improvement:.4f} (generation {max_improvement_gen})")

def main():
    # Path to the per_generation_overview.txt file
    file_path = "C:/dev-files/neat-dnfs/neat-dnfs/data/Two robot team/2025-04-23 07h57m11s/statistics/per_generation_overview.txt"  # Update this to your file path


    # Fitness threshold to consider the run successful
    fitness_threshold = 0.8  # Adjust based on your specific requirements
    
    # Analyze the run
    metrics = analyze_single_run(file_path, fitness_threshold)
    
    # Print summary
    print_summary(metrics)
    
    # Generate plots
    plot_convergence(metrics)

if __name__ == "__main__":
    main()