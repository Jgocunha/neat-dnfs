#!/usr/bin/env python3
"""
Script to convert neat-dnfs JSON architecture to LaTeX tables
Generates separate tables for field genes and connection genes

Usage:
    # Place JSON file in same directory as script
    python dnf_latex_generator.py architecture.json -o output.tex
    
    # Generate only tables (for embedding in existing documents)
    python dnf_latex_generator.py architecture.json --tables-only -o tables.tex
"""

import json
import sys
import argparse
from pathlib import Path

def parse_architecture(data):
    """Parse JSON data and categorize components"""
    field_genes = []
    connection_genes = []
    stimuli = []
    noise = []
    
    for component in data:
        label = component.get('label', [])
        if len(label) >= 2:
            component_type = label[1]
            
            if component_type == "neural field":
                field_genes.append(component)
            elif component_type in ["gauss kernel", "mexican hat kernel"]:
                # Check if it's a connection gene (has "cg" in uniqueName)
                if "cg" in component.get('uniqueName', ''):
                    connection_genes.append(component)
                else:
                    # Self-excitation kernel (part of field gene)
                    continue
            elif component_type == "gauss stimulus":
                stimuli.append(component)
            elif component_type == "normal noise":
                noise.append(component)
    
    return field_genes, connection_genes, stimuli, noise

def generate_unified_genome_table(field_genes, connection_genes, data):
    """Generate unified LaTeX table for both field genes and connection genes"""
    latex = []
    latex.append("\\begin{table}[!h]")
    latex.append("\\centering")
    latex.append("\\caption{neat-dnfs Genome Representation}")
    latex.append("\\label{tab:genome}")
    latex.append("\\resizebox{\\textwidth}{!}{%")
    latex.append("\\begin{tabular}{llc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Gene Type/ID} & \\textbf{Connection/Type} & \\textbf{Parameters} \\\\")
    latex.append("\\midrule")
    
    # Create a lookup for all components
    component_lookup = {comp.get('uniqueName', ''): comp for comp in data}
    
    # Add field genes section
    latex.append("\\multicolumn{3}{l}{\\textbf{Field Genes}} \\\\")
    latex.append("\\midrule")
    
    for field in field_genes:
        field_id = field.get('uniqueName', 'N/A')
        field_type = "Input" if any("gs " in inp[0] for inp in field.get('inputs', [])) else "Hidden/Output"
        resting_level = f"{field.get('restingLevel', 'N/A'):.2f}" if field.get('restingLevel') is not None else 'N/A'
        tau = f"{field.get('tau', 'N/A'):.2f}" if field.get('tau') is not None else 'N/A'
        
        # Find associated self-excitation kernel and its parameters
        kernel_info = "None"
        kernel_params = ""
        if field.get('inputs'):
            for inp in field.get('inputs', []):
                kernel_name = inp[0]
                # Check if this is a self-excitation kernel (kernel has this field as input)
                if kernel_name in component_lookup:
                    kernel_comp = component_lookup[kernel_name]
                    kernel_inputs = kernel_comp.get('inputs', [])
                    if kernel_inputs and any(kinp[0] == field_id for kinp in kernel_inputs):
                        # This is a self-excitation kernel
                        kernel_label = kernel_comp.get('label', [])
                        if len(kernel_label) >= 2:
                            if kernel_label[1] == "gauss kernel":
                                kernel_info = "Gaussian"
                                amplitude = f"{kernel_comp.get('amplitude', 'N/A'):.2f}" if kernel_comp.get('amplitude') is not None else 'N/A'
                                width = f"{kernel_comp.get('width', 'N/A'):.2f}" if kernel_comp.get('width') is not None else 'N/A'
                                global_amp = f"{kernel_comp.get('amplitudeGlobal', 'N/A'):.2f}" if kernel_comp.get('amplitudeGlobal') is not None else 'N/A'
                                kernel_params = f", $A={amplitude}$, $\\sigma={width}$, $A_{{glob}}={global_amp}$"
                            elif kernel_label[1] == "mexican hat kernel":
                                kernel_info = "Mexican-hat"
                                amp_exc = f"{kernel_comp.get('amplitudeExc', 'N/A'):.2f}" if kernel_comp.get('amplitudeExc') is not None else 'N/A'
                                width_exc = f"{kernel_comp.get('widthExc', 'N/A'):.2f}" if kernel_comp.get('widthExc') is not None else 'N/A'
                                amp_inh = f"{kernel_comp.get('amplitudeInh', 'N/A'):.2f}" if kernel_comp.get('amplitudeInh') is not None else 'N/A'
                                width_inh = f"{kernel_comp.get('widthInh', 'N/A'):.2f}" if kernel_comp.get('widthInh') is not None else 'N/A'
                                global_amp = f"{kernel_comp.get('amplitudeGlobal', 'N/A'):.2f}" if kernel_comp.get('amplitudeGlobal') is not None else 'N/A'
                                kernel_params = f", $A_{{exc}}={amp_exc}$, $\\sigma_{{exc}}={width_exc}$, $A_{{inh}}={amp_inh}$, $\\sigma_{{inh}}={width_inh}$, $A_{{glob}}={global_amp}$"
                        break
        
        type_info = f"{field_type}, {kernel_info} kernel"
        params = f"$h={resting_level}$, $\\tau={tau}${kernel_params}"
        
        latex.append(f"{field_id} & {type_info} & {params} \\\\")
    
    # Add connection genes section
    latex.append("\\midrule")
    latex.append("\\multicolumn{3}{l}{\\textbf{Connection Genes}} \\\\")
    latex.append("\\midrule")
    
    for conn in connection_genes:
        conn_id = conn.get('uniqueName', 'N/A')
        
        # Parse source and target from connection name
        source_field = "N/A"
        target_field = "N/A"
        if "cg" in conn_id:
            parts = conn_id.split()
            if len(parts) >= 5:
                source_field = f"nf {parts[2]}"
                target_field = f"nf {parts[4]}"
        
        connection_info = f"{source_field} $\\rightarrow$ {target_field}"
        
        # Determine kernel type and parameters
        label = conn.get('label', [])
        kernel_type = label[1] if len(label) >= 2 else "Unknown"
        
        params = ""
        
        if kernel_type == "gauss kernel":
            amplitude = f"{conn.get('amplitude', 'N/A'):.2f}" if conn.get('amplitude') is not None else 'N/A'
            width = f"{conn.get('width', 'N/A'):.2f}" if conn.get('width') is not None else 'N/A'
            global_amp = f"{conn.get('amplitudeGlobal', 'N/A'):.2f}" if conn.get('amplitudeGlobal') is not None else 'N/A'
            params = f"Gaussian: $A={amplitude}$, $\\sigma={width}$, $A_{{glob}}={global_amp}$"
            
        elif kernel_type == "mexican hat kernel":
            amp_exc = f"{conn.get('amplitudeExc', 'N/A'):.2f}" if conn.get('amplitudeExc') is not None else 'N/A'
            width_exc = f"{conn.get('widthExc', 'N/A'):.2f}" if conn.get('widthExc') is not None else 'N/A'
            amp_inh = f"{conn.get('amplitudeInh', 'N/A'):.2f}" if conn.get('amplitudeInh') is not None else 'N/A'
            width_inh = f"{conn.get('widthInh', 'N/A'):.2f}" if conn.get('widthInh') is not None else 'N/A'
            global_amp = f"{conn.get('amplitudeGlobal', 'N/A'):.2f}" if conn.get('amplitudeGlobal') is not None else 'N/A'
            
            params = f"Mexican-hat: $A_{{exc}}={amp_exc}$, $\\sigma_{{exc}}={width_exc}$, $A_{{inh}}={amp_inh}$, $\\sigma_{{inh}}={width_inh}$, $A_{{glob}}={global_amp}$"
        
        # Escape underscores for LaTeX
        conn_id_escaped = conn_id.replace('_', '\\_')
        connection_info_escaped = connection_info.replace('_', '\\_')
        
        latex.append(f"{conn_id_escaped} & {connection_info_escaped} & {params} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")  # Close resizebox
    latex.append("\\end{table}")
    latex.append("")
    
    return '\n'.join(latex)

def generate_latex_document(field_genes, connection_genes, stimuli, data):
    """Generate complete LaTeX document"""
    latex = []
    
    # Document header
    latex.append("\\documentclass{article}")
    latex.append("\\usepackage[utf8]{inputenc}")
    latex.append("\\usepackage{booktabs}")
    latex.append("\\usepackage{array}")
    latex.append("\\usepackage{geometry}")
    latex.append("\\geometry{a4paper, margin=1in}")
    latex.append("\\usepackage{longtable}")
    latex.append("")
    latex.append("\\title{neat-dnfs Architecture Analysis}")
    latex.append("\\author{Generated from JSON}")
    latex.append("\\date{\\today}")
    latex.append("")
    latex.append("\\begin{document}")
    latex.append("\\maketitle")
    latex.append("")
    
    # Summary section
    latex.append("\\section{Architecture Summary}")
    latex.append(f"\\begin{{itemize}}")
    latex.append(f"\\item Total Neural Fields: {len(field_genes)}")
    latex.append(f"\\item Total Inter-field Connections: {len(connection_genes)}")
    latex.append(f"\\item Total Stimuli: {len(stimuli)}")
    latex.append(f"\\end{{itemize}}")
    latex.append("")
    
    # Unified genome table
    latex.append("\\section{Genome Representation}")
    latex.append(generate_unified_genome_table(field_genes, connection_genes, data))
    
    # Stimuli information
    if stimuli:
        latex.append("\\section{External Stimuli}")
        latex.append("\\begin{itemize}")
        for stim in stimuli:
            stim_name = stim.get('uniqueName', 'N/A').replace('_', '\\_')
            position = stim.get('position', 'N/A')
            amplitude = stim.get('amplitude', 'N/A')
            width = stim.get('width', 'N/A')
            latex.append(f"\\item {stim_name}: Position={position}, Amplitude={amplitude}, Width={width}")
        latex.append("\\end{itemize}")
        latex.append("")
    
    latex.append("\\end{document}")
    
    return '\n'.join(latex)

def main():
    parser = argparse.ArgumentParser(description='Convert neat-dnfs JSON architecture to LaTeX tables')
    parser.add_argument('input_file', help='Input JSON file (will look in script directory)')
    parser.add_argument('-o', '--output', help='Output LaTeX file (default: architecture.tex)')
    parser.add_argument('--tables-only', action='store_true', help='Generate only tables, not full document')
    
    args = parser.parse_args()
    
    # Get script directory and construct full path to input file
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input_file
    
    # Read input JSON
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found in script directory '{script_dir}'.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{args.input_file}': {e}")
        sys.exit(1)
    
    # Parse architecture
    field_genes, connection_genes, stimuli, noise = parse_architecture(data)
    
    # Generate LaTeX
    if args.tables_only:
        output = generate_unified_genome_table(field_genes, connection_genes, data)
    else:
        output = generate_latex_document(field_genes, connection_genes, stimuli, data)
    
    # Write output
    output_file = args.output or 'architecture.tex'
    output_path = script_dir / output_file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"LaTeX output written to '{output_path}'")
        print(f"Found {len(field_genes)} field genes and {len(connection_genes)} connection genes")
    except IOError as e:
        print(f"Error writing to '{output_file}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()