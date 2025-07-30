#!/usr/bin/env python3
"""
Script to calculate the mean AUROC score from probe evaluation results.
"""

import json
import statistics
import sys
import argparse
from pathlib import Path

def calculate_mean_auroc(jsonl_file_path):
    """
    Read a JSONL file containing probe evaluation results and calculate the mean AUROC.
    
    Args:
        jsonl_file_path (str): Path to the JSONL file
        
    Returns:
        tuple: (mean_auroc, auroc_scores, dataset_names, model_name)
    """
    auroc_scores = []
    dataset_names = []
    model_name = None
    
    with open(jsonl_file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            # Parse each JSON line
            data = json.loads(line.strip())
            
            # Extract AUROC score and dataset name
            auroc = data['metrics']['metrics']['auroc']
            dataset_name = data['dataset_name']
            
            # Extract model name from the first line (should be the same for all lines in a file)
            if model_name is None:
                model_name = data['config']['model_name']
            
            auroc_scores.append(auroc)
            dataset_names.append(dataset_name)
    
    # Calculate mean
    mean_auroc = statistics.mean(auroc_scores) if auroc_scores else 0.0
    
    return mean_auroc, auroc_scores, dataset_names, model_name

def process_single_file(file_path, quiet=False):
    """Process a single file and return results."""
    try:
        mean_auroc, auroc_scores, dataset_names, model_name = calculate_mean_auroc(file_path)
        
        if not quiet:
            print(f"\nFile: {file_path}")
            print(f"Model: {model_name}")
            print("AUROC Scores by Dataset:")
            print("=" * 50)
            for dataset, auroc in zip(dataset_names, auroc_scores):
                print(f"{dataset:<35}: {auroc:.6f}")
            
            print("\n" + "=" * 50)
            print(f"Mean AUROC across all datasets: {mean_auroc:.6f}")
            # print(f"Number of datasets: {len(auroc_scores)}")
            
            # if len(auroc_scores) > 1:
            #     print(f"Standard deviation: {statistics.stdev(auroc_scores):.6f}")
            #     print(f"Min AUROC: {min(auroc_scores):.6f}")
            #     print(f"Max AUROC: {max(auroc_scores):.6f}")
        
        return {
            'file_path': file_path,
            'model_name': model_name,
            'mean_auroc': mean_auroc,
            'auroc_scores': auroc_scores,
            'dataset_names': dataset_names
        }
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in '{file_path}': {e}", file=sys.stderr)
        return None
    except KeyError as e:
        print(f"Error: Expected field {e} not found in '{file_path}'.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error processing '{file_path}': {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description='Calculate mean AUROC from probe evaluation results')
    parser.add_argument('files', nargs='*', 
                       default=["data/results/evaluate_probes/results_20250729_174007_p6Mo.jsonl"],
                       help='Path(s) to the JSONL results file(s)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Only output summary information')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='Show summary comparison across all files')
    
    args = parser.parse_args()
    
    # If no files specified, use default
    if not args.files:
        args.files = ["data/results/evaluate_probes/results_20250729_174007_p6Mo.jsonl"]
    
    all_results = []
    
    # Process each file
    for file_path in args.files:
        result = process_single_file(file_path, args.quiet)
        if result:
            all_results.append(result)
            
            if args.quiet:
                print(f"{Path(file_path).name}: {result['model_name']} - Mean AUROC: {result['mean_auroc']:.6f}")
    
    # Show summary if requested and multiple files processed
    if args.summary and len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)
        print(f"{'File':<30} {'Model':<35} {'Mean AUROC':<12} {'Datasets'}")
        print("-" * 70)
        
        for result in all_results:
            file_name = Path(result['file_path']).name
            model_name = result['model_name'][:33] + "..." if len(result['model_name']) > 35 else result['model_name']
            print(f"{file_name:<30} {model_name:<35} {result['mean_auroc']:<12.6f} {len(result['auroc_scores'])}")
        
        # Overall statistics
        all_mean_aurocs = [r['mean_auroc'] for r in all_results]
        print("\n" + "-" * 70)
        print(f"Overall mean AUROC across all files: {statistics.mean(all_mean_aurocs):.6f}")
        if len(all_mean_aurocs) > 1:
            print(f"Standard deviation across files: {statistics.stdev(all_mean_aurocs):.6f}")
            print(f"Best performing file: {max(all_results, key=lambda x: x['mean_auroc'])['file_path']}")
            print(f"Best AUROC: {max(all_mean_aurocs):.6f}")

if __name__ == "__main__":
    main() 