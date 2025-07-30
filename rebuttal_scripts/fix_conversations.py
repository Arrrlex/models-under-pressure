#!/usr/bin/env python3
"""
Script to fix conversation turn violations in evaluation datasets.

Fixes applied:
1. Toolace dataset: Convert 'tool' roles to 'assistant' roles
2. Anthropic/MTS datasets: Merge consecutive assistant turns
"""

import json
import yaml
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path

def load_config(config_path: str) -> Dict[str, str]:
    """Load the configuration file and return active datasets."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Filter out commented entries
    active_datasets = {}
    for key, value in config.items():
        if not key.startswith('#'):
            active_datasets[key] = value
    
    return active_datasets

def fix_toolace_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix toolace conversations by converting 'tool' roles to 'assistant' roles."""
    fixed_conversation = []
    for turn in conversation:
        fixed_turn = turn.copy()
        if fixed_turn.get('role') == 'tool':
            fixed_turn['role'] = 'assistant'
        fixed_conversation.append(fixed_turn)
    return fixed_conversation

def split_mts_context(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split MTS conversations at '## Context' to create system -> user -> assistant flow."""
    if not conversation:
        return conversation
    
    # Check if first turn is system and contains "## Context"
    first_turn = conversation[0]
    if (first_turn.get('role') != 'system' or 
        '## Context' not in first_turn.get('content', '')):
        return conversation  # No splitting needed
    
    content = first_turn.get('content', '')
    context_pos = content.find('## Context')
    
    if context_pos == -1:
        return conversation  # No context marker found
    
    # Split the content
    system_content = content[:context_pos].rstrip()
    user_content = content[context_pos:]
    
    # Create new turns
    new_system_turn = first_turn.copy()
    new_system_turn['content'] = system_content
    
    new_user_turn = {
        'role': 'user',
        'content': user_content
    }
    
    # Copy any additional fields from system turn to user turn if needed
    for key, value in first_turn.items():
        if key not in ['role', 'content']:
            new_user_turn[key] = value
    
    # Rebuild conversation with split turns
    new_conversation = [new_system_turn, new_user_turn] + conversation[1:]
    
    return new_conversation

def merge_consecutive_assistants(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive assistant turns into single turns."""
    if not conversation:
        return conversation
    
    merged = []
    i = 0
    while i < len(conversation):
        current_turn = conversation[i].copy()
        
        # Check if this is an assistant turn followed by another assistant turn
        if (i < len(conversation) - 1 and 
            current_turn.get('role') == 'assistant' and 
            conversation[i + 1].get('role') == 'assistant'):
            
            # Merge consecutive assistant turns
            merged_content_parts = [current_turn.get('content', '')]
            j = i + 1
            
            # Collect all consecutive assistant turns
            while (j < len(conversation) and 
                   conversation[j].get('role') == 'assistant'):
                merged_content_parts.append(conversation[j].get('content', ''))
                j += 1
            
            # Create merged turn
            merged_content = '\n\n'.join(part.strip() for part in merged_content_parts if part.strip())
            merged_turn = {
                'role': 'assistant',
                'content': merged_content
            }
            
            # Copy any additional fields from the first turn
            for key, value in current_turn.items():
                if key not in ['role', 'content']:
                    merged_turn[key] = value
            
            merged.append(merged_turn)
            i = j  # Skip all the merged turns
        else:
            merged.append(current_turn)
            i += 1
    
    return merged

def fix_conversation(conversation: List[Dict[str, Any]], dataset_name: str) -> List[Dict[str, Any]]:
    """Apply appropriate fixes based on dataset type."""
    if dataset_name == 'toolace':
        # First convert tool roles to assistant, then merge consecutive assistants
        fixed_conversation = fix_toolace_conversation(conversation)
        return merge_consecutive_assistants(fixed_conversation)
    elif dataset_name == 'mts':
        # First split at "## Context", then merge consecutive assistants
        split_conversation = split_mts_context(conversation)
        return merge_consecutive_assistants(split_conversation)
    elif dataset_name == 'anthropic':
        return merge_consecutive_assistants(conversation)
    else:
        # For other datasets (mt, mental_health, redteaming), no fixes needed
        return conversation

def process_dataset(input_path: str, output_path: str, dataset_name: str) -> Tuple[int, int]:
    """
    Process a dataset file and save the fixed version.
    Returns (total_conversations, fixed_conversations)
    """
    if not os.path.exists(input_path):
        print(f"Warning: Input file not found: {input_path}")
        return 0, 0
    
    total_count = 0
    fixed_count = 0
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    total_count += 1
                    
                    # Parse the inputs field
                    inputs_str = data.get("inputs", "[]")
                    if isinstance(inputs_str, str):
                        original_conversation = json.loads(inputs_str)
                    else:
                        original_conversation = inputs_str
                    
                    # Apply fixes
                    fixed_conversation = fix_conversation(original_conversation, dataset_name)
                    
                    # Check if conversation was actually modified
                    if fixed_conversation != original_conversation:
                        fixed_count += 1
                    
                    # Update the data with fixed conversation
                    data["inputs"] = json.dumps(fixed_conversation)
                    
                    # Write the fixed line
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num} in {input_path}: {e}")
                    # Write the original line if we can't parse it
                    outfile.write(line + '\n')
                except Exception as e:
                    print(f"Error processing line {line_num} in {input_path}: {e}")
                    # Write the original line if there's any other error
                    outfile.write(line + '\n')
    
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")
        return 0, 0
    
    return total_count, fixed_count

def generate_output_path(input_path: str) -> str:
    """Generate output path by replacing date with jul_30."""
    path = Path(input_path)
    
    # Replace the date pattern with jul_30
    stem = path.stem
    
    # Common patterns to replace
    date_patterns = ['apr_23', 'apr_22', 'apr_30', 'jul_29']
    
    new_stem = stem
    for pattern in date_patterns:
        if pattern in stem:
            new_stem = stem.replace(pattern, 'jul_30')
            break
    
    # If no date pattern found, just append _jul_30
    if new_stem == stem:
        new_stem = f"{stem}_jul_30"
    
    return str(path.parent / f"{new_stem}{path.suffix}")

def create_fixed_config(original_config_path: str, fixed_config_path: str, 
                       dataset_mappings: Dict[str, str]) -> None:
    """Create a new config file pointing to the fixed datasets."""
    
    # Load original config
    with open(original_config_path, 'r') as f:
        original_config = yaml.safe_load(f)
    
    # Create new config with updated paths
    fixed_config = {}
    for key, value in original_config.items():
        if key.startswith('#'):
            # Keep commented entries as-is
            fixed_config[key] = value
        elif key in dataset_mappings:
            # Update path to fixed version
            fixed_config[key] = dataset_mappings[key]
        else:
            # Keep other entries as-is
            fixed_config[key] = value
    
    # Write new config
    with open(fixed_config_path, 'w') as f:
        yaml.dump(fixed_config, f, default_flow_style=False, sort_keys=False)

def main():
    """Main function to fix all datasets and create new config."""
    # Get config paths relative to script location
    script_dir = Path(__file__).parent
    config_dir = script_dir.parent / "config" / "eval_datasets"
    original_config_path = str(config_dir / "test_balanced.yaml")
    fixed_config_path = str(config_dir / "test_balanced_fixed.yaml")
    
    print("Loading original configuration...")
    datasets = load_config(original_config_path)
    
    print(f"Found {len(datasets)} datasets to process:")
    for name, path in datasets.items():
        print(f"  - {name}: {path}")
    
    print("\nProcessing datasets...\n")
    
    dataset_mappings = {}
    total_fixed = 0
    
    for dataset_name, input_path in datasets.items():
        output_path = generate_output_path(input_path)
        dataset_mappings[dataset_name] = output_path
        
        print(f"Processing {dataset_name}...")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        
        total_conversations, fixed_conversations = process_dataset(input_path, output_path, dataset_name)
        
        print(f"  Processed {total_conversations} conversations")
        print(f"  Fixed {fixed_conversations} conversations")
        total_fixed += fixed_conversations
        print()
    
    print("Creating fixed configuration file...")
    create_fixed_config(original_config_path, fixed_config_path, dataset_mappings)
    print(f"Created: {fixed_config_path}")
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total conversations fixed: {total_fixed}")
    print(f"Fixed datasets saved with '_jul_30' suffix")
    print(f"New config file created: {fixed_config_path}")
    print("\nReady for validation!")

if __name__ == "__main__":
    main() 