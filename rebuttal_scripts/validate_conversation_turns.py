#!/usr/bin/env python3
"""
Script to validate conversation turns in evaluation datasets.

Rules for good conversation turns:
- system -> user is fine
- system -> assistant is not fine
- system -> user -> assistant is fine
- system -> user -> assistant -> user is fine, and any continuation is fine
- ... assistant -> assistant is not fine
- any role other than system, user, assistant is not fine
"""

import json
import yaml
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_config(config_path: str) -> Dict[str, str]:
    """Load the configuration file and return active datasets."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Filter out commented entries (those starting with #)
    active_datasets = {}
    for key, value in config.items():
        if not key.startswith('#'):
            active_datasets[key] = value
    
    return active_datasets

def validate_conversation_turns(conversation: List[Dict[str, Any]]) -> tuple[bool, str]:
    """
    Validate conversation turns according to the rules.
    
    Returns (is_valid, error_message).
    """
    if not conversation:
        return False, "Empty conversation"
    
    valid_roles = {"system", "user", "assistant"}
    
    for i, turn in enumerate(conversation):
        role = turn.get("role", "").strip()
        
        # Check if role is valid
        if role not in valid_roles:
            return False, f"Invalid role '{role}' at position {i+1}"
        
        # Check turn-to-turn transitions
        if i > 0:
            prev_role = conversation[i-1].get("role", "").strip()
            
            # Rule: assistant -> assistant is not allowed
            if prev_role == "assistant" and role == "assistant":
                return False, f"Consecutive assistant turns at positions {i} and {i+1}"
            
            # Rule: system -> assistant is not allowed
            if prev_role == "system" and role == "assistant":
                return False, f"System -> Assistant transition at positions {i} and {i+1}"
    
    return True, ""

def print_conversation(conversation: List[Dict[str, Any]], conversation_id: str, error_msg: str):
    """Print the entire conversation in a readable format."""
    print(f"\n{'='*80}")
    print(f"INVALID CONVERSATION ID: {conversation_id}")
    print(f"ERROR: {error_msg}")
    print(f"{'='*80}")
    
    roles = [turn['role'] for turn in conversation]
    print(f"Role sequence: {' -> '.join(roles)}")
    print(f"Turn count: {len(conversation)}")
    print()
    
    for i, turn in enumerate(conversation, 1):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        
        # Truncate very long content for readability
        if len(content) > 500:
            content = content[:497] + "..."
        
        print(f"{i:2d}. [{role.upper()}]: {content}")
        print()

def process_dataset(dataset_path: str, verbose: bool = False) -> List[tuple[str, List[Dict[str, Any]], str]]:
    """
    Process a single dataset file and return invalid conversations.
    
    Returns list of tuples: (conversation_id, conversation, error_message)
    """
    invalid_conversations = []
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset file not found: {dataset_path}")
        return invalid_conversations
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    conversation_id = data.get("ids", f"unknown_line_{line_num}")
                    
                    # Parse the inputs field (which is a JSON string)
                    inputs_str = data.get("inputs", "[]")
                    if isinstance(inputs_str, str):
                        conversation = json.loads(inputs_str)
                    else:
                        conversation = inputs_str
                    
                    # Validate conversation turns
                    is_valid, error_msg = validate_conversation_turns(conversation)
                    if not is_valid:
                        invalid_conversations.append((str(conversation_id), conversation, error_msg))
                        if verbose:
                            print_conversation(conversation, str(conversation_id), error_msg)
                        
                except json.JSONDecodeError as e:
                    error_msg = f"JSON parsing error on line {line_num}: {e}"
                    print(f"Error parsing JSON on line {line_num} in {dataset_path}: {e}")
                    invalid_conversations.append((f"parse_error_line_{line_num}", [], error_msg))
                except Exception as e:
                    error_msg = f"Processing error on line {line_num}: {e}"
                    print(f"Error processing line {line_num} in {dataset_path}: {e}")
                    invalid_conversations.append((f"error_line_{line_num}", [], error_msg))
    
    except Exception as e:
        print(f"Error reading file {dataset_path}: {e}")
    
    return invalid_conversations

def main():
    """Main function to process all datasets and report invalid conversation IDs."""
    parser = argparse.ArgumentParser(description='Validate conversation turns in evaluation datasets')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Output entire conversation for invalid cases')
    # Get default config path relative to script location
    script_dir = Path(__file__).parent
    default_config = str(script_dir.parent / "config" / "eval_datasets" / "test_balanced.yaml")
    
    parser.add_argument('config_file', nargs='?', 
                        default=default_config,
                        help='Path to the configuration file (default: config/eval_datasets/test_balanced.yaml)')
    args = parser.parse_args()
    
    config_path = args.config_file
    
    print("Loading configuration...")
    datasets = load_config(config_path)
    
    print(f"Found {len(datasets)} active datasets:")
    for name, path in datasets.items():
        print(f"  - {name}: {path}")
    
    print("\nValidating conversation turns...\n")
    
    all_invalid_conversations = []
    
    for dataset_name, dataset_path in datasets.items():
        print(f"Processing {dataset_name}...")
        invalid_conversations = process_dataset(dataset_path, verbose=args.verbose)
        
        if invalid_conversations:
            print(f"  Found {len(invalid_conversations)} invalid conversations in {dataset_name}")
            if not args.verbose:
                for conv_id, _, error_msg in invalid_conversations:
                    print(f"    Invalid ID: {conv_id}")
            for conv_id, conversation, error_msg in invalid_conversations:
                all_invalid_conversations.append((dataset_name, conv_id, conversation, error_msg))
        else:
            print(f"  All conversations in {dataset_name} have valid turns")
        
        print()
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_invalid_conversations:
        print(f"Total invalid conversations found: {len(all_invalid_conversations)}")
        if not args.verbose:
            print("\nAll invalid IDs:")
            for dataset_name, conv_id, _, _ in all_invalid_conversations:
                print(f"  {dataset_name}:{conv_id}")
    else:
        print("All conversations have valid turns!")
    
    print(f"\nValidation complete. Processed {len(datasets)} datasets.")
    
    # Return data for programmatic use
    return all_invalid_conversations

if __name__ == "__main__":
    main() 