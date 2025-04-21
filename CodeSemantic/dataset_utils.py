import random
from typing import Tuple, List, Dict

def select_shots_and_split_dataset(
    dataset: List[Dict], 
    args, 
    shot_key: str = 'Source Code',
    min_length_key: str = 'Source Code'
) -> Tuple[List[Dict], List[Dict]]:
    """
    Select shot examples from the dataset and return both the shots and the remaining dataset.
    
    Args:
        dataset: The original dataset (list of dictionaries)
        args: Command line arguments containing shot, incontext, CoT etc.
        shot_key: The key to use for selecting shots (default 'Source Code')
        min_length_key: The key to use for finding minimum length examples (default 'Source Code')
        
    Returns:
        Tuple of (shots, remaining_dataset)
    """
    if args.shot == 0:
        return [], dataset
    
    if args.prediction == 'loop':
        min_length_key = 'loop_code'
        shot_key = 'loop_code'
    elif args.prediction == 'alias':
        min_length_key = 'Source Code'
        shot_key = 'Source Code'
    elif args.prediction == 'statement':
        min_length_key = 'Source Code'
        shot_key = 'Source Code'
    
    valid_examples = [d for d in dataset if min_length_key in d]
    
    if not valid_examples:
        raise ValueError(f"No examples found with key '{min_length_key}' in dataset")
    
    sorted_examples = sorted(
        valid_examples,
        key=lambda x: len(x[min_length_key])
    )
    
    selected_shots = sorted_examples[:args.shot]
    
    remaining_dataset = [
        d for d in dataset 
        if d not in selected_shots
    ]
    
    return selected_shots, remaining_dataset


def incontext_shots_with_same_statement(args, query):
    """Returns smallest in-context examples that:
    - Don't match the query by index/source
    - Match the query's statement type
    - Are sorted by length (shortest first)
    """
    if args.shot == 0:
        return []
    
    if args.prediction == 'loop':
        min_length_key = 'loop_code'
    else:  
        min_length_key = 'Source Code'
    
    query_stmt_type = query.get('Statement Type') if args.prediction == 'statement' else None
    

    valid_examples = [
        d for d in args.dataset 
        if min_length_key in d
        and d != query  
        and (not query_stmt_type or d.get('Statement Type') == query_stmt_type)
    ]
    
    if not valid_examples:
        raise ValueError(
            f"No matching in-context examples found (type: {query_stmt_type}, " 
            f"min_length_key: {min_length_key})"
        )
    
    sorted_examples = sorted(
        valid_examples,
        key=lambda x: len(x[min_length_key])
    )
    
    return sorted_examples[:args.shot]
    