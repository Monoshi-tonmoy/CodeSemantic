import json

def print_dataset_fields(file_path):
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            try:
                data = json.loads(line.strip())
                
                print(f"=== Entry {line_num} ===")
                print("LOOP CODE:")
                print(data.get('loop_code', 'N/A'))
                print("\nQUESTION:")
                print(data.get('question', 'N/A'))
                print("\nANSWER:")
                print(data.get('answer', 'N/A'))
                print("\n" + "="*50 + "\n")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

# Replace with your actual file path
file_path = "/home/monoshi/CodeSemantic/CodeSemantic/dataset/loop_iteration_dataset_python.jsonl"
print_dataset_fields(file_path)