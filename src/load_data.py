from datasets import load_dataset
import json
from pathlib import Path

def download_hotpot_sample(n=100):
    """Download small sample of HotpotQA for development."""
    print(f"Downloading HotpotQA (this takes ~2 min)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    
    sample = ds.select(range(n))
    output_path = Path("data/hotpot_sample.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump([dict(x) for x in sample], f, indent=2)
    
    print(f"Saved {n} samples to {output_path}")
    
    # Print one example
    example = sample[0]
    print(f"\n--- Example ---")
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answer']}")
    print(f"Number of context paragraphs: {len(example['context']['title'])}")

if __name__ == "__main__":
    download_hotpot_sample(n=100)