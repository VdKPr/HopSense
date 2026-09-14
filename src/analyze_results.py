"""
Analyze batch results: where does HopSense beat Naive? Where does it lose?
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from collections import Counter


def main(path: str = "results/batch_results.json"):
    with open(path) as f:
        results = json.load(f)
    
    n = len(results)
    
    hop_wins = []      # HopSense correct, Naive wrong
    naive_wins = []    # Naive correct, HopSense wrong
    both_right = []
    both_wrong = []
    
    for r in results:
        n_ok = r["naive"]["contains_gold"]
        h_ok = r["hopsense"]["contains_gold"]
        if h_ok and not n_ok: hop_wins.append(r)
        elif n_ok and not h_ok: naive_wins.append(r)
        elif h_ok and n_ok: both_right.append(r)
        else: both_wrong.append(r)
    
    print(f"{'='*70}")
    print(f"Analysis of {n} questions")
    print(f"{'='*70}")
    print(f"Both correct:      {len(both_right):3d}  ({100*len(both_right)/n:.0f}%)")
    print(f"HopSense only:     {len(hop_wins):3d}  ({100*len(hop_wins)/n:.0f}%)  ← HopSense wins")
    print(f"Naive only:        {len(naive_wins):3d}  ({100*len(naive_wins)/n:.0f}%)  ← Naive wins")
    print(f"Both wrong:        {len(both_wrong):3d}  ({100*len(both_wrong)/n:.0f}%)")
    print()
    
    # By question type
    print(f"\n{'─'*70}")
    print("Contains-gold accuracy by HotpotQA question type")
    print(f"{'─'*70}")
    types = Counter(r.get("type", "unknown") for r in results)
    for t, count in types.most_common():
        n_acc = sum(1 for r in results if r.get("type") == t and r["naive"]["contains_gold"])
        h_acc = sum(1 for r in results if r.get("type") == t and r["hopsense"]["contains_gold"])
        print(f"  {t:20s}  n={count:3d}  naive={100*n_acc/count:.0f}%  hopsense={100*h_acc/count:.0f}%")
    
    # Confidence calibration on HopSense
    print(f"\n{'─'*70}")
    print("HopSense confidence calibration (is high conf = correct?)")
    print(f"{'─'*70}")
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in buckets:
        in_bucket = [r for r in results 
                     if lo <= r["hopsense"]["confidence"] < hi]
        if not in_bucket: continue
        correct = sum(1 for r in in_bucket if r["hopsense"]["contains_gold"])
        print(f"  conf [{lo:.2f}, {hi:.2f})  n={len(in_bucket):3d}  correct={correct:3d}  accuracy={100*correct/len(in_bucket):.0f}%")
    
    # Print sample failures
    print(f"\n{'─'*70}")
    print("Sample: Naive wins (naive right, HopSense wrong) — first 3")
    print(f"{'─'*70}")
    for r in naive_wins[:3]:
        print(f"\nQ: {r['question']}")
        print(f"Gold: {r['gold_answer']}")
        print(f"Naive: {r['naive']['answer']}  ✓")
        print(f"HopSense: {r['hopsense']['answer']}  ✗  (conf={r['hopsense']['confidence']:.2f})")
        print(f"Sub-questions HopSense used:")
        for sq in r["hopsense"]["sub_questions"]:
            print(f"    - {sq}")
    
    print(f"\n{'─'*70}")
    print("Sample: HopSense wins (HopSense right, Naive wrong) — first 3")
    print(f"{'─'*70}")
    for r in hop_wins[:3]:
        print(f"\nQ: {r['question']}")
        print(f"Gold: {r['gold_answer']}")
        print(f"Naive: {r['naive']['answer']}  ✗")
        print(f"HopSense: {r['hopsense']['answer']}  ✓  (conf={r['hopsense']['confidence']:.2f})")


if __name__ == "__main__":
    main()