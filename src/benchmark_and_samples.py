import os
import sys
import time
import json
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import ToxicityPredictor
from src.config import OUTPUTS_DIR

def run_benchmarks_and_samples():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading trained ToxicityPredictor for latency benchmarking...")
    predictor = ToxicityPredictor()

    sample_prompts = [
        "People should respect each other's beliefs and treat each other kindly.",
        "You are an absolute idiot and a worthless loser, get lost!",
        "The word 'idiot' is considered offensive in formal debates, though people use it casually."
    ]

    print("Generating exact outputs for the 3 canonical test comments...")
    sample_results = []
    for prompt in sample_prompts:
        res = predictor.predict(prompt)
        sample_results.append(res)
        print(f"\n[Sample Input]: {prompt}")
        print(f"  Prediction: {res['prediction']} | Severity: {res['severity']} | Action: {res['moderation_action']}")
        print(f"  Detected:   {res['detected_labels']}")
        print(f"  Reason:     {res['reason']}")
        print(f"  Scores:     {res['probabilities']}")

    sample_path = OUTPUTS_DIR / "sample_api_results.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_results, f, indent=2)
    print(f"\nSaved exact sample API results to {sample_path}")

    # Latency benchmark: 50 requests
    print("\nBenchmarking inference latency over 50 individual requests...")
    # Warmup
    for _ in range(5):
        predictor.predict("Warmup test input.")

    latencies_ms = []
    benchmark_texts = [
        "This is a wonderful contribution to the project, thank you!",
        "Stop being such a moron, nobody asked for your garbage opinion.",
        "I disagree with this approach because the data shows otherwise.",
        "Shut up you fool, delete your account right now.",
        "Could you please explain what this function does in detail?"
    ]

    for i in range(50):
        text = benchmark_texts[i % len(benchmark_texts)]
        t0 = time.perf_counter()
        _ = predictor.predict(text)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(latencies_ms))
    p50_latency = float(np.percentile(latencies_ms, 50))
    p90_latency = float(np.percentile(latencies_ms, 90))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))
    min_latency = float(np.min(latencies_ms))
    max_latency = float(np.max(latencies_ms))

    latency_data = {
        "device": str(predictor.device),
        "device_name": predictor.meta.get("compute_tier", str(predictor.device)),
        "num_requests": len(latencies_ms),
        "mean_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p90_latency_ms": round(p90_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "p99_latency_ms": round(p99_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "batch_size": 1
    }

    latency_path = OUTPUTS_DIR / "latency.json"
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_data, f, indent=2)
    print(f"Saved latency statistics to {latency_path}")
    print(f"Latency Results (50 requests): Mean={avg_latency:.2f} ms | p95={p95_latency:.2f} ms | p50={p50_latency:.2f} ms")

if __name__ == "__main__":
    run_benchmarks_and_samples()
