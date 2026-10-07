"""
Privacy Mitigations & Defenses Study Runner.
Benchmarks Differential Privacy noise addition and sampling rate throttling
to analyze the privacy vs. utility tradeoff for consumer IMU telemetry.
"""

import argparse
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import load_or_generate_dataset
from src.defense import evaluate_mitigation_tradeoff
from src.evaluation import plot_defense_benchmark


def run_mitigation_study(
    num_users: int = 15,
    sessions_per_user: int = 4,
    duration_sec: float = 20.0,
    output_dir: str = "reports",
):
    print("=" * 70)
    print("  SENSOR-BASED AUTHENTICATION RISK STUDY: PRIVACY MITIGATION BENCHMARK")
    print("=" * 70)
    
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    data_cache = os.path.join("data", "processed", f"imu_population_{num_users}u.parquet")
    df = load_or_generate_dataset(
        cache_path=data_cache,
        num_users=num_users,
        sessions_per_user=sessions_per_user,
        duration_sec=duration_sec,
    )
    
    print("[+] Evaluating Differential Privacy & Frequency Throttling countermeasures...")
    results_df = evaluate_mitigation_tradeoff(
        raw_df=df,
        noise_epsilons=[0.1, 0.5, 1.0, 5.0],
        throttling_rates=[10.0, 20.0, 50.0, 100.0],
    )
    
    csv_path = os.path.join(output_dir, "mitigation_benchmark.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved mitigation results to: {csv_path}")
    print("\nMitigation Benchmark Summary:")
    print(results_df.to_string(index=False))
    
    fig_path = os.path.join(output_dir, "figures", "defense_tradeoff.png")
    plot_defense_benchmark(results_df, fig_path)
    print(f"\n[+] Generated mitigation tradeoff plot: {fig_path}")
    print("\n[SUCCESS] Mitigation study completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Privacy Mitigation Study")
    parser.add_argument("--users", type=int, default=15)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    run_mitigation_study(
        num_users=args.users,
        sessions_per_user=args.sessions,
        duration_sec=args.duration,
    )
