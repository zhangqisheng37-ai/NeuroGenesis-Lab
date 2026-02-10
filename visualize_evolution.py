import subprocess
import sys
import glob
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

def run_experiment():
    print("Running evolution experiment...")
    env = os.environ.copy()
    # Add src to PYTHONPATH
    src_path = os.path.abspath("src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path
        
    subprocess.check_call([sys.executable, "examples/experiment_evolution.py"], env=env)

def find_latest_output():
    # Looking for outputs/evolution_*
    search_path = os.path.join("outputs", "evolution_*")
    dirs = glob.glob(search_path)
    if not dirs:
        raise FileNotFoundError("No evolution output directories found.")
    # Sort by modification time
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir

def visualize(output_dir):
    jsonl_path = os.path.join(output_dir, "run.jsonl")
    print(f"Analyzing {jsonl_path}...")
    
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    if "metrics" in record:
                        data.append(record["metrics"])
                    elif record.get("type") == "tick": # Fallback for older format if any
                         data.append(record["data"])
                except json.JSONDecodeError:
                    continue
    
    if not data:
        print("No tick data found.")
        return

    ticks = [d["tick"] for d in data]
    
    # 1. Population Dynamics
    alive = [d["alive_neurons"] for d in data]
    total = [d["total_neurons"] for d in data]
    corpses = [d["corpse_count"] for d in data]
    
    # 2. Energy & Emotion
    energy = [d["avg_energy"] for d in data]
    emotion = [d.get("avg_emotion", 0) for d in data]
    
    # 3. Connectivity
    conns = [d.get("avg_connections", 0) for d in data]

    # Plotting
    plt.style.use('ggplot')
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot 1: Population
    ax0 = axes[0]
    ax0.plot(ticks, total, label="Total Neurons", color="gray", linestyle="--")
    ax0.plot(ticks, alive, label="Alive Neurons", color="green", linewidth=2)
    ax0.plot(ticks, corpses, label="Corpses", color="black", linestyle=":")
    ax0.set_title("Population Dynamics (Digital Life)")
    ax0.set_ylabel("Count")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    
    # Plot 2: Internal State (Energy & Emotion)
    ax1 = axes[1]
    ax1.plot(ticks, energy, label="Avg Energy", color="orange")
    ax1.set_ylabel("Energy Level")
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(ticks, emotion, label="Avg Emotion", color="purple", alpha=0.6)
    ax1_twin.set_ylabel("Emotion State")
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Metabolism & State")
    ax1.grid(True, alpha=0.3)

    # Plot 3: Network Structure
    ax2 = axes[2]
    ax2.plot(ticks, conns, label="Avg Connections/Neuron", color="blue")
    ax2.set_title("Network Topology Evolution")
    ax2.set_xlabel("Time (Ticks)")
    ax2.set_ylabel("Connections")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    output_image = "evolution_report.png"
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Visualization saved to {os.path.abspath(output_image)}")

if __name__ == "__main__":
    run_experiment()
    latest_dir = find_latest_output()
    visualize(latest_dir)
