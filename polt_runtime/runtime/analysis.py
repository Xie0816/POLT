"""Offline metric and diagnostic helpers for POLT experiment outputs."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _process_log_data(self, json_path):
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    if not log_data:
        return None

    stamps = sorted(int(ts) for ts in log_data.keys())
    gpr_costs = [log_data[str(ts)]["gpr_cost"] for ts in stamps]
    proprio_costs = [log_data[str(ts)]["proprio_cost"] for ts in stamps]
    errors = [log_data[str(ts)]["error"] for ts in stamps]
    node_nums = [log_data[str(ts)].get("nodes_num", 0) for ts in stamps]
    train_durations = [log_data[str(ts)].get("train_time", 0.0) for ts in stamps]
    cumulative_train_time = np.cumsum(train_durations).tolist()

    return {
        "stamps": stamps,
        "gpr_costs": gpr_costs,
        "proprio_costs": proprio_costs,
        "errors": errors,
        "node_nums": node_nums,
        "train_durations": train_durations,
        "cumulative_train_time": cumulative_train_time,
        "filename": os.path.basename(json_path),
    }


def visualize_mae_analysis(self, json_path):
    data = self._process_log_data(json_path)
    if not data:
        return

    rel_times = [(ts - data["stamps"][0]) / 1000.0 for ts in data["stamps"]]

    plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(rel_times, data["gpr_costs"], label="GPR (Vision)", color="royalblue", alpha=0.8)
    ax1.plot(rel_times, data["proprio_costs"], label="Proprio (GT)", color="forestgreen", linestyle="--", alpha=0.7)
    ax1.set_title(f'Method Analysis: {data["filename"]}', fontsize=14)
    ax1.set_ylabel("Cost", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = plt.subplot(2, 1, 2)
    ax2_right = ax2.twinx()
    ln1 = ax2.plot(rel_times, data["node_nums"], label="Node Count", color="blue", linewidth=2, alpha=0.6)
    ln2 = ax2_right.plot(
        rel_times,
        data["cumulative_train_time"],
        label="Cumul. Train Time (ms)",
        color="red",
        linestyle="-.",
        alpha=0.6,
    )

    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Num. of Nodes", color="blue", fontsize=12)
    ax2_right.set_ylabel("Total Train Time (ms)", color="red", fontsize=12)

    lines = ln1 + ln2
    ax2.legend(lines, [line.get_label() for line in lines], loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    save_path = json_path.replace(".json", "_detailed_analysis.png")
    plt.savefig(save_path, dpi=300)
    plt.show()


def visualize_comparison_mae_analysis(
    self,
    json_path1,
    json_path2,
    label1="Dynamic Memory",
    label2="Data Buffer",
):
    d1 = self._process_log_data(json_path1)
    d2 = self._process_log_data(json_path2)
    if not d1 or not d2:
        return

    global_start = min(d1["stamps"][0], d2["stamps"][0])
    t1 = [(ts - global_start) / 1000.0 for ts in d1["stamps"]]
    t2 = [(ts - global_start) / 1000.0 for ts in d2["stamps"]]

    plt.figure(figsize=(14, 12))

    ax1 = plt.subplot(2, 1, 1)
    ax1_err = ax1.twinx()
    ax1.plot(t1, d1["gpr_costs"], label=f"{label1} GPR", color="blue", alpha=0.7)
    ax1.plot(t2, d2["gpr_costs"], label=f"{label2} GPR", color="red", alpha=0.7)
    ax1.plot(t1, d1["proprio_costs"], color="gray", linestyle="--", alpha=0.3, label="Ground Truth")
    ax1_err.plot(t1, d1["errors"], label=f"{label1} Error", color="blue", linestyle=":", alpha=0.4)
    ax1_err.plot(t2, d2["errors"], label=f"{label2} Error", color="red", linestyle=":", alpha=0.4)

    ax1.set_title(f"Comparison: {label1} vs {label2}", fontsize=14)
    ax1.set_ylabel("Cost (Left)", fontsize=12)
    ax1_err.set_ylabel("Prediction Error (Right)", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1_err.set_ylim(0, 0.5)
    ax1.legend(loc="upper left", ncol=2)
    ax1_err.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)

    ax2 = plt.subplot(2, 1, 2)
    ax2_time = ax2.twinx()
    ax2.plot(t1, d1["node_nums"], label=f"{label1} Nodes", color="blue", linewidth=2, alpha=0.7)
    ax2.plot(t2, d2["node_nums"], label=f"{label2} Nodes", color="red", linewidth=2, alpha=0.7)
    ax2_time.plot(t1, d1["cumulative_train_time"], label=f"{label1} Cumul. Time", color="blue", linestyle="--", alpha=0.4)
    ax2_time.plot(t2, d2["cumulative_train_time"], label=f"{label2} Cumul. Time", color="red", linestyle="--", alpha=0.4)

    ax2.set_xlabel("Global Time (seconds)", fontsize=12)
    ax2.set_ylabel("Num. of Nodes (Left)", fontsize=12)
    ax2_time.set_ylabel("Total Training Time (ms, Right)", fontsize=12)
    ax2.legend(loc="upper left")
    ax2_time.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(json_path1), f"comparison_{label1}_vs_{label2}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    metrics_summary = {}
    print("\n" + "=" * 50)
    print(f"{'Performance Metrics Comparison':^50}")
    print("=" * 50)

    for data, label in zip([d1, d2], [label1, label2]):
        mae = float(np.mean(data["errors"]))
        final_nodes = int(data["node_nums"][-1])
        total_time = float(data["cumulative_train_time"][-1])
        avg_time = float(total_time / len(data["stamps"]))
        metrics_summary[label] = {
            "mae": mae,
            "final_nodes": final_nodes,
            "total_train_time_ms": total_time,
            "avg_train_per_frame_ms": avg_time,
        }
        print(f"[{label}]")
        print(f"  > MAE:                  {mae:.6f}")
        print(f"  > Final Nodes:          {final_nodes}")
        print(f"  > Total Training Time:  {total_time:.2f} ms")
        print(f"  > Avg Time Per Frame:   {avg_time:.2f} ms")
        print("-" * 50)

    metrics_json_path = os.path.join(os.path.dirname(json_path1), "metrics_comparison.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"定量指标已保存至: {metrics_json_path}")
