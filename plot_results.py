#!/usr/bin/env python3
"""
Plot reproduction and ablation from *_eval_scores.npy in results folder.
Saves reproduction_plot.png (no_aug vs crop) and ablation_plot.png (no_aug, crop, flip).
Run from repo root: python plot_results.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results/cartpole-swingup-02-15-im84-b128-s42-pixel"
KEY_PREFIX = "cartpole-swingup"

def load_curve(data_augs):
    path = os.path.join(RESULTS_DIR, f"cartpole--swingup-{data_augs}--s42--eval_scores.npy")
    if not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True).item()
    key = f"{KEY_PREFIX}-{data_augs}"
    if key not in data:
        return None, None
    steps = sorted(data[key].keys())
    means = [data[key][s]["mean_ep_reward"] for s in steps]
    return np.array(steps), np.array(means)

def main():
    no_aug_steps, no_aug_means = load_curve("no_aug")
    crop_steps, crop_means = load_curve("crop")
    flip_steps, flip_means = load_curve("flip")

    if no_aug_steps is None:
        print("Missing no_aug .npy - run the three experiments first.")
        return

    # --- Figure 1: Reproduction (no_aug vs crop) ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(no_aug_steps, no_aug_means, label="no_aug (baseline)", color="C0")
    if crop_steps is not None:
        ax.plot(crop_steps, crop_means, label="crop (RAD)", color="C1")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval return (mean)")
    ax.set_title("Reproduction: Baseline vs RAD (crop)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("reproduction_plot.png", dpi=150)
    plt.close()
    print("Saved reproduction_plot.png")

    # --- Figure 2: Ablation (no_aug, crop, flip) ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(no_aug_steps, no_aug_means, label="no_aug", color="C0")
    if crop_steps is not None:
        ax.plot(crop_steps, crop_means, label="crop", color="C1")
    if flip_steps is not None:
        ax.plot(flip_steps, flip_means, label="flip", color="C2")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval return (mean)")
    ax.set_title("Ablation: no_aug vs crop vs flip")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("ablation_plot.png", dpi=150)
    plt.close()
    print("Saved ablation_plot.png")

if __name__ == "__main__":
    main()
