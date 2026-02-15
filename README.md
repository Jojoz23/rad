# Reinforcement Learning with Augmented Data (RAD)

Official codebase for [Reinforcement Learning with Augmented Data](https://mishalaskin.github.io/rad). This codebase was originally forked from [CURL](https://mishalaskin.github.io/curl). 

Additionally, here is the [codebase link for ProcGen experiments](https://github.com/pokaxpoka/rad_procgen) and [codebase link for OpenAI Gym experiments](https://github.com/pokaxpoka/rad_openaigym).


## BibTex

```
@article{laskin2020reinforcement,
  title={Reinforcement learning with augmented data},
  author={Laskin, Michael and Lee, Kimin and Stooke, Adam and Pinto, Lerrel and Abbeel, Pieter and Srinivas, Aravind},
  journal={arXiv preprint arXiv:2004.14990},
  year={2020}
}
```

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Instructions
To train a RAD agent on the `cartpole swingup` task from image-based observations run `bash script/run.sh` from the root of this directory. The `run.sh` file contains the following command, which you can modify to try different environments / augmentations / hyperparamters.

```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel --work_dir ./tmp/cartpole \
    --action_repeat 8 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 84 \
    --agent rad_sac --frame_stack 3 --data_augs flip  \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 200000 &
```

## Data Augmentations 

Augmentations can be specified through the `--data_augs` flag. This codebase supports the augmentations specified in `data_augs.py`. To chain multiple data augmentation simply separate the augmentation strings with a `-` string. For example to apply `crop -> rotate -> flip` you can do the following `--data_augs crop-rotate-flip`. 

All data augmentations can be visualized in `All_Data_Augs.ipynb`. You can also test the efficiency of our modules by running `python data_aug.py`.


## Logging 

In your console, you should see printouts that look like this:

```
| train | E: 13 | S: 2000 | D: 9.1 s | R: 48.3056 | BR: 0.8279 | A_LOSS: -3.6559 | CR_LOSS: 2.7563
| train | E: 17 | S: 2500 | D: 9.1 s | R: 146.5945 | BR: 0.9066 | A_LOSS: -5.8576 | CR_LOSS: 6.0176
| train | E: 21 | S: 3000 | D: 7.7 s | R: 138.7537 | BR: 1.0354 | A_LOSS: -7.8795 | CR_LOSS: 7.3928
| train | E: 25 | S: 3500 | D: 9.0 s | R: 181.5103 | BR: 1.0764 | A_LOSS: -10.9712 | CR_LOSS: 8.8753
| train | E: 29 | S: 4000 | D: 8.9 s | R: 240.6485 | BR: 1.2042 | A_LOSS: -13.8537 | CR_LOSS: 9.4001
```
The above output decodes as:

```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CR_LOSS - average loss of critic
```

All data related to the run is stored in the specified `working_dir`. To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`. To visualize progress with tensorboard run:

```
tensorboard --logdir log --port 6006
```

and go to `localhost:6006` in your browser. If you're running headlessly, try port forwarding with ssh.

---

## Assignment 1: Running the code (CSC415)

This section gives step-by-step setup so the code runs on lab machines (e.g. no conda, headless, disk quota). The written report (implementation details, ablation description, results analysis) is in the main assignment document; this repo contains the code, run commands, plots, and results. The repository includes compatibility fixes for headless rendering and gym 0.26; see **Troubleshooting** below if you hit further errors.

### Option A: Conda (if available)

```bash
git clone https://github.com/Jojoz23/rad.git
cd rad
conda env create -f conda_env.yml
conda activate rad
# Then run the commands under "Run experiments" below.
```

### Option B: uv + virtual environment (recommended on lab machines without conda)

1. **Install uv** (no sudo):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

2. **Create venv** (use `/tmp` if you hit disk quota in home):
   ```bash
   cd rad
   uv venv /tmp/rad-venv
   source /tmp/rad-venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   export UV_LINK_MODE=copy
   uv pip install --python /tmp/rad-venv/bin/python torch torchvision numpy gym termcolor imageio imageio-ffmpeg scikit-image tabulate
   uv pip install --python /tmp/rad-venv/bin/python dm_control
   uv pip install --python /tmp/rad-venv/bin/python "gym<0.27"
   uv pip install --python /tmp/rad-venv/bin/python git+https://github.com/1nadequacy/dmc2gym.git
   uv pip install --python /tmp/rad-venv/bin/python tensorboard matplotlib
   ```

4. **Apply compatibility patches (required with gym 0.26 + NumPy 2).** After the install above, the following errors will occur unless you patch the **installed** packages. Set `VENV=/tmp/rad-venv` (or your venv path), then apply the edits below.

   - **dmc2gym** – In `$VENV/lib/python3.10/site-packages/dmc2gym/__init__.py`: change `if not env_id in gym.envs.registry.env_specs:` to `if env_id not in gym.envs.registry:`.
   - **dmc2gym** – In `$VENV/lib/python3.10/site-packages/dmc2gym/wrappers.py`: change `dim = np.int(np.prod(s.shape))` to `dim = int(np.prod(s.shape))`. In the same file, change the `step` return from `return obs, reward, done, extra` to `return obs, reward, done, False, extra` (i.e. add `terminated, truncated` so the return is 5 values: `obs, reward, terminated, truncated, extra` with `terminated=done`, `truncated=False`).
   - **gym** – In `$VENV/lib/python3.10/site-packages/gym/utils/passive_env_checker.py`: replace all three occurrences of `(bool, np.bool8)` with `bool` (so the three `isinstance` checks use `bool` only).

   After these patches, `python train.py --domain_name cartpole --task_name swingup --encoder_type pixel --work_dir ./results --num_train_steps 100` should run and print an eval line.

5. **Headless:** The code sets `MUJOCO_GL=egl` automatically for headless rendering. If you still see OpenGL errors, run: `export MUJOCO_GL=egl` (or `MUJOCO_GL=osmesa`) before `python train.py`.

### Run experiments

From the repo root with your environment activated (e.g. `source /tmp/rad-venv/bin/activate`):

**1. Baseline (no augmentation)**  
```bash
python train.py --domain_name cartpole --task_name swingup --encoder_type pixel --work_dir ./results --action_repeat 8 --num_eval_episodes 5 --pre_transform_image_size 100 --image_size 84 --agent rad_sac --frame_stack 3 --data_augs no_aug --seed 42 --eval_freq 5000 --batch_size 128 --num_train_steps 100000
```

**2. RAD with crop (main result)**  
```bash
python train.py --domain_name cartpole --task_name swingup --encoder_type pixel --work_dir ./results --action_repeat 8 --num_eval_episodes 5 --pre_transform_image_size 100 --image_size 84 --agent rad_sac --frame_stack 3 --data_augs crop --seed 42 --eval_freq 5000 --batch_size 128 --num_train_steps 100000
```

**3. Ablation: flip**  
```bash
python train.py --domain_name cartpole --task_name swingup --encoder_type pixel --work_dir ./results --action_repeat 8 --num_eval_episodes 5 --pre_transform_image_size 100 --image_size 84 --agent rad_sac --frame_stack 3 --data_augs flip --seed 42 --eval_freq 5000 --batch_size 128 --num_train_steps 100000
```

Eval scores are saved under `./results` as `.npy` files for plotting.

**Plotting results (after all three runs):** From the repo root with your env activated, run:
```bash
python plot_results.py
```
This reads the `*_eval_scores.npy` files and saves `reproduction_plot.png` (no_aug vs crop) and `ablation_plot.png` (no_aug, crop, flip) in the repo root. The script looks for results in `results/cartpole-swingup-02-15-im84-b128-s42-pixel` (the committed run). If you ran the experiments yourself, the output folder name will include the current date—either copy your `*_eval_scores.npy` files into that folder or set `RESULTS_DIR` in `plot_results.py` to your results subfolder.

### Troubleshooting

- **Disk quota exceeded:** Create the venv in `/tmp`: `uv venv /tmp/rad-venv`. Note: `/tmp` may be cleared on reboot; re-run the install steps if needed.
- **`gym.envs.registry.env_specs` / `np.int` / `np.bool8` / "expected 5, got 4":** These come from gym 0.26 and NumPy 2. Follow **Option B step 4** above to patch the installed dmc2gym and gym packages. This repo’s `train.py` and `utils.py` already include headless and FrameStack fixes.
- **No module named 'tensorboard' or 'matplotlib':** `uv pip install tensorboard matplotlib` (or `pip install tensorboard matplotlib`).

