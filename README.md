# 🧠 Options In Simulation
**Learning-Driven Torque Control for Skid-Steer Robots**

A reinforcement learning framework for **learning torque-level control** of skid-steer robots in simulation, designed with **reproducibility**, **scalability**, and **deployment awareness** in mind.  
Built on **NVIDIA Isaac Sim**, **OmniIsaacGymEnvs**, and **skrl**, it demonstrates a full ML engineering workflow — from training and evaluation to containerized execution on HPC and cloud systems.

📘 Full thesis: [TU Delft Repository](https://repository.tudelft.nl/record/uuid:0bcc777f-cf1c-49fe-8dbf-c18237864841)

---

## 🚀 Highlights

- **Algorithms:** Knowledge-Assisted DDPG (KA-DDPG) and KAMMA with curriculum learning  
- **Environment:** NVIDIA Isaac Sim 2022.2.1 + OmniIsaacGymEnvs + skrl  
- **Evaluation Metric:** **Tracking Error (TE)** — deviation between commanded and actual velocity  
- **Infrastructure:**  
  - ✅ **Apptainer (HPC)** – fully reproducible Isaac Sim container  
  - ☁️ **AWS EC2** – Docker setup for visual simulation and monitoring  
  - ⚙️ **SLURM scripts** – batch execution of large-scale experiments  
- **CI:** Automated CLI testing for reproducibility

---

## 🧩 Command-Line Interface

| Command | Purpose | Example |
|----------|----------|---------|
| **`options-sim-train`** | Launch training for a given configuration. Auto-resolves the correct legacy script based on the chosen parameters. | ```/isaac-sim/python.sh -m options_sim.cli.train   --algorithm kaddpg --action-dim 1d   --fifo nofifo --curriculum random   --learning-strategy rlil --root . -- --seed 80``` *All experiments are deterministic for a given seed (set internally in the legacy script). Use consistent seeds to reproduce results across runs.* |
| **`options-sim-eval`** | Evaluate a trained model checkpoint using specific training and evaluation seeds. Seeds are crucial for reproducibility and naming (`_s{seed}` in folder structure). | ```/isaac-sim/python.sh -m options_sim.cli.eval   --algorithm kamma --action-dim 4d   --fifo nofifo --curriculum random   --strategy RLIL --train-seed 1 --seed 777   --checkpoint-step 500000 --root /workspace/Options_In_Simulation``` *The `--train-seed` identifies the correct trained model directory, while `--seed` controls randomness during evaluation (e.g., initial state sampling). Each combination creates a uniquely named folder under `my_runs/` for traceability.* |
| **`options-sim-artifacts`** | Aggregate and visualize training rewards from TensorBoard logs, producing a CSV and PNG of six key reward components for diagnostics. | ```/isaac-sim/python.sh -m options_sim.cli.artifacts   --run kamma_4d_nofifo_random_RLIL_s1 --mirror-to-artifacts``` *Generates time-series CSV and plot files (`reward_components_env0_timeseries.csv`, `reward_components_env0.png`) in the run directory and mirrors them under `artifacts/`.* |
| **`options-sim-eval-artifacts`** | Generate plots of **Tracking Error vs Speed** from evaluation runs. Uses TensorBoard logs or synthetic speed ramps when desired velocity isn’t logged. | ```/isaac-sim/python.sh -m options_sim.cli.eval_artifacts   --run eval_kaddpg_1d_fifo_random_RLIL_s42_a500000_s42   --smooth 10 --mirror-to-artifacts``` *Produces `tracking_error_vs_speed.csv` and `tracking_error_vs_speed.png` showing mean tracking deviation over commanded speed. `--seed` from evaluation ensures deterministic log naming and matching metrics.* |

---

## 🧱 Container & HPC Integration

### 🔒 Apptainer Environment

For interactive environments:

`apptainer/isaac-sim-custom.def` builds a GPU-enabled container with:
- Bootstrap of Isaac Sim 2022.2.1 base image  
- OmniIsaacGymEnvs & skrl preinstalled  


```bash
apptainer build isaac-sim-custom.sif isaac-sim-custom.def
git clone https://github.com/meesjansen/Options_In_Simulation.git 
```

Create container and link to created host directories:

```bash
apptainer shell --nv \
-C \
--env "ACCEPT_EULA=Y" \
--env "PRIVACY_CONSENT=Y" \
-B /path/to/Options_In_Simulation:/workspace/Options_In_Simulation \
-B /path/to/isaac-sim/cache/kit:/isaac-sim/kit/cache \
-B /path/to/isaac-sim/cache/ov:/root/.cache/ov \
-B /path/to/isaac-sim/cache/pip:/root/.cache/pip \
-B /path/to/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
-B /path/to/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
-B /path/to/isaac-sim/logs:/root/.nvidia-omniverse/logs \
-B /path/to/isaac-sim/data:/root/.local/share/ov/data \
-B /path/to/isaac-sim/documents:/root/Documents \
/path/to/isaac-sim-custom.sif

cd /workspace/Options_In_Simulation
```

Inside the container, install dependencies (if not already baked in):
```bash
/isaac-sim/python.sh -m pip install --upgrade pip setuptools wheel

/isaac-sim/python.sh -m pip install . --no-build-isolation --no-cache-dir --use-feature=in-tree-build
```

Alternatively, you can simply extend the Python path instead of installing:
```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Finally start a Workflow with:
```bash
/isaac-sim/python.sh -m options_sim.cli.train \
  --algorithm kaddpg --action-dim 1d --fifo nofifo --curriculum random --learning-strategy rlil \
  --root /workspace/Options_In_Simulation -- \
  --seed 1
```

> 🧩 **Tip:**  
> For cloud or on-premise clusters without Apptainer, the same workflow can be executed using the "Running Isaac Sim Container" section from 
> [Isaac Sim AWS Docker setup](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_advanced_cloud_setup_aws.html).
> Use the .def file as a blueprint to manually install the right packages and remember to use ```/isaac-sim/python.sh``` to launch python scripts. Both methods yield identical results when the same seeds and configuration are used. Make sure to pull the correct image from below and create the correct cached volume mounts on the host:
```bash
docker pull nvcr.io/nvidia/isaac-sim:4.0.0

docker run --name isaac-sim-oige --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
-e "PRIVACY_CONSENT=Y" \
-v ${PWD}:/workspace/omniisaacgymenvs \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
nvcr.io/nvidia/isaac-sim:4.0.0
```
---

## ⚙️ HPC & SLURM Integration

The project includes **SLURM batch scripts** under `apptainer/` that automate training and evaluation on HPC clusters.  
These scripts ensure large-scale, reproducible experiments while managing GPU resources efficiently.

### 🎛️ SLURM Scripts Overview

Each script:
- Requests the appropriate number of **GPU nodes and CPUs**  
- Loads the **Apptainer** module and activates the Isaac Sim container  
- Executes the training or evaluation command using the project’s CLI tools  
- Stores logs, checkpoints, and metrics under `my_runs/`, with results mirrored to `artifacts/` for post-analysis  

This design allows running **parallel experiments** (e.g., 10 seeds × 2 algorithms) reproducibly across a compute cluster.

---

## 🔁 Experiment Workflow & Evaluation

The project’s workflow mirrors a **full ML lifecycle**, from training and evaluation to artifact generation and performance tracking.  
All stages are executed through CLI commands for reproducibility and automation.

### 🧠 End-to-End Process

1. **Training**  
   Run experiments with `options-sim-train`, which automatically selects and executes the correct legacy training script (e.g., `train_kamma_4d_fifo_random_rlil.py`).  
   Training logs, checkpoints, and TensorBoard data are stored under `my_runs/<run_name>_s<seed>/`.

2. **Evaluation**  
   Use `options-sim-eval` to test trained policies on unseen conditions.  
   Each evaluation creates a new directory under `my_runs/eval_<run_name>_a<checkpointstep>_s<seed>/` containing metrics such as episode rewards and **Tracking Error** time series.

3. **Artifact Generation**  
   - `options-sim-artifacts` aggregates and visualizes reward components over training.  
   - `options-sim-eval-artifacts` generates **Tracking Error vs Speed** plots from evaluation logs, producing `.csv` and `.png` outputs under `artifacts/<run_name>/`.


### 📈 Tracking Error (TE) as Core Metric

**Tracking Error (TE)** — the deviation between target and actual velocity — is the project’s central measure of control accuracy.  
TE is computed continuously during evaluation, representing the robot’s ability to follow desired velocity profiles across terrains.

| Metric | Description | Source |
|---------|-------------|--------|
| **Tracking Error (TE)** | Average absolute difference between commanded and measured velocity | Computed from TensorBoard logs via `eval_artifacts.py` |
| **Velocity Correlation** | TE is evaluated along a velocity ramp | Derived from TE and reward logs |
| **Reward Components** | Six sub-terms contributing to the total reward during training | Extracted via `artifacts.py` |

Visualizing these metrics across experiments helps identify:
- Algorithmic stability across seeds  
- Policy smoothness and performance under curriculum progression  
- Generalization to unseen conditions (e.g., new slopes, surface friction)


### 🧩 Outputs Created

```text
my_runs/
├── kamma_4d_nofifo_random_RLIL_s1/
│   ├── checkpoints/agent_500000.pt
│   ├── events.out.tfevents.*  (TensorBoard logs)
│   ├── reward_components_env0_timeseries.csv
│   └── reward_components_env0.png
│   └── ...
├── eval_kamma_4d_nofifo_random_RLIL_s1_a500000_s777/
│   ├── metrics_tracking_error.csv
│   └── eval_summary.json
artifacts/
├── kamma_4d_nofifo_random_RLIL_s1/
│   ├── reward_components_env0.png
│   └── reward_components_env0_timeseries.csv
└── eval_kamma_4d_nofifo_random_RLIL_s1_a500000_s777/
    ├── tracking_error_vs_speed.png
    └── tracking_error_vs_speed.csv
```

Together, these logs and plots capture the full experiment lifecycle, allowing rapid comparison across configurations or algorithmic variants.

> 🧩 **Tip:**  
> Use consistent seeds (`--train-seed` and `--seed`) across runs to ensure deterministic reproducibility between local, HPC, and AWS environments.  
> Both training and evaluation pipelines are fully deterministic when executed inside the same Apptainer container image.

---

## 💡 Why It Matters

- **End-to-End ML System Design** — Integrates simulation, training, evaluation, and deployment into a single reproducible workflow.  
- **Production-Grade Engineering** — Emphasizes modular design, containerization, CLI-driven automation, and CI validation.  
- **Metric-Driven Insights** — Replaces abstract reward signals with interpretable control metrics (e.g., Tracking Error).  
- **Scalable Infrastructure** — Supports both cloud (Docker on AWS) and on-prem HPC (Apptainer + SLURM).  

This project demonstrates how research-grade reinforcement learning can be engineered into **production-ready ML systems** — combining algorithmic innovation with scalable, maintainable infrastructure.

---

## 🗂️ Repository Structure

```text
Options_In_Simulation/
├── apptainer/              # Container & SLURM definitions
├── src
│   ├──/options_sim/        # CLI tools for training, evaluation, and artifacts
│   └──/packages            # Core modular components: custom agents, assets, utilities, and simulation helpers
├── train/, eval/           # Legacy simulation scripts
├── artifacts/, my_runs/    # Generated logs, metrics, and plots
└── TUD_Report_MJ.pdf       # MSc Thesis reference
```

---

## 🧩 Technologies

**Python**, **PyTorch**, **NVIDIA Isaac Sim**, **OmniIsaacGymEnvs**, **skrl**,  
**Apptainer (Singularity)**, **SLURM**, **Docker**, **Continuous Integration (CI/CD)**

---

## 📘 Reference

> **M. Jansen**,  
> *Learning-Driven Torque Control for Skid-Steer Robots*,  
> TU Delft, 2024.  
> [Full Thesis →](https://repository.tudelft.nl/record/uuid:0bcc777f-cf1c-49fe-8dbf-c18237864841)

---

*This repository showcases the complete lifecycle of a reinforcement-learning control system, from simulation and training to evaluation and scalable, reproducible deployment.*
