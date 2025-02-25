# Options In Simulation

Welcome to **Options In Simulation** – a research and experimentation platform built on top of NVIDIA’s Omniverse simulation ecosystem. This project leverages the [omniisaacgymenvs](https://github.com/NVIDIA-Omniverse/omniisaacgymenvs) framework for creating robust, high-fidelity simulation environments and integrates the [skrl framework](https://github.com/ToniRV/skrl) to implement and compare state-of-the-art reinforcement learning algorithms. The aim is to provide a modular and extensible platform where researchers and students can introduce, test, and compare different simulation options and learning strategies.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running Experiments](#running-experiments)



---

## Overview

**Options In Simulation** is designed as an experimental framework for:

- **Developing and testing new simulation scenarios:** Built on omniisaacgymenvs, the platform offers an adaptable simulation backbone for robotics and physics-based experiments.
- **Evaluating reinforcement learning algorithms:** By integrating the skrl framework, the project allows for systematic testing of various RL strategies within a high-fidelity simulated environment.
- **Collaborative development and research:** With a clear, modular structure, the project is intended for easy collaboration, making it straightforward for new contributors to introduce their own experiments and simulation “options.”

This repository is ideal for students and researchers looking to explore simulation-based learning and robotics control strategies in a controlled and extensible environment.

---

## Features

- **Modular Environment Design:** Extend or modify base environments from omniisaacgymenvs.
- **Advanced RL Integration:** Utilize the skrl framework to implement, train, and compare diverse reinforcement learning algorithms.
- **Extensible Codebase:** Designed to be easily adapted for additional simulation scenarios and learning methods.
- **Detailed Documentation:** This README and accompanying docs provide a comprehensive guide to the project’s structure and usage.

---

## Prerequisites

Before getting started, ensure that you have the following installed:

- **Python 3.8+:** Will come with and thus depend on your IsaacSim Installation
- **NVIDIA Omniverse IsaacSim:** Check compatibility with the version required by omniisaacgymenvs.
- **omniisaacgymenvs:** Base simulation environment. Refer to the [official repository](https://github.com/NVIDIA-Omniverse/omniisaacgymenvs) for setup instructions.
- **skrl Framework:** For reinforcement learning implementations. See the [skrl GitHub page](https://github.com/ToniRV/skrl) for details.

---

## Installation

## Docker-based Installation

For this project, we strongly recommend using Docker to ensure a consistent, reproducible, and isolated environment. Docker simplifies the setup process by bundling all dependencies—including NVIDIA IsaacSim, omniisaacgymenvs, and the skrl framework—into a single container. This guarantees that everyone working on the project has the exact same configuration, which is especially critical when working with GPU-based simulations.

### 1. Host Setup

Make sure you have Docker installed on a system with a compatible GPU. Then, from your local machine, navigate to a working directory and connect to your remote machine (if applicable):

```bash
cd ~/Downloads

ssh -i "RLSim1.pem" -vvv ubuntu@ec2-18-159-195-237.eu-central-1.compute.amazonaws.com
```

### 2. Pull the IsaacSim Docker Image

Pull the official NVIDIA IsaacSim Docker image (version 4.0.0):

```bash
docker pull nvcr.io/nvidia/isaac-sim:4.0.0
```

### 3. Run the Docker Container

Start the container with GPU support and mount the necessary directories. This command sets up the container with proper volume mappings and environment variables required by IsaacSim and our project:

```bash
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

### 4. Inside the Docker Container

After launching the container, run the following commands to set up the necessary prerequisites and install project dependencies:

1. **Update Packages and Install Essential Tools**

   ```bash
   apt-get update 
   apt-get install -y git nano 
   /isaac-sim/python.sh -m pip install tensorflow
   ```

2. **Install omniisaacgymenvs**

   Navigate to the omniisaacgymenvs directory and install it in editable mode:

   ```bash
   cd /workspace/omniisaacgymenvs/OmniIsaacGymEnvs 
   /isaac-sim/python.sh -m pip install -e .
   ```

3. **Install the skrl Framework**

   Switch to the skrl directory and install the package with PyTorch support:

   ```bash
   cd /workspace/omniisaacgymenvs/skrl
   /isaac-sim/python.sh -m pip install .[torch]
   ```

4. **Set Up the Options In Simulation Project**

   If you haven't already cloned the repository, you can do so. Otherwise, update the repository and install the project dependencies:

   ```bash
   # If necessary, clone the repository (uncomment the lines below)
   # cd /workspace/omniisaacgymenvs/
   # git clone https://github.com/meesjansen/Options_In_Simulation.git

   cd /workspace/omniisaacgymenvs/Options_In_Simulation
   git pull
   /isaac-sim/python.sh -m pip install -e .
   /isaac-sim/python.sh train.py
   ```

These steps will install all necessary dependencies and run the training script, ensuring that the environment is correctly set up within the Docker container.

---

Using Docker in this manner guarantees that the complex dependencies required by NVIDIA IsaacSim and the simulation frameworks are properly managed, reducing setup time and avoiding conflicts across different systems.

## Project Structure

Our repository is organized to streamline both simulation-based and real-world experiments, drawing inspiration from the [skrl documentation](https://skrl.readthedocs.io/en/latest/intro/examples.html#real-world-examples). In particular, we recommend reviewing the KUKA LBR iiwa example in the Real World Examples section for practical insights.

The key components of our project include:

- **Train Files**: Scripts dedicated to launching training sessions.
- **Eval File**: A dedicated script for evaluating the performance of trained models.
- **Environment Files**: Define and configure the simulation environments, integrating with Omniverse Isaac Gym.
- **Utils**: Utility scripts for creating terrains and other environmental features.
- **Assets**: Contains URDF files that are imported into IsaacSim to generate USDs.
- **Robots**: Components that load the USDs into the IsaacSim environment, representing the physical robots.
- **Agents**: Houses both custom and skrl reinforcement learning algorithms.
- **Models**: Stores the trained model checkpoints and weights used by the agents.
- **Trainers**: Implements the update logic that governs how the agents are trained.


## Running Experiments

The process for running experiments is designed to be straightforward, following a workflow similar to the skrl examples. Here’s a brief overview:

1. **Training**:  
   Use the train files to start a training session. For example:
   ```bash
   python train.py
   ```
This command initializes the environment, loads the necessary assets, and begins training the agents using the defined trainers.

2. **Evaluation**:
After training, you can evaluate the performance of your model by running the evaluation file:
```bash
python eval.py
```
