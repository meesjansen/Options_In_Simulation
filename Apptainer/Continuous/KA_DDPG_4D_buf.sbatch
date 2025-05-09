#!/bin/bash
#SBATCH --job-name=isaac-sim-job
#SBATCH --partition=general  # Request partition. Default is 'general' 
#SBATCH --qos=medium      # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=20:00:00
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=2   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=16GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=KA_DDPG_4D_buf_extra_output_%j.log
#SBATCH --error=KA_DDPG_4D_buf_extra_error_%j.log # Set name of error log. %j is the Slurm jobId

#SBATCH --gres=gpu:a40:1 # Request 1 GPU

export WANDB_API_KEY="e01d8dc5d40f0a16f58885a3808bf560a32837d7"

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"  # check sbatch directives are working

set -x

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2') 

/usr/bin/nvidia-smi # Check sbatch settings are working (it should show the GPU that you requested)

# Load necessary modules
module use /opt/insy/modulefiles # Use DAIC INSY software collection

export APPTAINER_ROOT="/winhome/meesjansen"
export APPTAINER_NAME="isaac-sim-custom.sif"

export ISAACSIM_PATH="/isaac-sim"
export ISAACSIM_PYTHON_EXE="/isaac-sim/python.sh"

# Set environment variables for the container
export data=/home/nfs/meesjansen/isaac-sim  # Replace with your HOST data directory
export HOME=/root  # Replace with your CONTAINER home directory path


# Run script
srun apptainer exec --nv \
    -C \
    --env "ACCEPT_EULA=Y" \
    --env "PRIVACY_CONSENT=Y" \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    -B "/home/nfs/meesjansen/Options_In_Simulation:/workspace/Options_In_Simulation" \
    -B "$data/cache/kit:/isaac-sim/kit/cache" \
    -B "$data/cache/ov:$HOME/.cache/ov" \
    -B "$data/cache/pip:$HOME/.cache/pip" \
    -B "$data/cache/glcache:$HOME/.cache/nvidia/GLCache" \
    -B "$data/cache/computecache:$HOME/.nv/ComputeCache" \
    -B "$data/logs:$HOME/.nvidia-omniverse/logs" \
    -B "$data/data:$HOME/.local/share/ov/data" \
    -B "$data/documents:$HOME/Documents" \
    $APPTAINER_ROOT/$APPTAINER_NAME /isaac-sim/python.sh /workspace/Options_In_Simulation/train_KA-DDPG_4D_buf.py



# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"