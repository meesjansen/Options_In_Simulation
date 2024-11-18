#!/bin/bash
#SBATCH --job-name=isaac-sim-job
#SBATCH --partition=general  # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=04:00:00
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=2   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=16GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=isaac-sim-output_%j.log
#SBATCH --error=isaac-sim-error_%j.log # Set name of error log. %j is the Slurm jobId

#SBATCH --gres=gpu:A40:1 # Request 1 GPU

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"  # check sbatch directives are working

set -x

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2') 

/usr/bin/nvidia-smi # Check sbatch settings are working (it should show the GPU that you requested)

# Load necessary modules
module use /opt/insy/modulefiles # Use DAIC INSY software collection
module load cuda/12.1  # Adjust the CUDA version as needed

export APPTAINER_ROOT="/winhome/meesjansen"
export APPTAINER_NAME="isaac-sim-custom.sif"

export ISAACSIM_PATH="/isaac-sim"
export ISAACSIM_PYTHON_EXE="/isaac-sim/python.sh"

# Set environment variables for the container
export data=/winhome/meesjansen/isaac-sim  # Replace with your HOST data directory
export HOME=/workspace  # Replace with your CONTAINER home directory path


# Run script
srun apptainer exec \
    --nv -C \
    --env "ACCEPT_EULA=Y" \
    --env "PRIVACY_CONSENT=Y" \
    -B "$data/cache/kit:/isaac-sim/kit/cache" \
    -B "$data/kit/exts/omni.gpu_foundation/cache:/isaac-sim/kit/exts/omni.gpu_foundation/cache" \
    -B "$data/cache/ov:$HOME/.cache/ov" \
    -B "$data/cache/pip:$HOME/.cache/pip" \
    -B "$data/cache/glcache:$HOME/.cache/nvidia/GLCache" \
    -B "$data/cache/computecache:$HOME/.nv/ComputeCache" \
    -B "$data/logs:$HOME/.nvidia-omniverse/logs" \
    -B "$data/data:$HOME/.local/share/ov/data" \
    -B "$data/documents:$HOME/Documents" \
    $APPTAINER_ROOT/$APPTAINER_NAME    # Path to the container to run



# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"