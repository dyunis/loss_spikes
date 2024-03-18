# install instructions:

```bash
# install miniconda
conda install -c pytorch -c nvidia pytorch pytorch-cuda=11.8 torchvision torchaudio

pip install lightning==2.0.6  # for trainer, slurm with multi-gpu is weird with later versions
pip install datasets  # for wikitext 103 dataset
pip install matplotlib  # for visualization
pip install wandb  # for experiment tracking
```

# Apptainer install
```bash
cd [container directory]

# singularity process, can do this on jindo, then copy final *.sif file to slurm for cluster
# need a bunch of --nv tools from adam to install with gpu support
apptainer build --sandbox lmc_svd docker://nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
# sudo is needed for --writable, but makes it impossible to remove the directory later
sudo apptainer shell --no-home --nv --writable lmc_svd  # -nv binds gpu access within container, home dir is nfs which has issues

# below is inside container
apt update
apt upgrade
apt install wget
apt install vim

# need this for installing conda inside the container and making it accessible
mkdir /conda_tmp
cd /conda_tmp

# from slurm docs
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /conda_tmp/mc3
rm Miniconda3-latest-Linux-x86_64.sh
eval "$(/conda_tmp/mc3/bin/conda 'shell.bash' 'hook')"

conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 torchvision torchaudio

pip install lightning==2.0.6  # for trainer, slurm with multi-gpu is weird with later versions
pip install datasets  # for wikitext103 dataset
pip install matplotlib  # for visualization
pip install wandb  # for experiment tracking
pip install jupyter  # for plotting
pip install nbstripout  # to strip output from ipynbs if loading takes forever
exit

# below is outside container
sudo apptainer build lmc_svd.sif lmc_svd/
# copy to jindo from adam's box/nfs
# scp lmc_svd.sif dyunis@jindo.ttic.edu:/scratch/dyunis/lmc_svd.sif
# to run:
# /scratch/dyunis/./lmc_svd.sif [python cmd]
# need --nv for gpu 
# mount instructions: https://apptainer.org/docs/user/main/bind_paths_and_mounts.html
apptainer shell --mount type=bind,src=/scratch,dst=/scratch --mount type=bind,src=/share,dst=/share --nv /scratch/dyunis/apptainer/lmc_svd.sif
source /conda_tmp/mc3/bin/activate

# see slurm_run.py for batch mode
```
