#%%
from subprocess import Popen
from time import sleep

# TODO only same number of GPUs as parameter indices to run supported right now
gpus = range(6)
parameter_indices_to_run = range(6)

procs = []
for gpu_num, parameter_num in zip(gpus, parameter_indices_to_run):
    p = Popen(["python", "train.py", str(gpu_num), str(parameter_num)])
    procs.append(p)
    print(f"Started process {p.pid} on GPU {gpu_num} with parameter {parameter_num}")

for p in procs:
    p.wait()
# %%
