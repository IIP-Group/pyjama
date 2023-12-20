#%%
# from subprocess import Popen
# from time import sleep

# # TODO only same number of GPUs as parameter indices to run supported right now

# # gpus = range(4)
# # parameter_indices_to_run = range(4)
# gpus = [0, 5]
# parameter_indices_to_run = [4, 5]


# procs = []
# for gpu_num, parameter_num in zip(gpus, parameter_indices_to_run):
#     p = Popen(["python", "train.py", str(gpu_num), str(parameter_num)])
#     procs.append(p)
#     print(f"Started process {p.pid} on GPU {gpu_num} with parameter {parameter_num}")

# for p in procs:
#     p.wait()
# %%


from subprocess import Popen
from collections import deque
from threading import Thread

gpus = [0, 5, 6, 7]
parameter_indices_to_run = range(64)

# Create a queue for each GPU
gpu_queues = {gpu: deque() for gpu in gpus}

# Assign each parameter to a GPU queue in a round-robin fashion
for i, parameter_num in enumerate(parameter_indices_to_run):
    gpu_num = gpus[i % len(gpus)]
    gpu_queues[gpu_num].append(parameter_num)

# Function to handle a single GPU queue
def handle_gpu_queue(gpu_num, queue):
    while queue:
        parameter_num = queue.popleft()
        p = Popen(["python", "train.py", str(gpu_num), str(parameter_num)])
        print(f"Started process {p.pid} on GPU {gpu_num} with parameter {parameter_num}")
        p.wait()

# Start a thread for each GPU
threads = []
for gpu_num, queue in gpu_queues.items():
    t = Thread(target=handle_gpu_queue, args=(gpu_num, queue))
    t.start()
    threads.append(t)

# Wait for all threads to finish
for t in threads:
    t.join()
