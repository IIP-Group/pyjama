from subprocess import Popen
from collections import deque
from threading import Thread
import datetime

# gpus = [0, 1, 2, 3, 4, 5, 6, 7]
gpus = [0, 7]
parameter_indices_to_run = range(3)
filename = "train2.py"

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
        p = Popen(["python", filename, str(gpu_num), str(parameter_num)])
        print(f"Started process {p.pid} on GPU {gpu_num} with parameter {parameter_num} at {datetime.datetime.now().strftime('%X')}")
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
