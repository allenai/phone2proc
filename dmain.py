import os
from datetime import datetime
from multiprocessing import Pool, Value
from time import sleep

import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from main import generate_house

print("Starting at", datetime.now())


processes = 32
n_gpus = torch.cuda.device_count()

controllers = {}


def generate_houses(i: int) -> None:
    global counter
    global house_generator
    global n_gpus

    pid = os.getpid()
    print(f"Using {pid}")
    if pid not in controllers:
        gpu_i = pid % n_gpus
        controllers[pid] = Controller(
            branch="nanna",
            scene="Procedural",
            quality="Low",
            platform=CloudRendering,
            gpu_device=gpu_i,
        )

    controller = controllers[pid]

    # NOTE: sometimes house_generator.sample() hangs
    print("generating house")
    generate_house(controller)
    print("generated house")

    sleep(0.1)
    print(f"houses:", len(os.listdir("back-apartment")))


with Pool(processes=processes) as p:
    r = p.map(generate_houses, range(100_000))
