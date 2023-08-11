import os
import subprocess
import time

while True:
    start = time.time()
    print("running dmain")
    process = subprocess.Popen(
        [
            "python3",
            "dmain.py",
        ]
    )

    # sleep for 5 minutes
    print("sleeping for 5 minutes")
    while time.time() - start < 60 * 10:
        time.sleep(1)

    print("Killing process")
    process.kill()
    # os.system("sudo killall python3")
    os.system("sudo ps aux | grep thor- | awk '{print $2}' | xargs kill")
    print("start sleeping")
    time.sleep(15)
    print("done sleeping")
