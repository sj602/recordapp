import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def audio_resolution(original_audio):
    file = f"{BASE_DIR}/src/file.txt"
    with open(file, "w+") as f:
        f.write(original_audio)
        os.system(
            f"python {BASE_DIR}/src/run.py eval --logname {BASE_DIR}/src/model.ckpt --wav-file-list {original_audio} --r 4"
        )
