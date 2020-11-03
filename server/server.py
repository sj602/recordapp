import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def audio_resolution(original_audio):
    with open(f"{BASE_DIR}/src/file.txt", "w+") as f:
        f.write(original_audio)
    os.system(
        f"python {BASE_DIR}/src/run.py eval --logname {BASE_DIR}/src/model.ckpt --wav-file-list {BASE_DIR}/src/file.txt --r 4"
    )

    # with open(f"{SERVER_DIR}/src/file.txt", "w+") as f:
    #     f.write(original_audio)
    # os.system(
    #     f"python {SERVER_DIR}/src/run.py eval --logname {SERVER_DIR}/src/model.ckpt --wav-file-list {SERVER_DIR}/src/file.txt --r 4"
    # )
