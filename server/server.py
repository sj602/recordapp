import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
# from src.models.io import upsample_wav
# from src.run import get_model

def audio_resolution(original_audio):
    print('audio_resolution 진입')
    file = f"{BASE_DIR}/src/file.txt"
    with open(file, "w+") as f:
        f.write(original_audio)
        print(f'BASE_DIR: {BASE_DIR}')
        print(f'write file: {original_audio}')
        # args = {
        #     "logname": f"{BASE_DIR}\src\\model.ckpt",
        #     "r": 4,
        #     "sr": 16000
        # }
        # model = get_model(args, 0, args['r'], from_ckpt=True, train=False)
        # upsample_wav(original_audio, args, model)
        # os.system(
        #     f"python {BASE_DIR}\src\\run.py eval --logname {BASE_DIR}\src\\model.ckpt --wav-file-list {file} —r 4"
        # )
        # f.write(original_audio)

        os.system(
            f"python {BASE_DIR}/src/run.py eval --logname {BASE_DIR}/src/model.ckpt --wav-file-list {original_audio} --r 4"
        )
        print(f'exit')