# Print out realtime audio volume as ascii bars

import sounddevice as sd
import numpy as np

duration = 50  # seconds

def print_sound(indata, outdata, frames, time, status):
    volume_norm = np.linalg.norm(indata)*10
    print ("|" * int(volume_norm))
    #print(str(volume_norm))

with sd.Stream(callback=print_sound):
    sd.sleep(duration * 1000)
