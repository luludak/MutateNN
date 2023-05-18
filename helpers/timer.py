import math
import time

class Timer:

    def __init__(self):
        pass

    def get_epoch_timestamp(self, use_floor = True):
        time_to_return = time.time()
        return math.floor(time_to_return) if use_floor else time_to_return
