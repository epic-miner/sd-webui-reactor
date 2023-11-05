import os
from pathlib import Path

from loguru import logger as debug_logger
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "_globals.log")
debug_logger.add(log_path, backtrace=True, diagnose=True)

IS_RUN: bool = False
BASE_PATH = os.path.join(Path(__file__).parents[1])
DEVICE_LIST: list = ["CPU", "CUDA"]

@debug_logger.catch
def updateDevice():
    try:
        LAST_DEVICE_PATH = os.path.join(BASE_PATH, "last_device.txt")
        with open(LAST_DEVICE_PATH) as f:
            for el in f:
                device = el.strip()
    except:
        debug_logger.exception("Error")
        device = "CPU"
    return device

DEVICE = updateDevice()

# @debug_logger.catch
# def test(a, b):
#     return a / b

# test(1, 0)
