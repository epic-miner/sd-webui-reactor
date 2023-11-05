import subprocess
import os, sys
from typing import Any
import pkg_resources
from tqdm import tqdm
import urllib.request
from packaging import version as pv

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        model_path = os.path.abspath("models")

from loguru import logger as debug_logger
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "_install.log")
debug_logger.add(log_path, backtrace=True, diagnose=True)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

req_file = os.path.join(BASE_PATH, "requirements.txt")

models_dir_old = os.path.join(models_path, "roop")
models_dir = os.path.join(models_path, "insightface")

# Check the deprecated 'roop' folder
if os.path.exists(models_dir_old):
    if not os.listdir(models_dir_old) and (not os.listdir(models_dir) or not os.path.exists(models_dir)):
        os.rename(models_dir_old, models_dir)
    else:
        import shutil
        for file in os.listdir(models_dir_old):
            shutil.move(os.path.join(models_dir_old, file), os.path.join(models_dir, file))
        try:
            os.rmdir(models_dir_old)
        except Exception as e:
            print(f"OSError: {e}")
            
model_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)

def pip_install(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args])

def pip_uninstall(*args):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *args])

def is_installed (
        package: str, version: str | None = None, strict: bool = True
):
    has_package = None
    try:
        has_package = pkg_resources.get_distribution(package)
        if has_package is not None:
            installed_version = has_package.version
            if (installed_version != version and strict == True) or (pv.parse(installed_version) < pv.parse(version) and strict == False):
                return False
            else:
                return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading...', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(model_path):
    download(model_url, model_path)

print("ReActor preheating...", end=' ')

last_device = None
first_run = False

try:
    last_device_log = os.path.join(BASE_PATH, "last_device.txt")
    with open(last_device_log) as f:
        for el in f:
            last_device = el.strip()
except:
    last_device = "CPU"
    first_run = True
    with open(os.path.join(BASE_PATH, "last_device.txt"), "w") as txt:
        txt.write(last_device)

with open(req_file) as file:
    install_count = 0
    ort = "onnxruntime-gpu"
    import torch
    try:
        if torch.cuda.is_available():
            if first_run:
                last_device = "CUDA"
        elif torch.backends.mps.is_available() or hasattr(torch,'dml'):
            ort = "onnxruntime"
            # to prevent errors when ORT-GPU is installed but we want ORT instead:
            if first_run:
                pip_uninstall("onnxruntime", "onnxruntime-gpu")
            # just in case:
            if last_device == "CUDA":
                last_device = "CPU"
        else:
            if last_device == "CUDA":
                last_device = "CPU"
        with open(os.path.join(BASE_PATH, "last_device.txt"), "w") as txt:
            txt.write(last_device)
        if not is_installed(ort,"1.16.1",False):
            install_count += 1
            pip_install(ort)
    except Exception as e:
        debug_logger.exception("InstallError")
        print(e)
        print(f"\nERROR: Failed to install {ort} - ReActor won't start")
        raise e
    print(f"Device: {last_device}")
    strict = True
    for package in file:
        package_version = None
        try:
            package = package.strip()
            if "==" in package:
                package_version = package.split('==')[1]
            elif ">=" in package:
                package_version = package.split('>=')[1]
                strict = False
            if not is_installed(package,package_version,strict):
                install_count += 1
                pip_install(package)
        except Exception as e:
            debug_logger.exception("InstallError")
            print(e)
            print(f"\nERROR: Failed to install {package} - ReActor won't start")
            raise e
    if install_count > 0:
        print(f"""
        +---------------------------------+
        --- PLEASE, RESTART the Server! ---
        +---------------------------------+
        """)
