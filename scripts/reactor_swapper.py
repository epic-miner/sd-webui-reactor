import copy
import os
from dataclasses import dataclass
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy import stats

import insightface
from insightface.app.common import Face

from scripts.reactor_globals import FACE_MODELS_PATH
from scripts.reactor_helpers import (
    get_image_md5hash, 
    get_Device, 
    save_face_model, 
    load_face_model, 
    get_images_from_folder,
    get_random_image_from_folder,
    get_images_from_list,
    set_SDNEXT
)
from scripts.console_log_patch import apply_logging_patch

from modules.face_restoration import FaceRestoration
try: # A1111
    from modules import codeformer_model, gfpgan_model
except: # SD.Next
    from modules.postprocess import codeformer_model, gfpgan_model
    set_SDNEXT()
from modules.upscaler import UpscalerData
from modules.shared import state
from scripts.reactor_logger import logger
from reactor_modules.reactor_mask import apply_face_mask

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")

import warnings

np.warnings = warnings
np.warnings.filterwarnings('ignore')


DEVICE = get_Device()
if DEVICE == "CUDA":
    PROVIDERS = ["CUDAExecutionProvider"]
else:
    PROVIDERS = ["CPUExecutionProvider"]
class MaskOption:
    DEFAULT_FACE_AREAS = ["Face"]
    DEFAULT_FACE_SIZE:int = 512
    DEFAULT_VIGNETTE_THRESHOLD:float = 0.1
    DEFAULT_MASK_BLUR:int = 12,
    DEFAULT_USE_MINIMAL_AREA:bool = True

@dataclass
class EnhancementOptions:
    do_restore_first: bool = True
    scale: int = 1
    upscaler: UpscalerData = None
    upscale_visibility: float = 0.5
    face_restorer: FaceRestoration = None
    restorer_visibility: float = 0.5
    codeformer_weight: float = 0.5
    upscale_force: bool = False

@dataclass
class DetectionOptions:
    det_thresh: float = 0.5
    det_maxnum: int = 0

@dataclass
class MaskOptions:
    mask_areas:List[str]
    save_face_mask: bool = False
    mask_blur:int = 12   
    face_size:int  = 512
    vignette_fallback_threshold:float =0.10
    use_minimal_area:bool = True

MESSAGED_STOPPED = False
MESSAGED_SKIPPED = False

def reset_messaged():
    global MESSAGED_STOPPED, MESSAGED_SKIPPED
    if not state.interrupted:
        MESSAGED_STOPPED = False
    if not state.skipped:
        MESSAGED_SKIPPED = False

def check_process_halt(msgforced: bool = False):
    global MESSAGED_STOPPED, MESSAGED_SKIPPED
    if state.interrupted:
        if not MESSAGED_STOPPED or msgforced:
            logger.status("Stopped by User")
            MESSAGED_STOPPED = True
        return True
    if state.skipped:
        if not MESSAGED_SKIPPED or msgforced:
            logger.status("Skipped by User")
            MESSAGED_SKIPPED = True
        return True
    return False


FS_MODEL = None
ANALYSIS_MODEL = None
MASK_MODEL = None

CURRENT_FS_MODEL_PATH = None
CURRENT_MASK_MODEL_PATH = None

SOURCE_FACES = None
SOURCE_IMAGE_HASH = None
TARGET_FACES = None
TARGET_IMAGE_HASH = None
SOURCE_FACES_LIST = []
SOURCE_IMAGE_LIST_HASH = []

def clear_faces():
    global SOURCE_FACES, SOURCE_IMAGE_HASH
    SOURCE_FACES = None
    SOURCE_IMAGE_HASH = None
    logger.status("Source Images Hash has been reset (for Single Source or Face Model)")

def clear_faces_list():
    global SOURCE_FACES_LIST, SOURCE_IMAGE_LIST_HASH
    SOURCE_FACES_LIST = []
    SOURCE_IMAGE_LIST_HASH = []
    logger.status("Source Images Hash has been reset (for Multiple or Folder Source)")

def clear_faces_target():
    global TARGET_FACES, TARGET_IMAGE_HASH
    TARGET_FACES = None
    TARGET_IMAGE_HASH = None
    logger.status("Target Images Hash has been reset")

def clear_faces_all():
    global SOURCE_FACES, SOURCE_IMAGE_HASH, SOURCE_FACES_LIST, SOURCE_IMAGE_LIST_HASH, TARGET_FACES, TARGET_IMAGE_HASH
    SOURCE_FACES = None
    SOURCE_IMAGE_HASH = None
    TARGET_FACES = None
    TARGET_IMAGE_HASH = None
    SOURCE_FACES_LIST = []
    SOURCE_IMAGE_LIST_HASH = []
    logger.status("All Images Hash has been reset")

def getAnalysisModel():
    global ANALYSIS_MODEL
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=PROVIDERS, root=os.path.join(models_path, "insightface") # note: allowed_modules=['detection', 'genderage']
        )
    return ANALYSIS_MODEL


def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=PROVIDERS)

    return FS_MODEL


def restore_face(image: Image, enhancement_options: EnhancementOptions):
    result_image = image

    if check_process_halt(msgforced=True):
        return result_image
    
    if enhancement_options.face_restorer is not None:
        original_image = result_image.copy()
        numpy_image = np.array(result_image)
        if enhancement_options.face_restorer.name() == "CodeFormer":
            logger.status("Restoring the face with %s (weight: %s)", enhancement_options.face_restorer.name(), enhancement_options.codeformer_weight)
            numpy_image = codeformer_model.codeformer.restore(
                numpy_image, w=enhancement_options.codeformer_weight
            )
        else: # GFPGAN:
            logger.status("Restoring the face with %s", enhancement_options.face_restorer.name())
            numpy_image = gfpgan_model.gfpgan_fix_faces(numpy_image)
            # numpy_image = enhancement_options.face_restorer.restore(numpy_image)
        restored_image = Image.fromarray(numpy_image)
        result_image = Image.blend(
            original_image, restored_image, enhancement_options.restorer_visibility
        )
    
    return result_image

def upscale_image(image: Image, enhancement_options: EnhancementOptions):
    result_image = image

    if check_process_halt(msgforced=True):
        return result_image
    
    if enhancement_options.upscaler is not None and enhancement_options.upscaler.name != "None":
        original_image = result_image.copy()
        logger.status(
            "Upscaling with %s scale = %s",
            enhancement_options.upscaler.name,
            enhancement_options.scale,
        )
        result_image = enhancement_options.upscaler.scaler.upscale(
            original_image, enhancement_options.scale, enhancement_options.upscaler.data_path
        )
        if enhancement_options.scale == 1:
            result_image = Image.blend(
                original_image, result_image, enhancement_options.upscale_visibility
            )
    
    return result_image

def enhance_image(image: Image, enhancement_options: EnhancementOptions):
    result_image = image
    
    if check_process_halt(msgforced=True):
        return result_image
    
    if enhancement_options.do_restore_first:
        
        result_image = restore_face(result_image, enhancement_options)
        result_image = upscale_image(result_image, enhancement_options)

    else:

        result_image = upscale_image(result_image, enhancement_options)
        result_image = restore_face(result_image, enhancement_options)

    return result_image
def enhance_image_and_mask(image: Image.Image, enhancement_options: EnhancementOptions,target_img_orig:Image.Image,entire_mask_image:Image.Image)->Tuple[Image.Image,Image.Image]:
    result_image = image
    
    if check_process_halt(msgforced=True):
        return result_image
    
    if enhancement_options.do_restore_first:
        
      
        result_image = restore_face(result_image, enhancement_options)

        transparent = Image.new("RGBA",result_image.size)
        masked_faces = Image.composite(result_image.convert("RGBA"),transparent,entire_mask_image)
        result_image = Image.composite(result_image,target_img_orig,entire_mask_image)
        result_image = upscale_image(result_image, enhancement_options)

    else:

        result_image = upscale_image(result_image, enhancement_options)
        entire_mask_image = Image.fromarray(cv2.resize(np.array(entire_mask_image),result_image.size, interpolation=cv2.INTER_AREA)).convert("L")
       
        result_image = Image.composite(result_image,target_img_orig,entire_mask_image)
        result_image = restore_face(result_image, enhancement_options)
        transparent = Image.new("RGBA",result_image.size)
        masked_faces = Image.composite(result_image.convert("RGBA"),transparent,entire_mask_image)
    return result_image, masked_faces


    
def get_gender(face, face_index):
    gender = [
        x.sex
        for x in face
    ]
    gender.reverse()
    try:
        face_gender = gender[face_index]
    except:
        logger.error("Gender Detection: No face with index = %s was found", face_index)
        return "None"
    return face_gender

def get_face_gender(
        face,
        face_index,
        gender_condition,
        operated: str,
        gender_detected,
):
    face_gender = gender_detected
    if face_gender == "None":
        return None, 0
    logger.status("%s Face %s: Detected Gender -%s-", operated, face_index, face_gender)
    if (gender_condition == 1 and face_gender == "F") or (gender_condition == 2 and face_gender == "M"):
        logger.status("OK - Detected Gender matches Condition")
        try:
            return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
        except IndexError:
            return None, 0
    else:
        logger.status("WRONG - Detected Gender doesn't match Condition")
        return sorted(face, key=lambda x: x.bbox[0])[face_index], 1

def get_face_age(face, face_index):
    age = [
        x.age
        for x in face
    ]
    age.reverse()
    try:
        face_age = age[face_index]
    except:
        logger.error("Age Detection: No face with index = %s was found", face_index)
        return "None"
    return face_age

def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

def analyze_faces(img_data: np.ndarray, det_size=(640, 640), det_thresh=0.5, det_maxnum=0):
    logger.info("Applied Execution Provider: %s", PROVIDERS[0])
    face_analyser = copy.deepcopy(getAnalysisModel())
    face_analyser.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
    return face_analyser.get(img_data, max_num=det_maxnum)

def get_face_single(img_data: np.ndarray, face, face_index=0, det_size=(640, 640), gender_source=0, gender_target=0, det_thresh=0.5, det_maxnum=0):

    buffalo_path = os.path.join(models_path, "insightface/models/buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    face_age = "None"
    try:
        face_age = get_face_age(face, face_index)
    except:
        logger.error("Cannot detect any Age for Face index = %s", face_index)
    
    face_gender = "None"
    try:
        face_gender = get_gender(face, face_index)
        gender_detected = face_gender
        face_gender = "Female" if face_gender == "F" else ("Male" if face_gender == "M" else "None")
    except:
        logger.error("Cannot detect any Gender for Face index = %s", face_index)
    
    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half, det_thresh, det_maxnum), face_index, det_size_half, gender_source, gender_target, det_thresh, det_maxnum)
        faces, wrong_gender = get_face_gender(face,face_index,gender_source,"Source",gender_detected)
        return faces, wrong_gender, face_age, face_gender

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half, det_thresh, det_maxnum), face_index, det_size_half, gender_source, gender_target, det_thresh, det_maxnum)
        faces, wrong_gender = get_face_gender(face,face_index,gender_target,"Target",gender_detected)
        return faces, wrong_gender, face_age, face_gender
    
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(img_data, analyze_faces(img_data, det_size_half, det_thresh, det_maxnum), face_index, det_size_half, gender_source, gender_target, det_thresh, det_maxnum)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index], 0, face_age, face_gender
    except IndexError:
        return None, 0, face_age, face_gender


def swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    enhancement_options: Union[EnhancementOptions, None] = None,
    gender_source: int = 0,
    gender_target: int = 0,
    source_hash_check: bool = True,
    target_hash_check: bool = False,
    device: str = "CPU",
    mask_face: bool = False,
    select_source: int = 0,
    face_model: str = "None",
    source_folder: str = "",
    source_imgs: Union[List, None] = None,
    random_image: bool = False,
    detection_options: Union[DetectionOptions, None] = None,
    mask_options:Union[MaskOptions, None]= None,
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH, PROVIDERS, SOURCE_FACES_LIST, SOURCE_IMAGE_LIST_HASH

    result_image = target_img
    masked_faces = None
    PROVIDERS = ["CUDAExecutionProvider"] if device == "CUDA" else ["CPUExecutionProvider"]
    
    if check_process_halt():
        return result_image, [], 0
    if mask_options is None:
        mask_options = MaskOptions()
        
    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))

        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        target_img_orig = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        entire_mask_image = np.zeros_like(np.array(target_img))

        output: List = []
        output_info: str = ""
        swapped = 0

        # *****************
        # SWAP from FOLDER or MULTIPLE images:
        
        if (select_source == 0 and source_imgs is not None) or (select_source == 2 and (source_folder is not None and source_folder != "")):

            result = []

            if random_image and select_source == 2:
                source_images,source_images_names = get_random_image_from_folder(source_folder)
                logger.status(f"Processing with Random Image from the folder: {source_images_names[0]}")
            else:
                source_images,source_images_names = get_images_from_folder(source_folder) if select_source == 2 else get_images_from_list(source_imgs)

            if len(source_images) > 0:
                source_img_ff = []
                source_faces_ff = []
                for i, source_image in enumerate(source_images):

                    source_image = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
                    source_img_ff.append(source_image)

                    if source_hash_check:

                        source_image_md5hash = get_image_md5hash(source_image)

                        if len(SOURCE_IMAGE_LIST_HASH) == 0:
                            SOURCE_IMAGE_LIST_HASH = [source_image_md5hash]
                            source_image_same = False
                        elif len(SOURCE_IMAGE_LIST_HASH) == i:
                            SOURCE_IMAGE_LIST_HASH.append(source_image_md5hash)
                            source_image_same = False
                        else:
                            source_image_same = True if SOURCE_IMAGE_LIST_HASH[i] == source_image_md5hash else False
                            if not source_image_same:
                                SOURCE_IMAGE_LIST_HASH[i] = source_image_md5hash

                        logger.info("(Image %s) Source Image MD5 Hash = %s", i, SOURCE_IMAGE_LIST_HASH[i])
                        logger.info("(Image %s) Source Image the Same? %s", i, source_image_same)

                        if len(SOURCE_FACES_LIST) == 0:
                            logger.status(f"Analyzing Source Image {i}: {source_images_names[i]}...")
                            source_faces = analyze_faces(source_image, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
                            SOURCE_FACES_LIST = [source_faces]
                        elif len(SOURCE_FACES_LIST) == i and not source_image_same:
                            logger.status(f"Analyzing Source Image {i}: {source_images_names[i]}...")
                            source_faces = analyze_faces(source_image, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
                            SOURCE_FACES_LIST.append(source_faces)
                        elif len(SOURCE_FACES_LIST) != i and not source_image_same:
                            logger.status(f"Analyzing Source Image {i}: {source_images_names[i]}...")
                            source_faces = analyze_faces(source_image, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
                            SOURCE_FACES_LIST[i] = source_faces
                        elif source_image_same:
                            logger.status("(Image %s) Using Hashed Source Face(s) Model...", i)
                            source_faces = SOURCE_FACES_LIST[i]

                    else:
                        logger.status(f"Analyzing Source Image {i}...")
                        source_faces = analyze_faces(source_image, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)

                    if source_faces is not None:
                        source_faces_ff.append(source_faces)
            
            if len(source_faces_ff) > 0:

                if target_hash_check:
                
                    target_image_md5hash = get_image_md5hash(target_img)

                    if TARGET_IMAGE_HASH is None:
                        TARGET_IMAGE_HASH = target_image_md5hash
                        target_image_same = False
                    else:
                        target_image_same = True if TARGET_IMAGE_HASH == target_image_md5hash else False
                        if not target_image_same:
                            TARGET_IMAGE_HASH = target_image_md5hash

                    logger.info("Target Image MD5 Hash = %s", TARGET_IMAGE_HASH)
                    logger.info("Target Image the Same? %s", target_image_same)
                    
                    if TARGET_FACES is None or not target_image_same:
                        logger.status("Analyzing Target Image...")
                        target_faces = analyze_faces(target_img, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
                        TARGET_FACES = target_faces
                    elif target_image_same:
                        logger.status("Using Hashed Target Face(s) Model...")
                        target_faces = TARGET_FACES
                
                else:
                    logger.status("Analyzing Target Image...")
                    target_faces = analyze_faces(target_img, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)

                for i,source_faces in enumerate(source_faces_ff):

                    logger.status("(Image %s) Detecting Source Face, Index = %s", i, source_faces_index[0])
                    source_face, wrong_gender, source_age, source_gender = get_face_single(source_img_ff[i], source_faces, face_index=source_faces_index[0], gender_source=gender_source, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
                    
                    if source_age != "None" or source_gender != "None":
                        logger.status("(Image %s) Detected: -%s- y.o. %s", i, source_age, source_gender)

                    if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                        logger.status("Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.")
                    
                    elif source_face is not None:

                        result_image, output, swapped = operate(source_img_ff[i],target_img,target_img_orig,model,source_faces_index,faces_index,source_faces,target_faces,gender_source,gender_target,source_face,wrong_gender,source_age,source_gender,output,swapped,mask_face,entire_mask_image,enhancement_options,detection_options)

                        result.append(result_image)

                    result = [result_image] if len(result) == 0 else result
            
            return result, output, swapped
        
        # END
        # *****************
        
        # ***********************
        # SWAP from IMG or MODEL:

        else:
        
            if select_source == 0 and source_img is not None:
                
                source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

                if source_hash_check:

                    source_image_md5hash = get_image_md5hash(source_img)

                    if SOURCE_IMAGE_HASH is None:
                        SOURCE_IMAGE_HASH = source_image_md5hash
                        source_image_same = False
                    else:
                        source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                        if not source_image_same:
                            SOURCE_IMAGE_HASH = source_image_md5hash

                    logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
                    logger.info("Source Image the Same? %s", source_image_same)

                    if SOURCE_FACES is None or not source_image_same:
                        logger.status("Analyzing Source Image...")
                        source_faces = analyze_faces(source_img, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
                        SOURCE_FACES = source_faces
                    elif source_image_same:
                        logger.status("Using Hashed Source Face(s) Model...")
                        source_faces = SOURCE_FACES

                else:
                    logger.status("Analyzing Source Image...")
                    source_faces = analyze_faces(source_img, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
            
            elif select_source == 1 and (face_model is not None and face_model != "None"):
                source_face_model = [load_face_model(face_model)]
                if source_face_model is not None:
                    source_faces_index = [0]
                    source_faces = source_face_model
                    logger.status(f"Using Loaded Source Face Model: {face_model}")
                else:
                    logger.error(f"Cannot load Face Model File: {face_model}")
            
            else:
                logger.error("Cannot detect any Source")
                return result_image, [], 0

            if source_faces is not None:

                if target_hash_check:
                
                    target_image_md5hash = get_image_md5hash(target_img)

                    if TARGET_IMAGE_HASH is None:
                        TARGET_IMAGE_HASH = target_image_md5hash
                        target_image_same = False
                    else:
                        target_image_same = True if TARGET_IMAGE_HASH == target_image_md5hash else False
                        if not target_image_same:
                            TARGET_IMAGE_HASH = target_image_md5hash

                    logger.info("Target Image MD5 Hash = %s", TARGET_IMAGE_HASH)
                    logger.info("Target Image the Same? %s", target_image_same)
                    
                    if TARGET_FACES is None or not target_image_same:
                        logger.status("Analyzing Target Image...")
                        target_faces = analyze_faces(target_img, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
                        TARGET_FACES = target_faces
                    elif target_image_same:
                        logger.status("Using Hashed Target Face(s) Model...")
                        target_faces = TARGET_FACES
                
                else:
                    logger.status("Analyzing Target Image...")
                    target_faces = analyze_faces(target_img, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)

                logger.status("Detecting Source Face, Index = %s", source_faces_index[0])
                if select_source == 0 and source_img is not None:
                    source_face, wrong_gender, source_age, source_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
                else:
                    source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                    wrong_gender = 0
                    source_age = source_face["age"]
                    source_gender = "Female" if source_face["gender"] == 0 else "Male"
                
                if source_age != "None" or source_gender != "None":
                    logger.status("Detected: -%s- y.o. %s", source_age, source_gender)

                output_info = f"SourceFaceIndex={source_faces_index[0]};Age={source_age};Gender={source_gender}\n"
                output.append(output_info)
                
                if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                    logger.status("Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.")
                
                elif source_face is not None:

                    result_image, output, swapped = operate(source_img,target_img,target_img_orig,model,source_faces_index,faces_index,source_faces,target_faces,gender_source,gender_target,source_face,wrong_gender,source_age,source_gender,output,swapped,mask_face,entire_mask_image,enhancement_options,detection_options)
                
                else:
                    logger.status("No source face(s) in the provided Index")
            else:
                logger.status("No source face(s) found")
    
            return result_image, output, swapped
        
        # END
        # **********************
    
    return result_image, [], 0

def build_face_model(image: Image.Image, name: str, save_model: bool = True, det_size=(640, 640)):
    if image is None:
        error_msg = "Please load an Image"
        logger.error(error_msg)
        return error_msg
    if name is None:
        error_msg = "Please filled out the 'Face Model Name' field"
        logger.error(error_msg)
        return error_msg
    apply_logging_patch(1)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if save_model:
        logger.status("Building Face Model...")
    face_model = analyze_faces(image, det_size)

    if len(face_model) == 0:
        det_size_half = half_det_size(det_size)
        face_model = analyze_faces(image, det_size_half)
    
    if face_model is not None and len(face_model) > 0:
        if save_model:
            face_model_path = os.path.join(FACE_MODELS_PATH, name + ".safetensors")
            save_face_model(face_model[0],face_model_path)
            logger.status("--Done!--")
            done_msg = f"Face model has been saved to '{face_model_path}'"
            logger.status(done_msg)
            return done_msg
        else:
            return face_model[0]
    else:
        no_face_msg = "No face found, please try another image"
        logger.error(no_face_msg)
        return no_face_msg

def blend_faces(images_list: List, name: str, compute_method: int = 0, shape_check: bool = False):
    faces = []
    embeddings = []
    images: List[Image.Image] = []
    images, images_names = get_images_from_list(images_list)
    for i,image in enumerate(images):
        logger.status(f"Building Face Model for {images_names[i]}...")
        face = build_face_model(image,str(i),save_model=False)
        if isinstance(face, str):
            # logger.error(f"No faces found in {images_names[i]}, skipping")
            continue
        if shape_check:
            if i == 0:
                embedding_shape = face.embedding.shape
            elif face.embedding.shape != embedding_shape:
                logger.error(f"Embedding Shape Mismatch for {images_names[i]}, skipping")
                continue
        faces.append(face)
        embeddings.append(face.embedding)
    if len(faces) > 0:
        # if shape_check:
        #     embedding_shape = embeddings[0].shape
        #     for embedding in embeddings:
        #         if embedding.shape != embedding_shape:
        #             logger.error("Embedding Shape Mismatch")
        #             break
        compute_method_name = "Mean" if compute_method == 0 else "Median" if compute_method == 1 else "Mode"
        logger.status(f"Blending with Compute Method {compute_method_name}...")
        blended_embedding = np.mean(embeddings, axis=0) if compute_method == 0 else np.median(embeddings, axis=0) if compute_method == 1 else stats.mode(embeddings, axis=0)[0].astype(np.float32)
        blended_face = Face(
            bbox=faces[0].bbox,
            kps=faces[0].kps,
            det_score=faces[0].det_score,
            landmark_3d_68=faces[0].landmark_3d_68,
            pose=faces[0].pose,
            landmark_2d_106=faces[0].landmark_2d_106,
            embedding=blended_embedding,
            gender=faces[0].gender,
            age=faces[0].age
        )
        if blended_face is not None:
            face_model_path = os.path.join(FACE_MODELS_PATH, name + ".safetensors")
            save_face_model(blended_face,face_model_path)
            logger.status("--Done!--")
            done_msg = f"Face model has been saved to '{face_model_path}'"
            logger.status(done_msg)
            return done_msg
        else:
            no_face_msg = "Something went wrong, please try another set of images"
            logger.error(no_face_msg)
            return no_face_msg
    return "No faces found"


def operate(
        source_img,
        target_img,
        target_img_orig,
        model,
        source_faces_index,
        faces_index,
        source_faces,
        target_faces,
        gender_source,
        gender_target,
        source_face,
        wrong_gender,
        source_age,
        source_gender,
        output,
        swapped,
        mask_face,
        entire_mask_image,
        enhancement_options,
        detection_options,
    ):
    result = target_img
    face_swapper = getFaceSwapModel(model)

    source_face_idx = 0

    for face_num in faces_index:
        if check_process_halt():
            return result_image, [], 0
        if len(source_faces_index) > 1 and source_face_idx > 0:
            logger.status("Detecting Source Face, Index = %s", source_faces_index[source_face_idx])
            source_face, wrong_gender, source_age, source_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
            if source_age != "None" or source_gender != "None":
                logger.status("Detected: -%s- y.o. %s", source_age, source_gender)

            output_info = f"SourceFaceIndex={source_faces_index[source_face_idx]};Age={source_age};Gender={source_gender}\n"
            output.append(output_info)

        source_face_idx += 1

        if source_face is not None and wrong_gender == 0:
            logger.status("Detecting Target Face, Index = %s", face_num)
            target_face, wrong_gender, target_age, target_gender = get_face_single(target_img, target_faces, face_index=face_num, gender_target=gender_target, det_thresh=detection_options.det_thresh, det_maxnum=detection_options.det_maxnum)
            if target_age != "None" or target_gender != "None":
                logger.status("Detected: -%s- y.o. %s", target_age, target_gender)

                output_info = f"TargetFaceIndex={face_num};Age={target_age};Gender={target_gender}\n"
                output.append(output_info)
                
                if target_face is not None and wrong_gender == 0:
                    logger.status("Swapping Source into Target")
                    swapped_image = face_swapper.get(result, target_face, source_face)
                                            
                    if mask_face:
                        result = apply_face_mask(swapped_image=swapped_image,target_image=result,target_face=target_face,entire_mask_image=entire_mask_image)
                    else:
                        result = swapped_image
                    swapped += 1
                
                elif wrong_gender == 1:
                    wrong_gender = 0
                    
                    if source_face_idx == len(source_faces_index):
                        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                        
                        if enhancement_options is not None and len(source_faces_index) > 1:
                            result_image = enhance_image(result_image, enhancement_options)
                        
                        return result_image, output, swapped
                
                else:
                    logger.status(f"No target face found for {face_num}")
            
            elif wrong_gender == 1:
                wrong_gender = 0
                
                if source_face_idx == len(source_faces_index):
                    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                    
                    if enhancement_options is not None and len(source_faces_index) > 1:
                        result_image = enhance_image(result_image, enhancement_options)
                    
                    return result_image, output, swapped
            
            else:
                logger.status(f"No source face found for face number {source_face_idx}.")

                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                
                if enhancement_options is not None and swapped > 0:
                    if mask_face and entire_mask_image is not None:
                        result_image = enhance_image_and_mask(result_image, enhancement_options,Image.fromarray(target_img_orig),Image.fromarray(entire_mask_image).convert("L"))    
                    else:    
                        result_image = enhance_image(result_image, enhancement_options)
                elif mask_face and entire_mask_image is not None and swapped > 0:
                    result_image = Image.composite(result_image,Image.fromarray(target_img_orig),Image.fromarray(entire_mask_image).convert("L"))
               
        else:
            logger.status("No source face(s) in the provided Index")
    else:
        logger.status("No source face(s) found")
    
    return result_image, output, swapped



# def apply_face_mask(swapped_image:np.ndarray,target_image:np.ndarray,target_face,entire_mask_image:np.array)->np.ndarray:
#     logger.status("Masking Face")
#     mask_generator =  BiSeNetMaskGenerator()
#     face = Face(target_image,Rect.from_ndarray(np.array(target_face.bbox)),1.6,512,"")
#     face_image = np.array(face.image)
#     process_face_image(face)
#     face_area_on_image = face.face_area_on_image
#     mask = mask_generator.generate_mask(face_image,face_area_on_image=face_area_on_image,affected_areas=["Face"],mask_size=0,use_minimal_area=True)
#     mask = cv2.blur(mask, (12, 12))
#     """entire_mask_image = np.zeros_like(target_image)"""
#     larger_mask = cv2.resize(mask, dsize=(face.width, face.height))
#     entire_mask_image[
#         face.top : face.bottom,
#         face.left : face.right,
#     ] = larger_mask
   
   
#     result = Image.composite(Image.fromarray(swapped_image),Image.fromarray(target_image), Image.fromarray(entire_mask_image).convert("L"))
#     return np.array(result)


def correct_face_tilt(angle: float) -> bool:
        
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle > 40
def _dilate(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.dilate(arr, kernel, iterations=1)


def _erode(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.erode(arr, kernel, iterations=1)
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (0, 128, 128),
]


def color_generator(colors):
    while True:
        for color in colors:
            yield color


color_iter = color_generator(colors)
def process_face_image(
        face: Face,
        **kwargs,
    ) -> Image:
        image = np.array(face.image)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), next(color_iter), -1)
        l, t, r, b = face.face_area_on_image
        cv2.rectangle(overlay, (l, t), (r, b), (0, 0, 0), 10)
        if face.landmarks_on_image is not None:
            for landmark in face.landmarks_on_image:
                cv2.circle(overlay, (int(landmark.x), int(landmark.y)), 6, (0, 0, 0), 10)
        alpha = 0.3
        output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return Image.fromarray(output)
def dilate_erode(img: Image.Image, value: int) -> Image.Image:
    """
    The dilate_erode function takes an image and a value.
    If the value is positive, it dilates the image by that amount.
    If the value is negative, it erodes the image by that amount.

    Parameters
    ----------
        img: PIL.Image.Image
            the image to be processed
        value: int
            kernel size of dilation or erosion

    Returns
    -------
        PIL.Image.Image
            The image that has been dilated or eroded
    """
    if value == 0:
        return img

    arr = np.array(img)
    arr = _dilate(arr, value) if value > 0 else _erode(arr, -value)

    return Image.fromarray(arr)

def mask_to_pil(masks, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]

def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks

def rotate_image(image: Image, angle: float) -> Image:
    if angle == 0:
        return image
    return Image.fromarray(rotate_array(np.array(image), angle))


def rotate_array(image: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))
