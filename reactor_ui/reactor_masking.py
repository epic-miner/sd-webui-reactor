
import gradio as gr
from scripts.reactor_swapper import MaskOption

# TAB MASKING
def show():
    with gr.Tab("Masking"):
        save_face_mask = gr.Checkbox(False, label="Save Face Mask", info="Save the face mask as a separate image with alpha transparency.")
        use_minimal_area = gr.Checkbox(MaskOption.DEFAULT_USE_MINIMAL_AREA, label="Use Minimal Area", info="Use the least amount of area for the mask as possible. This is good for multiple faces that are close together or for preserving the most of the surrounding image.")
        
        mask_areas = gr.CheckboxGroup(
            label="Mask areas", choices=["Face", "Hair", "Hat", "Neck"], type="value", value= MaskOption.DEFAULT_FACE_AREAS
        )
        face_size = gr.Radio(
            label = "Face Size", choices = [512,256,128],value=MaskOption.DEFAULT_FACE_SIZE,type="value", info="Size of the masked area. Use larger numbers if the face is expected to be large, smaller if small. Default is 512."
        )
        mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=12,info="The number of pixels from the outer edge of the mask to blur.")

        mask_vignette_fallback_threshold = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            step=0.01,
            value=MaskOption.DEFAULT_VIGNETTE_THRESHOLD,
            label="Vignette fallback threshold",
            info="Switch to a rectangular vignette mask when masked area is only this specified percentage of Face Size."
        )
    return save_face_mask, use_minimal_area, mask_areas, face_size, mask_blur, mask_vignette_fallback_threshold
