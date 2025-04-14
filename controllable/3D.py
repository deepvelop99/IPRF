import numpy as np
import gradio as gr
from util.gradio_util import *
import os
import shutil

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "INTEL"
_ = np.dot(np.zeros((1, 1)), np.zeros((1, 1)))

with gr.Blocks() as demo:
    # Tab 1: 3D Interpolate
    with gr.Tab("3D Interpolate"):
        with gr.Row():
            with gr.Column():                
                with gr.Tab("llff"):
                    scene = gr.Radio(
                        choices=["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"], 
                        label="Select Scene",
                        value=None
                    )
                    scene_image = gr.Image(label="Scene Image", interactive=False, width=700, height=363)
                    scene.change(
                        run_plenoxels, 
                        inputs=[scene, scene],
                        outputs=scene_image
                    )
            with gr.Tab("Select Image"):
                style_idx = gr.Dropdown(
                        ["10", "14", "28", "120", "131", "137"],
                        value=None,
                        label="Select example style image",
                        interactive=True,
                        allow_custom_value=True
                    )
                style_image = gr.Image(label="Style Image", interactive=False, width=700, height=400)
                style_idx.change(
                    update_style_image, 
                    inputs=[style_idx],
                    outputs=[style_image]
                )
            with gr.Tab("Custom Image"):
              custom_style_image = gr.Image(label="Uplload your custom style image", type="pil", width=700, height=505)
              custom_style_image.change(
                  save_uploaded_image, 
                  inputs=[custom_style_image],
                  outputs=[style_idx]
              )

        run_custom_button = gr.Button("Run Albedo Optimization")
            
        with gr.Row():
            albedo_mp4_box = gr.Textbox(visible=False)
            shading_mp4_box = gr.Textbox(visible=False)
            style_albedo_image = gr.Image(label="Styled albedo image", interactive=False, visible=True)
            style_shading_image = gr.Image(label="Styled shading image", interactive=False, visible=True)
            run_custom_button.click(
                run_albedo_optimization, 
                inputs=[scene, style_idx], 
                outputs=[style_albedo_image, albedo_mp4_box, style_shading_image, shading_mp4_box]
            )
        
        with gr.Row():
            albedo_mp4_download = gr.DownloadButton("Download styled_albedo.mp4")
            shading_mp4_download = gr.DownloadButton("Download styled_shading.mp4")

            scene.change(
                lambda s, idx: update_buttons(s, idx),
                inputs=[scene, style_idx],
                outputs=[albedo_mp4_download, shading_mp4_download]
            )

            style_idx.change(
                lambda s, idx: update_buttons(s, idx),
                inputs=[scene, style_idx],
                outputs=[albedo_mp4_download, shading_mp4_download]
            )
        with gr.Accordion("Controllable Weight", open=True):
            slider = gr.Slider(
                0, 1,
                value=0.5,
                step=0.1,
                interactive=True,
                label="Optimized albedo of style and shading of content",
            )
        run_3d_interpolate_button = gr.Button("3D Interpolate")
        
        render_image = gr.Image(label="Result Image", interactive=False)
        
        result_mp4_download_button = gr.DownloadButton("Download 3D Interpolation result")
        run_3d_interpolate_button.click(
            run_3d_interpolate_script, 
            inputs=[scene, style_idx, slider],
            outputs=[render_image, result_mp4_download_button]
        )

if __name__ == "__main__":
    demo.launch(share=True)
