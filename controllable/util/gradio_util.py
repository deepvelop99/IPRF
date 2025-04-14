from PIL import Image
import os
import subprocess 
import gradio as gr
import random
import shutil

def save_image(image, save_dir):
    if image:
        save_path = os.path.join(save_dir, image.name)
        image.save(save_path)
        
def Image_from_path(image_path):
    try:
        image = Image.open(image_path)
        return image
    except FileNotFoundError:
        return "Image not found. Please check the file path."

def update_and_display(selected_value):
    return selected_value

def update_scene_image(scene):
    scene_image_path = f"../opt/ckpt_svox2/llff/{scene}/test_renders_path/0040.png"
    return Image_from_path(scene_image_path)

def update_style_image(style_idx):
    style_image_path = f"../data/styles/{style_idx}.jpg"
    return Image_from_path(style_image_path)

def update_result_image(scene, style_idx):
    result_image_path = f"../opt/ctrl/results/llff/{scene}_{style_idx}_ctrl/3d_interpolate/0040.png"
    return Image_from_path(result_image_path)

def update_result_mp4(scene, style_idx):
    original_path = f"../opt/ctrl/results/llff/{scene}_{style_idx}_ctrl/3d_interpolate.mp4"
    copy_path = f"./3d_interpolate_{scene}_{style_idx}.mp4"
    
    if os.path.exists(original_path) and not os.path.exists(copy_path):
        shutil.copy(original_path, copy_path)
        
    return copy_path

def update_buttons(scene, style_idx):
    albedo_path, shading_path = mp4_path(scene, style_idx)
    return albedo_path, shading_path

def save_uploaded_image(uploaded_image, save_dir="../data/styles/"):
    new_style_idx = random.randint(1000, 9999)
    save_path = os.path.join(save_dir, f"{new_style_idx}.jpg")
    uploaded_image.save(save_path)
    return gr.update(value=f"{new_style_idx}")

def mp4_path(scene, style_idx):
    if not scene or not style_idx:
        return None, None

    albedo_src = f"../opt/ctrl/ckpts/llff/{scene}_{style_idx}_ctrl/only_albedo/test_renders_path.mp4"
    shading_src = f"../opt/ctrl/ckpts/llff/{scene}_{style_idx}_ctrl/only_shading/test_renders_path.mp4"

    albedo_dst = f"./albedo_{scene}_{style_idx}.mp4"
    shading_dst = f"./shading_{scene}_{style_idx}.mp4"

    if os.path.exists(albedo_src) and not os.path.exists(albedo_dst):
        shutil.copy(albedo_src, albedo_dst)
    if os.path.exists(shading_src) and not os.path.exists(shading_dst):
        shutil.copy(shading_src, shading_dst)

    return albedo_dst, shading_dst


def get_download_path(input_path):
    if os.path.exists(input_path):
        return input_path
    else:
        return None

def run_3d_interpolate_script(scene, style_idx, weight):
    script_path = "./controllable_gradio.sh"
    subprocess.run(
        ["bash", script_path, scene, style_idx, str(1-weight), str(weight)], 
        capture_output=True, 
        text=True
    )
    return update_result_image(scene, style_idx), update_result_mp4(scene, style_idx)

def run_plenoxels(scene, scene_type):
    script_path = "./try_svox2.sh"
    scene_type = "llff"
    subprocess.run(
        ["bash", script_path, scene_type, scene], 
        capture_output=False, 
        text=False
    )
    return update_scene_image(scene)

def run_albedo_optimization(scene, style_idx):
    script_path = "./optimize_only_albedo_shading.sh"
    subprocess.run(
        ["bash", script_path, scene, style_idx], 
        capture_output=False, 
        text=False
    )
    style_albedo_image, albedo_mp4_box = update_style_albedo_image(scene, style_idx)
    style_shading_image, shading_mp4_box = update_style_shading_image(scene, style_idx)
    return style_albedo_image, albedo_mp4_box, style_shading_image, shading_mp4_box

def update_style_albedo_image(scene, style_idx):
    albedo_image_path = f"../opt/ctrl/ckpts/llff/{scene}_{style_idx}_ctrl/only_albedo/test_renders_path/0040.png"
    albedo_mp4_path = f"../opt/ctrl/ckpts/llff/{scene}_{style_idx}_ctrl/only_albedo/test_renders_path.mp4"
    return Image_from_path(albedo_image_path), albedo_mp4_path

def update_style_shading_image(scene, style_idx):
    shading_image_path = f"../opt/ctrl/ckpts/llff/{scene}_{style_idx}_ctrl/only_shading/test_renders_path/0040.png"
    shading_mp4_path = f"../opt/ctrl/ckpts/llff/{scene}_{style_idx}_ctrl/only_shading/test_renders_path.mp4"
    return Image_from_path(shading_image_path), shading_mp4_path
