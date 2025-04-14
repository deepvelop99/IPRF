import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from Network import DecScaleClampedIllumEdgeGuidedNetworkBatchNorm
from Utils import mor_utils

def load_model(model_path, device):
    net = DecScaleClampedIllumEdgeGuidedNetworkBatchNorm().to(device)
    support = mor_utils(device)
    net, _, _ = support.loadModels(net, model_path)
    net.eval()
    return net

def extract_IID(net, img_tensor, device):
    img_tensor = F.interpolate(img_tensor, size=(256, 256), mode='bilinear').to(device)
    return net(img_tensor)

def save_image(tensor, path):
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor.squeeze(0).cpu())
    img.save(path)

def save_images(content, style, styled, epoch_dir):
    save_image(content, os.path.join(epoch_dir, 'content.png'))
    save_image(style, os.path.join(epoch_dir, 'style.png'))
    save_image(styled, os.path.join(epoch_dir, 'styled.png'))

def create_directories(train_dir, epoch_id):
    result_epoch = os.path.join(train_dir, 'albedo_results', str(epoch_id))
    os.makedirs(result_epoch, exist_ok=True)
    return result_epoch

def save_iid_components(iid_results, result_epoch, prefix):
    keys_to_save = ["reflec_edge", "shading", "reflectance"]
    for key in keys_to_save:
        if key in iid_results:
            save_image(iid_results[key], os.path.join(result_epoch, f'{prefix}_{key}.png'))

def process_and_save_iid(net, images, result_epoch, prefixes, device):
    iid_results = {}
    for img, prefix in zip(images, prefixes):
        iid_result = extract_IID(net, img, device)
        save_iid_components(iid_result, result_epoch, prefix)
        iid_results[prefix] = iid_result
    return iid_results

def process_iid_and_save(net, content, style, styled, epoch_dir, device):
    images = [content, style, styled]
    prefixes = ["content", "style", "styled"]
    iid_results = process_and_save_iid(net, images, epoch_dir, prefixes, device)
    return iid_results