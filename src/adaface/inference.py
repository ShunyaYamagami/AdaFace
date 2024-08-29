import net
import torch
import os
from face_alignment import align
import numpy as np


def load_pretrained_model(architecture='ir_50', pretrained_path='pretrained/adaface_ir50_ms1mv2.ckpt'):
    # load model and pretrained statedict
    model = net.build_model(architecture)
    statedict = torch.load(pretrained_path)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor
