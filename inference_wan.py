import torch
from diffsynth import ModelManager, save_video, VideoData
from pipelines.wan_video import WanVideoPipeline
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as huggingface_snapshot_download
import types

from PIL import Image
import torch
from torchvision import transforms as TT
from torchvision.transforms.functional import resize, InterpolationMode

import insightface
from insightface.app import FaceAnalysis 
from insightface.data import get_image as ins_get_image
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation  
from insightface.utils import face_align

import cv2
import os 
import numpy as np
import argparse
import random

class ImageProcessor:
    def __init__(self):
        pass

    def resize_pad_crop(self, image: Image.Image, image_size: tuple) -> Image.Image:
        
        to_tensor = TT.ToTensor()
        arr = to_tensor(image).unsqueeze(0)  # Shape: [1, C, H, W]

        # Step 1: Resize arr to match its long side to target long side of image_size
        if arr.shape[2] / arr.shape[3] >= image_size[0] / image_size[1]:  # h/w >= target_h/target_w
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        # Step 2: Pad arr to image_size
        h, w = arr.shape[2], arr.shape[3]
        delta_h = image_size[0] - h
        delta_w = image_size[1] - w

        delta_h = max(delta_h, 0)
        delta_w = max(delta_w, 0)

        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        arr = torch.nn.functional.pad(arr, (left, right, top, bottom), mode='constant', value=0.0)

        # Step 3: Center crop to image_size
        h, w = arr.shape[2], arr.shape[3]
        delta_h = max(0, h - image_size[0])
        delta_w = max(0, w - image_size[1])

        top, left = delta_h // 2, delta_w // 2
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])

        arr = arr.squeeze(0)  
        to_image = TT.ToPILImage()
        return to_image(arr)

def get_face_image(frame, app, model, image_processor):
    if isinstance(frame, Image.Image):
        frame = np.array(frame)
    print('Starting detect face')
    with torch.autocast("cuda", enabled=False):
        faces = app.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if len(faces)>0:
        faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[::-1] 
        face_info = faces[0] # use the biggest face
        face = face_align.norm_crop(frame, face_info['kps'], image_size=112 * 4) # align face
    else:
        print('*'*10+'InsightFace did not detect a face, so the original image will be used.'+'*'*10)
        face = frame
        
    unmasked_face_image = Image.fromarray(face)
    
    # run inference on image
    inputs = image_processor(images=unmasked_face_image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

    # resize output to match input image dimensions
    upsampled_logits = torch.nn.functional.interpolate(logits,
                    size=unmasked_face_image.size[::-1], # H x W
                    mode='bilinear',
                    align_corners=False)

    # get label masks
    if len(upsampled_logits) > 0:
        labels = upsampled_logits.argmax(dim=1)[0]
        bg_label = [0, 14, 15, 16, 18] # background, hat, ear_r, neck_l, neck, cloth
        fg_mask = ~sum(labels == i for i in bg_label).bool() 
        if fg_mask.sum() >= 0.4 * fg_mask.shape[0] * fg_mask.shape[1]: 
            # continue
            face_image = (np.array(unmasked_face_image) * fg_mask[:,:,np.newaxis].cpu().numpy()).astype(np.uint8) 
            face_image = Image.fromarray(face_image)
    return face_image, unmasked_face_image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos by Concat-ID"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="How many frames to sample from a video. The number should be 4n+1"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output',
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="...",
        help="The negative prompt to generate the video from.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        required=True,
        help="The reference images used by Concat-ID.")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="The sampling steps.")
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--use_tiled", 
        action="store_true",  
        help="Using tiled for saving GPU memory." 
    )

    args = parser.parse_args()

    return args

args = parse_args()
if args.seed < 0:
    args.seed = random.randint(0, 2**8 - 1)
    print(f'Set seed {args.seed}')
        
# Download models
wan_model_path = "models/Wan-AI/Wan2.1-T2V-1.3B"
concat_id_path = "models/Concat-ID-Wan"
segformer_model_path = "models/jonathandinu/face-parsing"

snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir=wan_model_path)
huggingface_snapshot_download(repo_id="jonathandinu/face-parsing", local_dir=segformer_model_path)  
huggingface_snapshot_download(repo_id="yongzhong/Concat-ID-Wan", local_dir=concat_id_path)  

# load models
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    f"{wan_model_path}/diffusion_pytorch_model.safetensors",
    f"{wan_model_path}/models_t5_umt5-xxl-enc-bf16.pth",
    f"{wan_model_path}/Wan2.1_VAE.pth",
])

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = WanVideoPipeline.from_model_manager(model_manager, device=device)

# Monkey Patching
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)
def forward(self, x, context, t_mod, freqs):
    # msa: multi-head self-attention  mlp: multi-layer perceptron
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
    input_x = modulate(self.norm1(x), shift_msa, scale_msa)
    x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
    # x = x + self.cross_attn(self.norm3(x), context)
    num_image_tokens = x.shape[1] // ( (args.num_frames - 1) // pipe.dit.patch_size[0] // 4 + 1 + 1) # 1 for first frame, 1 for the ref image
    x[:, :-num_image_tokens, :] = x[:, :-num_image_tokens,:] + self.cross_attn(self.norm3(x[:, :-num_image_tokens,:]), context)
    input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
    x = self.gate(x, gate_mlp, self.ffn(input_x))
    return x

for block in pipe.dit.blocks:
    block.forward = types.MethodType(forward, block)

states = torch.load(f"{concat_id_path}/first_stage.pt", map_location=torch.device('cpu'))
pipe.dit.load_state_dict(states, strict=True)
pipe.enable_vram_management(num_persistent_param_in_dit=None)

seg_image_processor = SegformerImageProcessor.from_pretrained(segformer_model_path)
segmodel = SegformerForSemanticSegmentation.from_pretrained(segformer_model_path)
segmodel = segmodel.to(device)
app = FaceAnalysis(name=f"{os.path.join(os.path.dirname(__file__), concat_id_path)}/antelopev2", allowed_modules=['detection'],providers=['CUDAExecutionProvider'],
                    provider_options=[{"device_id": '0'}])
app.prepare(ctx_id=0, det_size=(640, 640))

# generate videos
processor = ImageProcessor() 

os.makedirs(f"{args.output_dir}", exist_ok=True)

# target_size = (480, 832)
target_size = (args.height, args.width)

processor = ImageProcessor()

face_image = Image.open(args.image_path).convert("RGB")
face_image, unmasked_face_image = get_face_image(face_image, app, segmodel, seg_image_processor)

face_image = processor.resize_pad_crop(face_image, target_size)

video = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    face_image=face_image,
    num_inference_steps=args.num_inference_steps,
    num_frames=args.num_frames, 
    cfg_scale=args.cfg_scale,
    seed=args.seed, tiled=args.use_tiled
)

face_image.save(f"{args.output_dir}/ref_face.png")
unmasked_face_image.save(f"{args.output_dir}/ref_face_unmasked.png")
save_video(video, f"{args.output_dir}/output.mp4", fps=16, quality=5)