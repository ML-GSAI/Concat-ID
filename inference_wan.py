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

from typing import Tuple, Optional
import imageio
from tqdm import tqdm

def save_gif(frames, save_path, fps, loop=0, quality=9):

    writer = imageio.get_writer(save_path, format='GIF', mode='I', loop=loop,
                                fps=fps, 
                                quality=quality,)
    
    for frame in tqdm(frames, desc="Saving GIF"):
        frame = np.array(frame)
        writer.append_data(frame)

    writer.close()
    

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


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

def dit_forward(self,
            x: torch.Tensor,
            timestep: torch.Tensor,
            context: torch.Tensor,
            clip_feature: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            use_gradient_checkpointing: bool = False,
            use_gradient_checkpointing_offload: bool = False,
            **kwargs,
            ):
    
    t = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, timestep))
    t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
    t_mod_ref = self.time_projection_ref(t).unflatten(1, (6, self.dim))
    
    context = self.text_embedding(context)
    
    if self.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = self.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    x, (f, h, w) = self.patchify(x)
    
    freqs = torch.cat([
        self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    freqs_ref = torch.cat([
        self.freqs[0][f - 1 : f].view(1, 1, 1, -1).expand(1, h, w, -1),
        self.freqs[1][0 : h].view(1, h, 1, -1).expand(1, h, w, -1),
        self.freqs[2][w : 2 * w].view(1, 1, w, -1).expand(1, h, w, -1)
    ], dim=-1).reshape(1 * h * w, 1, -1).to(x.device) 
    freqs[ - h * w:, ...]  = freqs_ref
    
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block in self.blocks:
        if self.training and use_gradient_checkpointing:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, t_mod_ref, freqs,
                        use_reentrant=False,
                    )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, t_mod_ref, freqs,
                    use_reentrant=False,
                )
        else:
            x = block(x, context, t_mod, t_mod_ref, freqs)

    x = self.head(x, t)
    x = self.unpatchify(x, (f, h, w))
    return x

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

def forward_adaln(self, x, context, t_mod, t_mod_ref, freqs):
    
    num_image_tokens = x.shape[1] // ( (args.num_frames - 1) // 4 + 1 + 1) # 1 for first frame, 1 for the ref image
    # msa: multi-head self-attention  mlp: multi-layer perceptron
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
    shift_msa_ref, scale_msa_ref, gate_msa_ref, shift_mlp_ref, scale_mlp_ref, gate_mlp_ref = (
        self.modulation_ref.to(dtype=t_mod_ref.dtype, device=t_mod_ref.device) + t_mod_ref).chunk(6, dim=1)
    
    norm_x = self.norm1(x)
    input_x = modulate(norm_x[:,:-num_image_tokens,:], shift_msa, scale_msa)
    input_x_ref = modulate(norm_x[:,-num_image_tokens:,:], shift_msa_ref, scale_msa_ref)
    
    attention_output = self.self_attn(torch.cat([input_x, input_x_ref], dim=1), freqs)
    
    output_main = self.gate(x[:, :-num_image_tokens, :], gate_msa, attention_output[:, :-num_image_tokens, :])
    output_ref = self.gate(x[:, -num_image_tokens:, :], gate_msa_ref, attention_output[:, -num_image_tokens:, :])
    x = torch.cat([output_main, output_ref], dim=1)

    output_cross = x[:, :-num_image_tokens, :] + self.cross_attn(self.norm3(x[:, :-num_image_tokens, :]), context)
    x = torch.cat([output_cross, x[:, -num_image_tokens:, :]], dim=1)

    norm_x = self.norm2(x)
    input_x = modulate(norm_x[:,:-num_image_tokens,:], shift_mlp, scale_mlp)
    input_x_ref = modulate(norm_x[:,-num_image_tokens:,:], shift_msa_ref, scale_msa_ref)
    ffn_output = self.ffn(torch.cat([input_x, input_x_ref], dim=1))

    output_main = self.gate(x[:, :-num_image_tokens, :], gate_mlp, ffn_output[:, :-num_image_tokens, :])
    output_ref = self.gate(x[:, -num_image_tokens:, :], gate_mlp_ref, ffn_output[:, -num_image_tokens:, :])
    x = torch.cat([output_main, output_ref], dim=1)
        
    return x

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
    parser.add_argument(
        "--use_adaln", 
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
if args.use_adaln:
    original_linear_layer = pipe.denoising_model().time_projection[-1]
    pipe.denoising_model().time_projection_ref = torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(original_linear_layer.in_features, original_linear_layer.out_features)
    )
    pipe.denoising_model().time_projection_ref.load_state_dict(pipe.denoising_model().time_projection.state_dict(), strict=True)   
    pipe.denoising_model().forward = dit_forward.__get__(pipe.denoising_model()) # update the forward function

    for block in pipe.denoising_model().blocks:            
        block.register_parameter( 'modulation_ref', torch.nn.Parameter(block.modulation.data.clone()) ) # add new parameters
        assert not block.modulation.data.is_set_to(block.modulation_ref.data), "Verification passed: No shared memory"
        assert block.modulation_ref.data.sum() == block.modulation.data.sum(), "Copy failure of Moulation weights"
        
        block.forward = forward_adaln.__get__(block) # update the forward function
else:
    for block in pipe.dit.blocks:
        block.forward = types.MethodType(forward, block)
        
if args.use_adaln:
    states = torch.load(f"{concat_id_path}/second_stage_adaln.pt", map_location=torch.device('cpu'))
else:
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
# save_gif(video, f"{args.output_dir}/output.gif", fps=16, quality=5)