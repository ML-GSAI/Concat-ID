import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio

import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as TT


from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from PIL import Image


def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
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

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr

def resize_pad_crop(arr, image_size):

    # resize arr to match its longside to target longside of image_size
    if arr.shape[2] / arr.shape[3] >= image_size[0] / image_size[1]: 
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
        

    # pad arr to image_size
    h, w = arr.shape[2], arr.shape[3]
    # arr = arr.squeeze(0)

    delta_h = image_size[0] - h
    delta_w = image_size[1] - w

    delta_h = max(delta_h, 0)
    delta_w = max(delta_w, 0)

    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    arr = torch.nn.functional.pad(arr, (left, right, top, bottom), mode='constant',value=0.0)
    
    # center crop
    h, w = arr.shape[2], arr.shape[3]
    delta_h = max(0, h - image_size[0])
    delta_w = max(0, w - image_size[1])
    
    top, left = delta_h // 2, delta_w // 2
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])

    return arr

import insightface
from insightface.app import FaceAnalysis 
from insightface.data import get_image as ins_get_image
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation  
from insightface.utils import face_align
import cv2
       
def get_face_image(frame, app, model, image_processor):
    if isinstance(frame, Image.Image):
        frame = np.array(frame)
    print('Starting detect face')
    with torch.autocast("cuda", enabled=False):
        # faces = app.get(frame)
        faces = app.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if len(faces)>0:
        faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[::-1] 
        face_info = faces[0] # use the biggest face
        face = face_align.norm_crop(frame, face_info['kps'], image_size=112 * 4) # align face
    else:
        print('*'*10+'InsightFace did not detect a face, so the original image will be used.'+'*'*10)
        face = frame
        
    face_image = Image.fromarray(face)
    unmasked_image = Image.fromarray(face)
    
    # run inference on image
    inputs = image_processor(images=face_image, return_tensors="pt").to(model.device)
    with torch.autocast("cuda", enabled=False):
        outputs = model(**inputs)
    
    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

    # resize output to match input image dimensions
    upsampled_logits = torch.nn.functional.interpolate(logits,
                    size=face_image.size[::-1], # H x W
                    mode='bilinear',
                    align_corners=False)

    # get label masks
    if len(upsampled_logits) > 0:
        labels = upsampled_logits.argmax(dim=1)[0]
        bg_label = [0, 14, 15, 16, 18] # background, hat, ear_r, neck_l, neck, cloth
        fg_mask = ~sum(labels == i for i in bg_label).bool() 
        if fg_mask.sum() >= 0.4 * fg_mask.shape[0] * fg_mask.shape[1]: 
            # continue
            face_image = (np.array(face_image) * fg_mask[:,:,np.newaxis].cpu().numpy()).astype(np.uint8) 
            face_image = Image.fromarray(face_image)
    return face_image, unmasked_image
                        
def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()

    if args.input_type == "cli":
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("rank and world_size", rank, world_size)
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        raise NotImplementedError

    image_size = [480, 720]

    if args.image2video or args.contextimage2video:
        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        transform = TT.Compose(chained_trainsforms)

    sample_func = model.sample
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    device = model.device

    # convenience expression for automatically determining device
    seg_image_processor = SegformerImageProcessor.from_pretrained("./models/jonathandinu/face-parsing")
    segmodel = SegformerForSemanticSegmentation.from_pretrained("./models/jonathandinu/face-parsing")
    segmodel = segmodel.to('cuda')
    app = FaceAnalysis(name=os.path.join(os.getcwd(), "models/antelopev2"), allowed_modules=['detection'],providers=['CUDAExecutionProvider'],
                       provider_options=[{"device_id": '0'}])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    with torch.no_grad():
        for text, cnt in tqdm(data_iter):
            if args.image2video:
                text, image_path = text.split("@@")
                assert os.path.exists(image_path), image_path
                image = Image.open(image_path).convert("RGB")
                image = transform(image).unsqueeze(0).to("cuda")
                image = resize_and_pad_for_rectangle(image, image_size)
                image = (image * 2.0 - 1.0).clip(-1,1)
                org_image = image.detach().cpu().clone()
                image = image.unsqueeze(2).to(torch.bfloat16)
                image = model.encode_first_stage(image, None)
                image = image.permute(0, 2, 1, 3, 4).contiguous()
                pad_shape = (image.shape[0], T - 1, C, H // F, W // F)
                image = torch.concat([image, torch.zeros(pad_shape).to(image.device).to(image.dtype)], dim=1)
            else:
                image = None
                
            if args.contextimage2video:
                temp_text = text.split("@@")
                text = temp_text[0]
                context_images = []
                org_images = []
                unmasked_images = []
                for image_path in temp_text[1:]:
                    assert os.path.exists(image_path), image_path
                    image = Image.open(image_path).convert("RGB")
                    ####crop and parser the face 
                    image, unmask_image = get_face_image(image, app, segmodel, seg_image_processor)
                    ####
                    image = transform(image).unsqueeze(0).to("cuda")
                    image = resize_pad_crop(image, image_size)
                    image = (image * 2.0 - 1.0).clip(-1,1)
                    org_image = image.detach().cpu().clone()
                    image = image.unsqueeze(2).to(torch.bfloat16)
                    image = model.encode_first_stage(image, None)
                    context_image = image.permute(0, 2, 1, 3, 4).contiguous()
                    context_images.append(context_image)
                    org_images.append(org_image)
                    unmasked_images.append(unmask_image)
                context_image = torch.cat(context_images, dim=1)
                org_image = torch.cat(org_images, dim=0)
            else:
                context_image = None

            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }
            model.to(device)
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            if args.image2video and image is not None:
                c["concat"] = image
                uc["concat"] = image

            for index in range(args.batch_size):
                # reload model on GPU
                model.to(device)
                model.first_stage_model.to('cpu')
                model.conditioner.to('cpu')
                samples_z = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                    context_image=context_image,
                )
                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

                # Unload the model from GPU to save GPU memory
                model.to("cpu")
                torch.cuda.empty_cache()
                first_stage_model = model.first_stage_model
                first_stage_model = first_stage_model.to(device)

                latent = 1.0 / model.scale_factor * samples_z

                # Decode latent serial to save GPU memory
                recons = []
                loop_num = (T - 1) // 2
                for i in range(loop_num):
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3
                    if i == loop_num - 1:
                        clear_fake_cp_cache = True
                    else:
                        clear_fake_cp_cache = False
                    with torch.no_grad():
                        recon = first_stage_model.decode(
                            latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                        )

                    recons.append(recon)

                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                save_path = os.path.join(
                    args.output_dir, str(cnt) + "_" + text.replace(" ", "_").replace("/", "")[:120], str(index)
                )
                if mpu.get_model_parallel_rank() == 0:
                    save_video_as_grid_and_mp4(samples, save_path, fps=args.sampling_fps)
                    for i in range(org_image.shape[0]):
                        now_save_path = os.path.join(save_path, f"{i:06d}.png")
                        Image.fromarray(((org_image[i,:,:,:].permute(1,2,0) + 1 ) / 2 * 255).clip(0,255).detach().cpu().numpy().astype(np.uint8)).save(now_save_path)
                        unmasked_images[i].save(os.path.join(save_path, f"unmasked_{i:06d}.png"))



if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    print(args_list, args)

    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    sampling_main(args, model_cls=SATVideoDiffusionEngine)
