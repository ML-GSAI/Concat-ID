import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import  decord       
from decord import VideoReader
import torchvision.transforms as TT
import json
decord.bridge.set_bridge("torch")
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import math
import random
from typing import Tuple, Optional


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

def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]


class FaceJsonMultiPerSFTDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dir, height, width, fps=16, max_num_frames=49, skip_frms_num=3):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(FaceJsonMultiPerSFTDataset, self).__init__()

        self.video_size = (height, width)
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        self.video_paths = []
        self.captions = []
        self.face_images = []
        self.mask_images = []
        
        self.transform = TT.Compose([TT.ToTensor()])

        if os.path.isdir(data_dir):
            for root, dirnames, filenames in os.walk(data_dir):
                for filename in filenames:
                    if filename.endswith(".mp4"):
                        video_path = os.path.join(root, filename)
                        self.video_paths.append(video_path)

                        caption_path = video_path.replace(".mp4", ".txt").replace("videos", "labels")
                        if os.path.exists(caption_path):
                            caption = open(caption_path, "r").read().splitlines()[0]
                        else:
                            caption = ""
                        self.captions.append(caption)

                        image_path = video_path.replace(".mp4", "").replace("videos", "images")
                        if os.path.exists(image_path):
                            self.face_images.append(image_path)
                        else:
                            raise Exception(f"face images of {video_path} does't exist")
        elif data_dir.endswith(".json"):
            with open(data_dir) as file:
                meta_datas = json.load(file)
                for meta_data in meta_datas:
                    self.video_paths.append(meta_data['video_path'])
                    self.captions.append(meta_data['caption'])
                    self.face_images.append(meta_data['image_path'])
                    if 'mask_path' in meta_data.keys():
                        self.mask_images.append(meta_data['mask_path'])
        else:
            raise NotImplementedError()
                        
        print(f'The number of samples: {len(self.video_paths)}')
    
    def resize_pad_crop(self, arr, image_size):

            # resize arr to match its longside to target longside of image_size
            if arr.shape[2] / arr.shape[3] >= image_size[0] / image_size[1]: # h >= w
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
    
    def __getitem__(self, index):

        video_path = self.video_paths[index]
        vr = VideoReader(uri=video_path, height=-1, width=-1)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)

        if ori_vlen / actual_fps * self.fps > self.max_num_frames:
            
            num_frames = self.max_num_frames
            start = int(self.skip_frms_num)
            end = math.ceil(start + num_frames / self.fps * actual_fps)
            end_safty = min(end, int(ori_vlen - self.skip_frms_num))
            indices = np.arange(start, end, max((end - start) // num_frames, 1)).astype(int)
            temp_frms = vr.get_batch(np.arange(start, end_safty))
            assert temp_frms is not None
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            tensor_frms = tensor_frms[torch.tensor((indices - start).clip(0, len(tensor_frms) - 1).tolist())]
        else:
            if ori_vlen > self.max_num_frames:
                num_frames = self.max_num_frames
                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                indices = np.arange(start, end, max((end - start) // num_frames, 1)).astype(int)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                # a bug
                tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
            else:
                def nearest_smaller_4k_plus_1(n):
                    remainder = n % 4
                    if remainder == 0:
                        return n - 3
                    else:
                        return n - remainder + 1

                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                num_frames = nearest_smaller_4k_plus_1(end - start)  # 3D VAE requires the number of frames to be 4k+1
                end = int(start + num_frames)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms

        tensor_frms = pad_last_frame(
            tensor_frms, self.max_num_frames
        )  # the len of indices may be less than num_frames, due to round error
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")
        tensor_frms = (tensor_frms - 127.5) / 127.5
        
        if self.face_images[index]!="":
            if os.path.isdir(self.face_images[index]):
                face_image = [os.path.join(self.face_images[index], name) for name in os.listdir(self.face_images[index]) if name.endswith(('png', 'jpg', 'jpeg'))]
                face_image = self.transform(Image.open(np.random.choice(face_image)).convert('RGB')).unsqueeze(dim=0) #[1, C, H, W]
                face_image = self.resize_pad_crop(face_image, self.video_size)
                face_image = (face_image * 2.0 - 1.0).clip(-1,1)
            else:
                if '@@' not in self.face_images[index]:
                    face_image = Image.open(self.face_images[index]).convert('RGB')
                    # if len(self.mask_images) > 0 and self.mask_images[index]!='' and np.random.rand() >= 1:
                    if len(self.mask_images) > 0 and self.mask_images[index]!='' and np.random.rand() < 0.9:
                        mask_image = Image.open(self.mask_images[index]).convert('L')
                        face_image = cv2.bitwise_and(np.array(face_image), np.array(face_image), mask=np.array(mask_image))
                        face_image = Image.fromarray(face_image)
                    face_image = self.transform(face_image).unsqueeze(dim=0) #[1, C, H, W]
                    face_image = self.resize_pad_crop(face_image, self.video_size)
                    face_image = (face_image * 2.0 - 1.0).clip(-1,1)
                else:
                    image_paths = self.face_images[index].split('@@')
                    face_image = [self.transform(Image.open(image_path).convert('RGB')).unsqueeze(dim=0) for image_path in image_paths]
                    face_image = [self.resize_pad_crop(temp_face, self.video_size) for temp_face in face_image]
                    
                    face_image = torch.cat(face_image, dim=0) #[N, C, H, W]
                    
                    face_image = (face_image * 2.0 - 1.0).clip(-1,1)
        else:
          face_image = tensor_frms[0:1,:,:,:]   
          
          
        assert tensor_frms.shape[-2:]==face_image.shape[-2:]
        
             
        tensor_frms = tensor_frms.permute(1, 0, 2, 3) # [T, C, H, W] -> [C, T, H, W]
        face_image = face_image.permute(1, 0, 2, 3) # [T, C, H, W] -> [C, T, H, W]
        
        text = self.captions[index]
        
        # if random.random() < 0.1:
        #     text = ""
        
        item = {
            "mp4": tensor_frms,
            "txt": text,
            "num_frames": num_frames,
            "fps": self.fps,
            "face_image":face_image,
        }

        return item

    def __len__(self):
        return len(self.video_paths)

class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16),
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None, resume_dit_path=None, num_frames=None,
    ):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_path.append(dit_path)
            model_manager.load_models(model_path)
        else:
            model_path = model_path + dit_path.split(",")
            model_manager.load_models(model_path)
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

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

        original_linear_layer = self.pipe.denoising_model().time_projection[-1]

        self.pipe.denoising_model().time_projection_ref = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.Linear(original_linear_layer.in_features, original_linear_layer.out_features)
        )

        self.pipe.denoising_model().time_projection_ref.load_state_dict( self.pipe.denoising_model().time_projection.state_dict(), strict=True)
            
        self.pipe.denoising_model().forward = dit_forward.__get__(self.pipe.denoising_model()) # update the forward function
                            
        def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
            return (x * (1 + scale) + shift)
        def forward(self, x, context, t_mod, t_mod_ref, freqs):

            num_image_tokens = x.shape[1] // ( (num_frames - 1) // 4 + 1 + 1) # 1 for first frame, 1 for the ref image
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
    
        for block in self.pipe.denoising_model().blocks:
                     
            block.register_parameter( 'modulation_ref', torch.nn.Parameter(block.modulation.data.clone()) ) # add new parameters
            
            assert not block.modulation.data.is_set_to(block.modulation_ref.data), "Verification passed: No shared memory"
            assert block.modulation_ref.data.sum() == block.modulation.data.sum(), "Copy failure of Moulation weights"
            
            block.forward = forward.__get__(block) # update the forward function
        
        if resume_dit_path is not None:
            
            states = torch.load(resume_dit_path, map_location=torch.device('cpu'))

            if 'module' in list(states.keys()):           
                for name in list(states.keys())[:18]:
                    del states[name]
                
            missing_keys, unexpected_keys = self.pipe.dit.load_state_dict(states, strict=True)
                
            if missing_keys:
                raise Exception(f"Warning: Missing keys in state_dict: {missing_keys}")
            elif unexpected_keys:
                raise Exception(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
            else:
                print(f"Successfully loaded the model weights of dit from {resume_dit_path}")
    
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    
    def add_noise_to_first_frame(self, image):
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(self.device)
        sigma = torch.exp(sigma).to(image.dtype)
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image = image + image_noise
        return image
    
    def training_step(self, batch, batch_idx):
        
        self.pipe.device = self.device

        # Data
        text, video, face_image = batch["txt"], batch["mp4"], batch["face_image"]
        with torch.no_grad():
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device).contiguous()
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)

            image_emb = {}
        
        context_images = []
        for i in range(face_image.shape[2]):
            image = self.add_noise_to_first_frame(face_image[:,:,i:i+1,:,:]).contiguous()
            context_image = self.pipe.encode_video(image, **self.tiler_kwargs)
            context_images.append(context_image)
        context_images = torch.cat(context_images, dim=2)
         
        for i in range(context_images.shape[0]):
            if random.random() < 0.1:
                context_images[i] = torch.zeros_like(context_images[i])
        
        # Loss
        # self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        # noise_pred = self.pipe.denoising_model()(
        #     noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
        #     use_gradient_checkpointing=self.use_gradient_checkpointing,
        #     use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        # )
        noise_pred = self.pipe.denoising_model()(
            torch.cat([noisy_latents, context_images], dim=2), timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )[:,:, :-context_images.shape[2],...]

        orginal_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        # loss = orginal_loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", orginal_loss, prog_bar=True, logger=True)
        # self.log("train_loss", loss, prog_bar=True, logger=True)
        
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] 
        self.log("learning_rate", current_lr, prog_bar=True, logger=True)
    
        # return loss
        return orginal_loss
    
    def configure_optimizers(self):
        
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        
        # total steps for training  
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(1000, int(total_steps * 0.05))

        print(f'Total number of warmup steps: {warmup_steps}, Total number of training steps: {total_steps}')
        
        # initial learning rate
        initial_lr = 1e-7

        # Warm-up scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((step + 1) * (1.0 - initial_lr / self.learning_rate) / warmup_steps + (initial_lr / self.learning_rate), 1.0)
        )

        # Linear decay scheduler
        linear_decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max(total_steps - warmup_steps, 1)  
        )

        # Combine both schedulers
        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, linear_decay_scheduler],
            milestones=[warmup_steps]  
        )

        assert warmup_steps < total_steps, "Warmup steps must be less than total steps."
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': combined_scheduler,
                'interval': 'step',  
                'frequency': 1      
            }
        } 

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--resume_dit_path",
        type=str,
        default=None,
        help="Path to the checkpoint used to resume the DiT model.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set seed.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of batchsize.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--project_name",
        default="wan",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--save_train_steps",
        type=int,
        default=100,
        help="how many training steps to save checkpoints",
    )
    parser.add_argument(
        "--skip_frms_num",
        type=int,
        default=3,
        help="The number of frames to be skipped from both the beginning and the end of the video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="the target fps",
    )
      
    args = parser.parse_args()
    return args
    
def train(args):
    
    if args.seed is not None:
        pl.seed_everything(args.seed + int(os.environ.get("OMPI_COMM_WORLD_RANK", 0)), workers=True) # sets seeds for numpy, torch and python.random.
        
    dataset = FaceJsonMultiPerSFTDataset(
        args.dataset_path,
        height=args.height,
        width=args.width,
        fps=args.fps, 
        max_num_frames=args.num_frames,
        skip_frms_num=args.skip_frms_num
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        resume_dit_path=args.resume_dit_path,
        num_frames=args.num_frames,
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project=f"{args.project_name}", 
            name=f"{args.project_name}",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=os.environ.get("MLP_GPU", "auto"),
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(every_n_train_steps=args.save_train_steps, save_last=True, save_top_k=-1)], # 
        logger=logger,
        log_every_n_steps=50, num_nodes=int(os.environ.get("MLP_WORKER_NUM", 1)),
    )
    trainer.fit(model, dataloader)


def get_node_indices(num_nodes, num_gpus):
    """
    Creates ordered list of node indices for each process

    Args
        num_nodes: number of nodes
        num_gpus: number of gpus per node
    """
    # convert to int incase we are using string environmental variables
    num_nodes = int(num_nodes)
    num_gpus = int(num_gpus)

    node_indices = [0]
    for i in range(1, num_nodes*num_gpus):
        # use modulo to decide when to increment node index
        increment = i % num_gpus == 0
        node_indices += [node_indices[-1] + increment]
    # convert to string
    return [ str(x) for x in node_indices ]

if __name__ == '__main__':
    
    # https://github.com/Lightning-AI/pytorch-lightning/issues/13639#issuecomment-1184719803
    # Map MPI environment variables to those expected by DeepSpeed/PyTorch
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ["NODE_RANK"] = get_node_indices(os.environ.get("MLP_WORKER_NUM"), os.environ.get("MLP_GPU"))[int(os.environ.get("OMPI_COMM_WORLD_RANK"))]
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        # os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', "")
        # os.environ["MASTER_ADDR"] = os.getenv('MASTER_ADDR', 'localhost')
    else:
        # raise EnvironmentError(
        #     "MPI environment variables are not set. "
        #     "Ensure you are running the script with an MPI-compatible launcher."
        # )
        pass
    
    args = parse_args()

    train(args)
