# Concat-ID
[![arXiv](https://img.shields.io/badge/arXiv-2503.13265-red.svg)](https://arxiv.org/abs/2503.14151)

[![deploy](https://img.shields.io/badge/Project%20Page-black)](https://ml-gsai.github.io/Concat-ID-demo/)

# 🔥 Latest News!
* May 7, 2025: We have released both the training and inference scripts for Concat-ID based on Wan2.1-T2V-1.3B, designed for single-identity scenarios. In this release, we introduce **an additional AdaLN module** to better process conditional reference images at different timesteps.

* Apr 26, 2025: We release the first-stage models of Concat-ID based on Wan2.1-T2V-1.3B for single-identity scenarios.

* March 18, 2025: We release the training and inference scripts and models of Concat-ID based on CogVideoX-5B for single-identity and two-identity scenarios.

# ToDo List
- [ ] Improving the model architecture and training strategy.
- [ ] Training the Wan2.1-T2V-1.3B on 720P videos.

Identity-preserving video generation is an interesting research topic, and therefore, we look forward to progressively improving this project.


# Concat-ID-Wan-AdaLN

We refine Concat-ID based on Wan2.1-T2V-1.3B for single identity scenarios, where we define an additional AdaLN module to better process conditional reference images at different timesteps. Note that the introduced AdaLN includes only about 14M parameters, which is negligible. We first trained Concat-ID-Wan on approximately 700,000 videos, and then fine-tuned it on approximately 200,000 cross-video pairs using a cosine similarity threshold of 0.87 to 0.97. We also release the training scripts based on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).


### 1. Install all dependencies

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
pip install insightface onnxruntime
```

### 2. Run scripts to generate videos
```python
python inference_wan.py --use_adaln  --image_path="{your_image_path}" --prompt="{your_prompt}" 
```

#### For instance:

``` python
python inference_wan.py --use_adaln  --image_path="examples/images/72_politicians_woman_Tulsi_Gabbard_2.png" --prompt="A woman with sun-kissed skin and a gentle smile pauses amidst the vibrant blooms of a flower field, kneeling gracefully to pick a wild daisy, while a soft breeze playfully lifts her hair. The warm, golden sunlight casts intricate patterns of light and shadow across the petals, creating a serene yet lively atmosphere. She momentarily glance toward the camera, her eyes reflecting a joyful curiosity, as the landscape around her buzzes with the soft hum of nature." --output_dir="output/1"
```

``` python
python inference_wan.py --use_adaln  --image_path="examples/images/73_politicians_woman_Harris_3.png" --prompt="A woman, with a sturdy build and focused demeanor, stands poised under the dappled sunlight, adjusting the settings on a sleek DSLR camera to capture the perfect shot of the vibrant, rolling hills that stretch out before her. The gentle breeze tousles her hair as she line up the frame, the golden hour casting warm shadows and creating a magical atmosphere. She take a deep breath, ready to capture the moment that will tell a story of tranquility and beauty, while her feet shift slightly to find the perfect angle, ensuring the composition is flawless." --output_dir="output/2"
```

``` python
python inference_wan.py --use_adaln  --image_path="examples/images/93_normal_woman_3.jpg" --prompt="A woman, exuding a vibrant energy, stands under the gentle glow of the afternoon sun, wearing large sunglasses and a wide-brimmed sunhat. The warm rays cast playful shadows on her face as she smile brightly. She casually twirl a colorful beach towel in one hand while adjusting her sunglasses with the other, surrounded by the serene ambiance of a sun-dappled garden." --output_dir="output/3"
```

<table style="width: 100%; border-collapse: collapse; text-align: center; border: 1px solid #ccc;">
  <tr>
    <th style="text-align: center;">
      <strong>Reference images</strong>
    </th>
    <th style="text-align: center;">
      <strong>Generated videos</strong>
    </th>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/cropped_images/4.png" alt="Image" style="height: 180px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
        <img src="examples/results/4.gif" alt="GIF" style="width: 400px;">
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/cropped_images/5.png" alt="Image" style="height: 180px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
        <img src="examples/results/5.gif" alt="GIF" style="width: 400px;">
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/cropped_images/6.png" alt="Image" style="height: 180px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
        <img src="examples/results/6.gif" alt="GIF" style="width: 400px;">
    </td>
  </tr>
</table>

### 3. Run scripts to training
Firstly, we need to install additional packages:
```
pip install peft lightning pandas
```


Then, we need to prepare the JSON file for training data (please refer to [this subsection](#preparing-the-dataset)).

Finally, we can run the training script `train_concat-id_wan2.1.py`. An example is shown below:

```python
python train_concat-id_wan2.1.py \
  --train_architecture full \
  --dataset_path "training-data.json" \
  --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --dit_path "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --max_epochs 1 \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --training_strategy deepspeed_stage_2 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --save_train_steps 1000 \
  --dataloader_num_workers 32 \
  --use_swanlab \
  --project_name train_concat-id_wan_adaln \
  --skip_frms_num 3 \
  --fps 16
```
For distributed training across multiple nodes and GPUs using MPI, please refer to [this issue](https://github.com/Lightning-AI/pytorch-lightning/issues/13639#issuecomment-1184719803). 

# Concat-ID-Wan

We release the first-stage model of Concat-ID based on Wan2.1-T2V-1.3B for single identity scenarios. We first train Concat-ID-Wan on approximately 600,000 49-frame videos, and then fine-tune it on approximately 700,000 81-frame videos.

### 1. Install all dependencies

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
pip install insightface onnxruntime
```

### 2. Run scripts to generate videos
```python
python inference_wan.py --image_path="{your_image_path}" --prompt="{your_prompt}"
```

#### For instance:

``` python
python inference_wan.py --image_path="examples/images/69_politicians_woman_Tulsi_Gabbard_4.png" --prompt="A woman, dressed in casual attire, sits by a sunlit window sketching in a notebook, pausing occasionally to look up with a playful grin. As sunlight filters through the sheer curtains, casting soft shadows across the room, the woman twirls a pencil absentmindedly before adding quick strokes to the page. The scene is alive with a sense of relaxed creativity, as the warm afternoon glow bathes the space in a gentle, inviting atmosphere." --output_dir="output/1/"
```

``` python
python inference_wan.py --image_path="examples/images/43_stars_man_Leonardo_DiCaprio_3.png" --prompt="A man with a distant look in his eyes stands alone on the deck of the Titanic, gripping the railing tightly. He watches the horizon, lost in thought, as the cold sea breeze brushes against his face. His mind drifts between excitement for the journey ahead and an unshakable sense of unease about what lies beneath the surface of the dark, mysterious waters." --output_dir="output/2/"
```

``` python
python inference_wan.py --image_path="examples/images/80_normal_man_5.jpg" --prompt="On a warm summer evening, a tall and athletic person is energetically playing a basketball game on the well-lit community court, dribbling the ball with expert precision, dodging imaginary opponents, and shooting hoops with impressive accuracy." --output_dir="output/3/"
```

<table style="width: 100%; border-collapse: collapse; text-align: center; border: 1px solid #ccc;">
  <tr>
    <th style="text-align: center;">
      <strong>Reference images</strong>
    </th>
    <th style="text-align: center;">
      <strong>Generated videos</strong>
    </th>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/cropped_images/1.png" alt="Image" style="height: 180px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
        <img src="examples/results/1.gif" alt="GIF" style="width: 400px;">
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/cropped_images/2.png" alt="Image" style="height: 180px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
        <img src="examples/results/2.gif" alt="GIF" style="width: 400px;">
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/cropped_images/3.png" alt="Image" style="height: 180px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
        <img src="examples/results/3.gif" alt="GIF" style="width: 400px;">
    </td>
  </tr>
</table>




# Concat-ID-CogVideo

This part, including both inference and training code, is based on [CogVideoX1.0 SAT](https://github.com/THUDM/CogVideo/releases/tag/v1.0) and [SAT](https://github.com/THUDM/SwissArmyTransformer).

## Inference Model

We only tested the inference script on the H800, and it required a maximum of 24 GB of GPU memory. However, this requirement may vary depending on different server environments.

### 1. Make sure you have installed all dependencies

```
pip install -r requirements.txt
```

### 2. Download the Model Weights

First, download the model weights from the SAT mirror to the project directory.

```
pip install modelscope
modelscope download --model 'yongzhong/Concat-ID' --local_dir 'models'
```
or

```
git lfs install
git clone https://www.modelscope.cn/yongzhong/Concat-ID.git models
```


If you want to download pre-training model in the first stage, just replace `Concat-ID` with `Concat-ID-pre-training` and replace `models` with `pre-training-models`. Note that the pre-training model offers better identity consistency but lower editability.

### 3. Modify `configs/inference/inference_single_identity.yaml` file.

```yaml
args:
  load: "./models/single-identity/" # Absolute path to transformer folder
  input_file: examples/single_identity.txt # Plain text file, can be edited
  output_dir: outputs/single-identity
```

+ Each line in `single_identity.txt` should follow the format `{prompt}@@{image_path}`, where `{image_path}` indicates the path to the reference image, and `{prompt}` indicates the corresponding prompt. If you are unsure how to write prompts, use [this code](https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py) to call an LLM for refinement.
+ To modify the output video location, change the `output_dir` parameter.

Modifying the configuration for multiple identities is similar to doing so for a single identity.

### 4. Run the Inference Code to Perform Inference
For single identity,
```
bash inference_single_identity.sh
```
For multiple identities,
```
bash inference_two_identities.sh
```

## 5. Fine-tuning the Model

### Preparing the Dataset


We need `training-data.json` and `validation-data.json` for fine-tuning and validation. Each element in the JSON file should be a dictionary formatted as follows:

```json
{
  "video_path": "{video_path}",
  "caption": "{prompt}",
  "image_path": "{image_path}"
}
```

+ `{video_path}` indicates the path of a training video.
+ `{prompt}` indicates the corresponding prompt.
+ `{image_path}` indicates the path of the corresponding reference image. For multiple identities, use `@@` to distinguish different reference images. 


For example, `training-data.json` with one training sample would look like this:
```json
[
  {
    "video_path": "/videos/training_1.mp4",
    "caption": "A man.",
    "image_path": "/images/reference_1.png"
  },
]
```

For multiple reference images:
```json
[
  {
    "video_path": "/videos/training_1.mp4",
    "caption": "Two people.",
    "image_path": "/images/reference_1.png@@/images/reference_2.png"
  },
]
```

### Modifying the Configuration File

We only tested full-parameter fine-tuning. 

We need to specify the paths of both training data and validation data in `configs/training/sft_single_identity.yaml` for single identity and in `configs/training/sft_two_identities.yaml` for multiple identities:

```yaml
train_data: [ "your_train_data_path" ]
valid_data: [ "your_val_data_path" ]  # Training and validation sets
```

For example:
```yaml
train_data: [ "/json/training-data.json" ]
valid_data: [ "/json/validation-data.json" ]  # Training and validation sets
```

### Fine-tuning and Validation

For single identity:

```
bash finetune_single_identity.sh # Multi GPUs
```

For multiple identities,
```
bash finetune_two_identities.sh # Multi GPUs
```

### 6. Converting to Huggingface Diffusers-compatible Weights

The SAT weight format differs from Huggingface's format. If you want to convert the weights, please run [this script](https://github.com/THUDM/CogVideo/blob/main/tools/convert_weight_sat2hf.py).

### Limitations
+ Due to limitations in the base model’s capabilities （i.e., CogVideoX-5B）, we do not compare our method with closed-source commercial tools.
+ Currently, we utilize VAEs solely as feature extractors, relying on the model’s inherent ability to process low-level features.
+ Similar to common video generation models, our approach faces challenges in preserving the integrity of human body structures, such as the number of fingers, when handling particularly complex motions.

### Related Links
We appreciate the following works:
+ [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
+ [Wan2.1](https://github.com/Wan-Video/Wan2.1)
+ [CogVideoX](https://github.com/THUDM/CogVideo/)
+ [ConsisID](https://github.com/PKU-YuanGroup/ConsisID)
+ [InsightFace](https://github.com/deepinsight/insightface)
+ [FaceParsing](https://huggingface.co/jonathandinu/face-parsing)