import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
import imageio
import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as TT
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu
from diffusion_video import SATVideoDiffusionEngine
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import gradio as gr
from arguments import get_args

# Load model once at the beginning
class ModelHandler:
    def __init__(self):
        self.model = None
        self.first_stage_model = None

    def load_model(self, args):
        if self.model is None:
            self.model = get_model(args, SATVideoDiffusionEngine)
            load_checkpoint(self.model, args)
            self.model.eval()
            self.first_stage_model = self.model.first_stage_model

    def get_model(self):
        return self.model

    def get_first_stage_model(self):
        return self.first_stage_model

model_handler = ModelHandler()

# Utility functions
def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N: List[int], T=None, device="cuda"):
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

def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5):
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

# Main inference function
def infer(prompt: str, sampling_num_frames: int, batch_size: int, latent_channels: int, sampling_fps: int):
    args = get_args(['--base', 'configs/cogvideox_2b.yaml', 'configs/inference.yaml', '--seed', '42'])
    args = argparse.Namespace(**vars(args))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    model_handler.load_model(args)
    model = model_handler.get_model()
    first_stage_model = model_handler.get_first_stage_model()

    image_size = [480, 720]
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    num_samples = [1]

    value_dict = {"prompt": prompt, "negative_prompt": "", "num_frames": torch.tensor(T).unsqueeze(0)}

    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
    )

    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=["txt"],
    )
    for k in c:
        if not k == "crossattn":
            c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

    samples_z = model.sample(
        c,
        uc=uc,
        batch_size=batch_size,
        shape=(T, C, H // F, W // F),
    )
    samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

    latent = 1.0 / model.scale_factor * samples_z

    # recons = []
    # loop_num = (T - 1) // 2
    # for i in range(loop_num):
    #     start_frame, end_frame = i * 2 + 1, i * 2 + 3 if i != 0 else (0, 3)
    #     recon = first_stage_model.decode(latent[:, :, start_frame:end_frame].contiguous())
    #     recons.append(recon)
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

    save_path = "outputs/demo"
    save_video_as_grid_and_mp4(samples, save_path, fps=sampling_fps)
    return os.path.join(save_path, "000000.mp4")

# Gradio Interface
def demo_interface(prompt):
    video_path = infer(prompt, 16, 1, 8, 5)
    return video_path

with gr.Blocks() as demo:
    gr.Markdown("""# RepVideo Gradio Demo
    Generate high-quality videos based on text prompts.
    """)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your text prompt here.", lines=3)
            generate_button = gr.Button("Generate Video")

        with gr.Column():
            video_output = gr.Video(label="Generated Video")

    generate_button.click(
        demo_interface,
        inputs=[prompt],
        outputs=[video_output]
    )

demo.launch(server_name="127.0.0.1", server_port=7860)
