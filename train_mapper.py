import argparse
import os
import math
import torch
from torch.utils.data import DataLoader
from PIL import Image

from neural_models import Projector_no_noise
from dataset import *
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPModel, CLIPTokenizer
from diffusers.models import VQModel
from tqdm.auto import tqdm
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Projector Decoder.")
    parser.add_argument("--vqae_directory", type=str, default="models_weights/vq_model_weights", help="Directory of VQAE model weights")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32", help="Name of the CLIP model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="models_weights/projector_no_noise", help="Output directory for model weights")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--validation_epoch", type=int, default=1, help="Frequency of validation")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--checkpointing_steps", type=int, default=5, help="Steps interval for checkpointing")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--num_val_image_upload", type=int, default=8, help="Random seed for reproducibility")
    parser.add_argument("--image_size", type=int, default=256, help="Random seed for reproducibility")

    return parser.parse_args()


def check_gpus_and_nodes_accelerate(accelerator):
    num_processes = accelerator.num_processes
    num_machines = 1
    num_gpus_per_machine = accelerator.num_processes // num_machines

    print(f"Number of processes (total): {num_processes}")
    print(f"Number of machines (nodes): {num_machines}")
    print(f"Number of GPUs per machine: {num_gpus_per_machine}")


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder.get_text_features(
        text_input_ids,
        attention_mask=attention_mask,
    )
    return prompt_embeds


def train_decoder(args):
    vqae = VQModel.from_pretrained(args.vqae_directory)
    clip_text_model = CLIPModel.from_pretrained(args.clip_model_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_name)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb")
    accelerator.init_trackers(project_name="train_decoder_no_noise", init_kwargs={"wandb": {"name": f"training"}})
    if accelerator.is_main_process:
        check_gpus_and_nodes_accelerate(accelerator)

    set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    dim_shape = args.image_size // 8
    model = Projector_no_noise(input_dim=512, initial_shape=(64, dim_shape, dim_shape), out_channels=4)
    vqae.requires_grad_(False)
    clip_text_model.requires_grad_(False)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model.to(accelerator.device, dtype=weight_dtype)
    vqae.to(accelerator.device, dtype=weight_dtype)
    clip_text_model.to(accelerator.device, dtype=weight_dtype)
    vqae.eval()
    clip_text_model.eval()

    def vqvae_encoder(images):
        with torch.no_grad():
            encoded_image = vqae.encode(images)[0]
        return encoded_image

    def vqvae_decoder(latents):
        with torch.no_grad():
            decoded = vqae.decode(latents)[0]
            decoded_images = decoded.permute(0, 2, 3, 1).cpu().numpy()
            decoded_images = (decoded_images * 0.5 + 0.5) * 255
            decoded_images = decoded_images.astype('uint8')
            decoded_images_pil = [Image.fromarray(image) for image in decoded_images]
        return decoded_images_pil

    train_dataset, test_dataset = get_coco_dataloader(batch_size=64, image_size=256, max_size_test=16)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda examples: collate_fn(examples), drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=lambda examples: collate_fn(examples))

    params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)

    decoder, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    global_step = 0
    initial_global_step = 0
    checkpointing_steps = num_update_steps_per_epoch * args.checkpointing_steps

    progress_bar = tqdm(range(0, max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process)

    print("#### STARTING TRAINING ####")
    for epoch in range(num_train_epochs):
        decoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(decoder):
                optimizer.zero_grad()
                images = batch['images'].to(accelerator.device)
                clip_texts = encode_prompt(clip_text_model, batch['input_ids'], batch['attention_mask'])
                encoded_images = vqvae_encoder(images)
                pred_images_1 = decoder(clip_texts)
                loss = torch.mean((encoded_images.to(accelerator.device) - pred_images_1) ** 2)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, max_norm=1)
                optimizer.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        torch.save(model.state_dict(), f'{args.output_dir}/model_{epoch}.pth')
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(decoder)
        torch.save(model.state_dict(), f'{args.output_dir}/decoder_weights.pth')


if __name__ == '__main__':
    args = parse_args()
    train_decoder(args)