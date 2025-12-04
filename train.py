import argparse
import os
import math
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel
from dataset import *
from neural_models import Projector_noise
from accelerate.utils import set_seed
from transformers import CLIPModel, CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from diffusers.models import VQModel
from tqdm.auto import tqdm
import wandb
from metrics import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Image Feature Matching.")
    parser.add_argument("--output_save_dir", type=str, default="models_weights", help="Directory to save the output models")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of steps for the training")
    parser.add_argument("--dim_z", type=int, default=32, help="Dimension of the Gaussian noise vector z")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32", help="Name of the CLIP model")
    parser.add_argument("--sigma", type=float, default=0.7, help="Sigma value for noise")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default='models_weights/fm_noise_weights', help="Output directory for model weights")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--validation_epoch", type=int, default=1, help="Frequency of validation")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--num_upload_images", type=int, default=16, help="Number of images to upload during validation")
    parser.add_argument("--weights_projector_path", type=str, default="models_weights/projector_noise/decoder_weights.pth", help="Path to the projector model weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--num_diversity", type=int, default=5, help="Number of diversity samples")
    parser.add_argument("--image_size", type=int, default=256, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--load_z", type=str, default="models_weights/projector_noise/z_final.pth")
    parser.add_argument("--checkpointing_steps", type=int, default=2, help="Steps interval for checkpointing")
    parser.add_argument("--load_fm_weights_checkpoint", type=str, default='models_weights/fm_noise_weights/weights', help="Steps interval for checkpointing")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=int,
        default=20,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),)
    return parser.parse_args()

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

def train_image_fm(args):
    set_seed(args.seed)
    output_save_dir = args.output_save_dir
    num_steps = args.num_steps
    dim_z = args.dim_z
    clip_model_name = args.clip_model_name
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    clip_text_model = CLIPTextModel.from_pretrained(clip_model_name)
    sigma = args.sigma
    vqae = VQModel.from_pretrained(f"{output_save_dir}/vq_model_weights")
    dim_shape = args.image_size // 8
    decoder = Projector_noise(input_dim=512 + args.dim_z, initial_shape=(64, dim_shape, dim_shape), out_channels=4)
    decoder.load_state_dict(torch.load(args.weights_projector_path))
    model = UNet2DConditionModel(in_channels=4, out_channels=4, cross_attention_dim=512)

    # init accelerator
    gradient_accumulation_steps = args.gradient_accumulation_steps
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps,
                              log_with="wandb", )
    accelerator.init_trackers(
        project_name="train_fm_noise",
        init_kwargs={
            "wandb": {
                "name": f"training_fm_steps_{num_steps}_{sigma}",
            }
        }
    )

    # freeze
    clip_model.requires_grad_(False)
    vqae.requires_grad_(False)
    decoder.requires_grad_(False)
    clip_text_model.requires_grad_(False)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model.to(accelerator.device, dtype=weight_dtype)
    vqae.to(accelerator.device, dtype=weight_dtype)
    decoder.to(accelerator.device, dtype=weight_dtype)
    clip_model.to(accelerator.device, dtype=weight_dtype)
    clip_text_model.to(accelerator.device, dtype=weight_dtype)
    decoder.eval()
    vqae.eval()
    clip_model.eval()
    clip_text_model.eval()

    output_dir = args.output_dir
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    num_train_epochs = args.num_train_epochs

    def vqvae_encoder(images):
        with torch.no_grad():
            encoded_image = vqae.encode(images)[0]
        return encoded_image

    def vqvae_decoder(latents):
        with torch.no_grad():
            decoded = vqae.decode(latents)[0]
        return decoded

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_dataset, test_dataset = get_coco_dataloader(batch_size=args.batch_size, image_size=256, max_size_test=16, is_test=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda examples: collate_fn(examples), drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=lambda examples: collate_fn(examples))
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    if args.resume_from_checkpoint:
        checkpoint_path = f'{args.load_fm_weights_checkpoint}_{args.resume_from_checkpoint}_{sigma}.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        if hasattr(prepare_dict, '__call__'):
            state_dict = prepare_dict(state_dict)
        model.load_state_dict(state_dict)
        initial_global_step = num_update_steps_per_epoch * args.resume_from_checkpoint
        global_step = initial_global_step
        first_epoch = args.resume_from_checkpoint
    else:
        global_step = 0
        initial_global_step = 0
        first_epoch = 0
    checkpointing_steps = num_update_steps_per_epoch * args.checkpointing_steps
    validation_steps = num_update_steps_per_epoch * args.validation_epoch

    progress_bar = tqdm(range(0, max_train_steps),
                        initial=initial_global_step,
                        desc="Steps",
                        disable=not accelerator.is_local_main_process,
                        )
    print("#### STARTING TRAINING ####")
    z_final = torch.load(args.load_z)
    z_means = z_final.mean(dim=0)
    z_stds = z_final.std(dim=0)
    del z_final
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        print(f"#### EPOCH {epoch+1} ####")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                images = batch['images'].to(accelerator.device)
                encoded_images  = vqvae_encoder(images)
                clip_texts = encode_prompt(clip_model, batch['input_ids'], batch['attention_mask'])
                cond_text = clip_text_model(batch['input_ids'].to(accelerator.device)).last_hidden_state
                zi = torch.randn((len(batch['images']), dim_z), device=accelerator.device, requires_grad=False)
                zi = zi * z_stds.to(accelerator.device) + z_means.to(accelerator.device)
                x0 = decoder(clip_texts, zi)

                x0 = x0 + (torch.randn_like(x0).type_as(x0).to(accelerator.device) * sigma)

                x1 = encoded_images.to(accelerator.device)

                t = torch.rand(x0.shape[0]).type_as(x0).to(accelerator.device)

                xt = sample_conditional_pt(x0, x1, t, sigma=0.01)
                ut = compute_conditional_vector_field(x0, x1)

                vt = model(xt, t, encoder_hidden_states=cond_text, return_dict=False)[0]

                loss = torch.mean((vt - ut) ** 2)
                accelerator.backward(loss)
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        print("### save checkpoint ####")
                        torch.save(model.state_dict(), f'{output_dir}/weights_{epoch}_{args.sigma}.pth')
                        print("### finished saving ####")

                    if global_step % validation_steps == 0:
                        print("#### VALIDATION ####")
                        with torch.no_grad():
                            model.eval()
                            for step, batch in enumerate(test_dataloader):
                                images = batch['images'].to(accelerator.device)
                                clip_texts = encode_prompt(clip_model, batch['input_ids'], batch['attention_mask'])
                                cond_text = clip_text_model(batch['input_ids'].to(accelerator.device)).last_hidden_state
                                zi = torch.randn((len(batch['images']), dim_z), device=accelerator.device)
                                zi = zi * z_stds + z_means
                                x0 = decoder(clip_texts, zi)
                                x0 = x0 + (torch.randn_like(x0).type_as(x0).to(accelerator.device) * sigma)
                                dt = torch.tensor([1 / num_steps]).to(accelerator.device)
                                xt = x0.to(accelerator.device)
                                for step in torch.linspace(0, 1, num_steps+1):
                                    t = torch.full((x0.shape[0],), step).type_as(x0).to(accelerator.device)
                                    vt = model(xt, t, encoder_hidden_states=cond_text, return_dict=False)[0]
                                    xt = xt + (dt * vt)

                                clip_text_div = clip_texts[9].unsqueeze(0)
                                diversity_arr = []
                                for _ in range(args.num_diversity):
                                    zi = torch.randn((1, dim_z), device=accelerator.device)
                                    zi = zi * z_stds + z_means
                                    x0_diversety = decoder(clip_text_div, zi)
                                    xt_diversety = x0_diversety
                                    for step in torch.linspace(0, 1, num_steps + 1):
                                        t = torch.full((x0_diversety.shape[0],), step).type_as(x0_diversety).to(accelerator.device)
                                        vt = model(xt_diversety, t, encoder_hidden_states=cond_text[9].unsqueeze(0), return_dict=False)[0]
                                        xt_diversety = xt_diversety + (dt * vt)
                                    diversity_arr.append(xt_diversety[0])

                                generated_images = vqvae_decoder(xt)
                                decoded_images = vqvae_decoder(x0)
                                diversity_arr = vqvae_decoder(torch.stack(diversity_arr))

                                captions = clip_tokenizer.batch_decode(batch['input_ids'].to(accelerator.device),
                                                                       skip_special_tokens=True)
                                num_upload_images = args.num_upload_images
                                print("#### Start Upload to Wandb ####")
                                for tracker in accelerator.trackers:
                                    tracker.log(
                                        {
                                            "Decoded": [
                                                wandb.Image(image, caption=f"{i}: {captions[i]}")
                                                for i, image in enumerate(decoded_images[:num_upload_images])
                                            ],
                                            "Ground_Truth": [
                                                wandb.Image(image, caption=f"{i}: {captions[i]}")
                                                for i, image in enumerate(images[:num_upload_images])

                                            ],
                                            "Generated": [
                                                wandb.Image(image, caption=f"{i}: {captions[i]}")
                                                for i, image in enumerate(generated_images[:num_upload_images])

                                            ],
                                            "Diversity": [
                                                wandb.Image(image, caption=f"")
                                                for i, image in enumerate(diversity_arr)

                                            ],
                                        }
                                    )
                                    print("#### END Uploading ####")
                                del generated_images
                                del decoded_images
                                del captions
                                del diversity_arr
                                model.train()
                                break
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        torch.save(model.state_dict(), f'{output_dir}/weights_{args.sigma}.pth')
    accelerator.end_training()

if __name__ == '__main__':
    args = parse_args()
    train_image_fm(args)

