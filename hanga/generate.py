import pathlib
import pickle
from typing import List, Optional

import torch
from IPython.display import display
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm


# TODO: convert this into a pipeline / class instead so the various models can be reused


def get_init_latents(
    generator,  # Optional
    scheduler,
    unet,
    batch_size: int,
    height: int = 512,
    width: int = 512,
    torch_device: str = "cuda",
):
    # generate random initial noise (64x64, the model transforms this into 512x512)
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(torch_device)

    # the K-LMS scheduler needs to multiply the latents by its sigma values
    # TODO: move this out / make it better
    latents *= scheduler.sigmas[0]

    return latents


def get_text_embeddings(
    tokenizer,
    text_encoder,
    batch_size: int,
    prompt: List[str],
    torch_device: str = "cuda",
):
    # get text embeddings for the prompt
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    # get unconditional text embeddings for classifier-free guidance (i.e. the embeddings for the padding token (empty text))
    # these need to have the same shape as the conditional text embeddings (batch_size & seq_length)
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    # for classifier-free guidance, we need to do two forward passes. One with the conditioned input (text_embeddings),
    # and another with the unconditional embeddings (uncond_embeddings). In practice, we can concatenate both into a
    # single batch to avoid doing two forward passes.
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings


def denoising_loop(
    scheduler,
    unet,
    text_embeddings,
    latents,
    guidance_scale: float,
    output_dir: Optional[str] = None,
    save_freq: Optional[int] = None,
):
    """"""
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):

            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # TODO: work out what this sigma stuff is & how I can tweak it
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
                # TODO: this is the core model stuff from latent —> image,
                # => these are the tools I have to work with... e.g. how does adding some noise to the embedding mess
                # with the image output (I guess it goes pretty wrong over 50 iterations...)
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

            if output_dir is not None and save_freq is not None and i % save_freq == 0:
                pickle.dump(latents, pathlib.Path(f"{output_dir}/latents_{i}.pklz").open("wb"))

    return latents


def get_vae_output(vae, latents):
    """"""
    # now use the VAE to decode the latents back to an image, scaling & decoding the image latents with VAE
    with torch.no_grad():
        return vae.decode(1 / 0.18215 * latents)


def vae_output_to_pil(output_image):
    # and then convert to PIL so we can display it
    output_image = (output_image.sample / 2 + 0.5).clamp(0, 1)
    output_image = output_image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (output_image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]


def pipeline(prompt: List[str], generator=None, return_latents: bool = False):
    # TODO: update to the new API needed here (e.g. passing the objects around)
    text_embeddings = get_text_embeddings(prompt)
    latents = get_init_latents(generator=generator, height=512, width=512)
    latents = denoising_loop(text_embeddings, latents)
    if return_latents:
        return latents
    vae_image = get_vae_output(vae, latents)
    image = vae_output_to_pil(vae_image)
    return image


def gen_from_seed_and_prompt(
    scheduler,
    unet,
    tokenizer,
    text_encoder,
    vae,
    batch_size: int,
    guidance_scale: float,
    seed: int,
    prompt: str,
    output_dir: Optional[str] = None,
    save_freq: Optional[int] = None,
    display_: bool = True,
    height: int = 512,
    width: int = 512,
    torch_device: str = "cuda",
):
    generator = torch.manual_seed(seed)
    init_latents = get_init_latents(
        scheduler=scheduler,
        unet=unet,
        generator=generator,
        batch_size=batch_size,
        height=height,
        width=width,
        torch_device=torch_device,
    )

    if output_dir is not None:
        pickle.dump(init_latents, pathlib.Path(f"{output_dir}/init_latents_{seed}.pklz").open("wb"))

    text_embeddings = get_text_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        batch_size=batch_size,
        prompt=prompt,
        torch_device=torch_device,
    )

    doot = denoising_loop(
        scheduler=scheduler,
        unet=unet,
        text_embeddings=text_embeddings,
        latents=init_latents,
        guidance_scale=guidance_scale,
        output_dir=output_dir,
        save_freq=save_freq,
    )
    doot = get_vae_output(vae, doot)
    pil_image = vae_output_to_pil(doot)

    if output_dir is not None:
        pil_image.save(f"{output_dir}/output.png")

    if display_:
        print(seed)
        # display(pil_image.resize((256, 256)))
        display(pil_image)

    return pil_image
