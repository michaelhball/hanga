import gc
import random
import queue
import uuid
from threading import Thread
from typing import List, Optional

import ffmpeg
import librosa
import numpy as np
import PIL.Image
import time
import torch
from tqdm import tqdm

from hanga import audio as h_audio, models as h_model, util as h_util


def chroma_weight_latents(chroma: torch.tensor, latents: torch.tensor) -> torch.tensor:
    """Creates chromagram weighted latent sequence

    Args:
        chroma: Chromagram
        latents: Latents (must have same number as number of notes in chromagram)

    Returns: Chromagram weighted latent sequence
    """
    base_latents = (chroma[..., None, None] * latents[None, ...]).sum(1)
    return base_latents


def get_noise_range(out_size: int, generator_resolution: int):
    """Gets the correct number of noise resolutions for a given resolution of StyleGAN 2"""
    log_max_res = int(np.log2(out_size))
    log_min_res = 2 + (log_max_res - int(np.log2(generator_resolution)))
    range_min = 2 * log_min_res + 1
    range_max = 2 * (log_max_res + 1)
    side_fn = lambda x: int(x / 2)
    return range_min, range_max, side_fn


class AudioVisualiser:
    def __init__(self, audio_file: str, offset: float, duration: float = -1, fps: int = 30):
        self.audio_file = audio_file
        self.offset = offset
        self.duration = duration
        self.fps = fps

        audio_dur = librosa.get_duration(filename=audio_file)
        if self.duration == -1 or audio_dur < self.duration:
            self.duration = audio_dur
            if offset != 0:
                self.duration -= offset

        self.y, self.sr = librosa.load(audio_file, offset=offset, duration=self.duration)
        self.n_frames = int(round(self.duration * fps))
        self.lo_onsets = h_audio.onsets(self.y, self.sr, self.n_frames, fmax=150, smooth=5, clip=97, power=2)
        self.hi_onsets = h_audio.onsets(self.y, self.sr, self.n_frames, fmax=500, smooth=5, clip=99, power=2)

    def get_latents(self, latent_selection: torch.tensor) -> torch.tensor:
        """"""
        raise NotImplementedError

    def get_noise(self, height: int, width: int, scale, num_scales) -> torch.tensor:
        """"""
        raise NotImplementedError

    def noise(self, out_size: int, G_res: int) -> List[torch.tensor]:
        noise = []
        range_min, range_max, exponent = get_noise_range(out_size, G_res)
        for scale in range(range_min, range_max):
            h = (2 if out_size == 1080 else 1) * 2 ** exponent(scale)
            w = (2 if out_size == 1920 else 1) * 2 ** exponent(scale)
            noise.append(self.get_noise(height=h, width=w, scale=scale - range_min, num_scales=range_max - range_min))
            if noise[-1] is not None:
                print(list(noise[-1].shape), f"amplitude={noise[-1].std()}")
            gc.collect()
            torch.cuda.empty_cache()
        return noise

    def render(
        self,
        output_dir: str,
        latent_count: int = 12,
        shuffle_latents: bool = False,
        G_res: int = 1024,
        out_size: int = 1024,
        latent_dim: int = 512,
        n_mlp: int = 2,
        channel_multiplier: int = 2,
        base_res_factor: int = 1,
        truncation: float = 1.0,
        data_parallel: bool = False,
        randomise_noise: bool = False,
        batch=8,
        ffmpeg_preset="slow",
        ckpt: Optional[str] = None,
    ):
        """"""
        time_taken = time.time()
        torch.set_grad_enabled(False)
        noconst = False

        # latents
        latent_selection = h_model.generate_latents(
            latent_count=latent_count,
            ckpt=ckpt,
            G_res=G_res,
            noconst=noconst,
            latent_dim=latent_dim,
            n_mlp=n_mlp,
            channel_multiplier=channel_multiplier,
        )
        if shuffle_latents:
            random_indices = random.sample(range(len(latent_selection)), len(latent_selection))
            latent_selection = latent_selection[random_indices]

        latents = self.get_latents(latent_selection=latent_selection).cpu()
        print(f"{list(latents.shape)} amplitude={latents.std()}\n")

        # noise
        noise = self.noise(out_size, G_res)

        # TODO: add bends / rewrites / truncation
        bends = []
        rewrites = {}
        truncation = float(truncation)

        gc.collect()
        torch.cuda.empty_cache()

        # load generator
        generator = h_model.StyleGAN2Generator(
            G_res,
            latent_dim,
            n_mlp,
            channel_multiplier=channel_multiplier,
            constant_input=not noconst,
            checkpoint=ckpt,
            output_size=out_size,
            base_res_factor=base_res_factor,
        ).cuda()
        if data_parallel:
            generator = torch.nn.DataParallel(generator)

        print(f"\npreprocessing took {time.time() - time_taken:.2f}s\n")

        # render outputs
        print(f"rendering {self.n_frames} frames...")
        if output_file is None:
            checkpoint_title = ckpt.split("/")[-1].split(".")[0].lower()
            track_title = self.audio_file.split("/")[-1].split(".")[0].lower()
            output_file = f"{output_dir}/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"
        render(
            generator=generator,
            latents=latents,
            noise=noise,
            audio_file=self.audio_file,
            offset=self.offset,
            duration=self.duration,
            batch_size=batch,
            truncation=truncation,
            bends=bends,
            rewrites=rewrites,
            out_size=out_size,
            output_file=output_file,
            randomize_noise=randomise_noise,
            ffmpeg_preset=ffmpeg_preset,
        )

        print(f"\ntotal time taken: {(time.time() - time_taken)/60:.2f} minutes")


class DefaultAV(AudioVisualiser):
    def get_latents(self, latent_selection: torch.tensor) -> torch.tensor:
        """"""
        chroma = h_audio.chroma(self.y, self.sr, self.n_frames)
        chroma_latents = chroma_weight_latents(chroma, latent_selection)
        latents = h_util.gaussian_filter(chroma_latents, 4)

        lo_onsets = self.lo_onsets[:, None, None]
        hi_onsets = self.hi_onsets[:, None, None]

        latents = hi_onsets * latent_selection[[-4]] + (1 - hi_onsets) * latents
        latents = lo_onsets * latent_selection[[-7]] + (1 - lo_onsets) * latents

        latents = h_util.gaussian_filter(latents, 2, causal=0.2)

        return latents

    def get_noise(self, height: int, width: int, scale, num_scales):
        """"""
        if width > 256:
            return None

        lo_onsets = self.lo_onsets[:, None, None, None].cuda()
        hi_onsets = self.hi_onsets[:, None, None, None].cuda()

        noise_noisy = h_util.gaussian_filter(torch.randn((self.n_frames, 1, height, width), device="cuda"), 5)
        noise = h_util.gaussian_filter(torch.randn((self.n_frames, 1, height, width), device="cuda"), 128)

        if width < 128:
            noise = lo_onsets * noise_noisy + (1 - lo_onsets) * noise
        if width > 32:
            noise = hi_onsets * noise_noisy + (1 - hi_onsets) * noise

        noise /= noise.std() * 2.5

        return noise.cpu()


def render(
    generator,
    latents,
    noise,
    offset,
    duration,
    batch_size,
    out_size,
    output_file,
    audio_file=None,
    truncation=1.0,
    bends=[],
    rewrites={},
    randomize_noise=False,
    ffmpeg_preset="slow",
):
    # TODO: serious documentation needed here...
    torch.backends.cudnn.benchmark = True

    split_queue = queue.Queue()
    render_queue = queue.Queue()

    # postprocesses batched torch tensors to individual RGB numpy arrays
    def split_batches(jobs_in, jobs_out):
        while True:
            try:
                imgs = jobs_in.get(timeout=5)
            except queue.Empty:
                return
            imgs = (imgs.clamp_(-1, 1) + 1) * 127.5
            imgs = imgs.permute(0, 2, 3, 1)
            for img in imgs:
                jobs_out.put(img.cpu().numpy().astype(np.uint8))
            jobs_in.task_done()

    # start background ffmpeg process that listens on stdin for frame data
    if out_size == 512:
        output_size = "512x512"
    elif out_size == 1024:
        output_size = "1024x1024"
    elif out_size == 1920:
        output_size = "1920x1080"
    elif out_size == 1080:
        output_size = "1080x1920"
    else:
        raise Exception("The only output sizes currently supported are: 512, 1024, 1080, or 1920")

    if audio_file is not None:
        audio = ffmpeg.input(audio_file, ss=offset, t=duration, guess_layout_max=0)
        video = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                framerate=len(latents) / duration,
                s=output_size,
            )
            .output(
                audio,
                output_file,
                framerate=len(latents) / duration,
                vcodec="libx264",
                pix_fmt="yuv420p",
                preset=ffmpeg_preset,
                audio_bitrate="320K",
                ac=2,
                v="warning",
            )
            .global_args("-hide_banner")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    else:
        video = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                framerate=len(latents) / duration,
                s=output_size,
            )
            .output(
                output_file,
                framerate=len(latents) / duration,
                vcodec="libx264",
                pix_fmt="yuv420p",
                preset=ffmpeg_preset,
                v="warning",
            )
            .global_args("-hide_banner")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    # writes numpy frames to ffmpeg stdin as raw rgb24 bytes
    def make_video(jobs_in):
        w, h = [int(dim) for dim in output_size.split("x")]
        for _ in tqdm(range(len(latents)), position=0, leave=True, ncols=80):
            img = jobs_in.get(timeout=5)
            if img.shape[1] == 2048:
                img = img[:, 112:-112, :]
                im = PIL.Image.fromarray(img)
                img = np.array(im.resize((1920, 1080), PIL.Image.BILINEAR))
            elif img.shape[0] == 2048:
                img = img[112:-112, :, :]
                im = PIL.Image.fromarray(img)
                img = np.array(im.resize((1080, 1920), PIL.Image.BILINEAR))
            assert (
                img.shape[1] == w and img.shape[0] == h
            ), f"""generator's output image size does not match specified output size: \n
                got: {img.shape[1]}x{img.shape[0]}\t\tshould be {output_size}"""
            video.stdin.write(img.tobytes())
            jobs_in.task_done()
        video.stdin.close()
        video.wait()

    splitter = Thread(target=split_batches, args=(split_queue, render_queue))
    splitter.daemon = True
    renderer = Thread(target=make_video, args=(render_queue,))
    renderer.daemon = True

    # make all data that needs to be loaded to the GPU float, contiguous, and pinned
    # the entire process is severly memory-transfer bound, but at least this might help a little
    latents = latents.float().contiguous().pin_memory()

    for ni, noise_scale in enumerate(noise):
        noise[ni] = noise_scale.float().contiguous().pin_memory() if noise_scale is not None else None

    param_dict = dict(generator.named_parameters())
    original_weights = {}
    for param, (rewrite, modulation) in rewrites.items():
        rewrites[param] = [rewrite, modulation.float().contiguous().pin_memory()]
        original_weights[param] = param_dict[param].copy().cpu().float().contiguous().pin_memory()

    for bend in bends:
        if "modulation" in bend:
            bend["modulation"] = bend["modulation"].float().contiguous().pin_memory()

    if not isinstance(truncation, float):
        truncation = truncation.float().contiguous().pin_memory()

    for n in range(0, len(latents), batch_size):
        # load batches of data onto the GPU
        latent_batch = latents[n : n + batch_size].cuda(non_blocking=True)

        noise_batch = []
        for noise_scale in noise:
            if noise_scale is not None:
                noise_batch.append(noise_scale[n : n + batch_size].cuda(non_blocking=True))
            else:
                noise_batch.append(None)

        bend_batch = []
        if bends is not None:
            for bend in bends:
                if "modulation" in bend:
                    transform = bend["transform"](bend["modulation"][n : n + batch_size].cuda(non_blocking=True))
                    bend_batch.append({"layer": bend["layer"], "transform": transform})
                else:
                    bend_batch.append({"layer": bend["layer"], "transform": bend["transform"]})

        for param, (rewrite, modulation) in rewrites.items():
            transform = rewrite(modulation[n : n + batch_size])
            rewritten_weight = transform(original_weights[param]).cuda(non_blocking=True)
            param_attrs = param.split(".")
            mod = generator
            for attr in param_attrs[:-1]:
                mod = getattr(mod, attr)
            setattr(mod, param_attrs[-1], th.nn.Parameter(rewritten_weight))

        if not isinstance(truncation, float):
            truncation_batch = truncation[n : n + batch_size].cuda(non_blocking=True)
        else:
            truncation_batch = truncation

        # forward through the generator
        outputs, _ = generator(
            styles=latent_batch,
            noise=noise_batch,
            truncation=truncation_batch,
            transform_dict_list=bend_batch,
            randomize_noise=randomize_noise,
            input_is_latent=True,
        )

        # send output to be split into frames and rendered one by one
        split_queue.put(outputs)

        if n == 0:
            splitter.start()
            renderer.start()

    splitter.join()
    renderer.join()


def write_video(arr: np.ndarray, output_file: str, fps: int):
    print(f"writing {arr.shape[0]} frames...")

    output_size = "x".join(reversed([str(s) for s in arr.shape[1:-1]]))

    ffmpeg_proc = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=fps, s=output_size)
        .output(output_file, framerate=fps, vcodec="libx264", preset="slow", v="warning")
        .global_args("-benchmark", "-stats", "-hide_banner")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in arr:
        ffmpeg_proc.stdin.write(frame.astype(np.uint8).tobytes())

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
