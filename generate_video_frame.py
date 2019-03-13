# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from training import misc
import scipy
import train

def transitionAtoB_v2(
    run_id          = 102,     # Run ID or network pkl to resume training from, None = start from scratch.
    snapshot        = None,
    num_frames      = 10,
    interpolate_dim = 350):

    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    print('Loading networks from "%s"...' % network_pkl)
    G, D, Gs = misc.load_pkl(network_pkl)

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState(10)  # seed = 10
    init_latent = rnd.randn(1, Gs.input_shape[1])[0]

    def apply_latent_fudge(fudge):
      copy = np.copy(init_latent)
      copy[interpolate_dim] += fudge
      return copy

    interpolate = np.linspace(0., 30., num_frames) - 15
    latents = np.array(list(map(apply_latent_fudge, interpolate)))

    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    os.makedirs(os.path.join(config.result_dir, 'video'), exist_ok=True)
    for idx in range(num_frames):
        # Save image.
        png_filename = os.path.join(os.path.join(config.result_dir, os.path.basename(network_pkl).replace(".mp4","")), 'frame_'+'{0:04d}'.format(i)+'.png')
        PIL.Image.fromarray(images[idx], 'RGB').save(png_filename)

def transitionAtoB(
    run_id          = 102,     # Run ID or network pkl to resume training from, None = start from scratch.
    snapshot        = None,
    num_gen_noise   = 100):
    #http://cedro3.com/ai/stylegan/

    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    print('Loading networks from "%s"...' % network_pkl)
    G, D, Gs = misc.load_pkl(network_pkl)

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState(10)  # seed = 10
    latentsA = rnd.randn(1, Gs.input_shape[1])
    for _ in range(num_gen_noise):
        latents_ = rnd.randn(1, Gs.input_shape[1])
    latentsB = rnd.randn(1, Gs.input_shape[1])

    num_split = 39  # 2つのベクトルを39分割
    for i in range(30):
        latents = latentsB+(latentsA-latentsB)*i/num_split
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        # Save image.
        os.makedirs(os.path.join(config.result_dir, 'video'), exist_ok=True)
        png_filename = os.path.join(os.path.join(config.result_dir, 'video'), 'photo'+'{0:04d}'.format(i)+'.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

def main(
    run_id          = 101,     # Run ID or network pkl to resume training from, None = start from scratch.
    snapshot        = None,     # Snapshot index to resume training from, None = autodetect.
    grid_size       = [1,1],
    image_shrink    = 1,
    image_zoom      = 1,
    duration_sec    = 3.0,
    smoothing_sec   = 1.0,
    mp4             = None,
    mp4_fps         = 30,
    mp4_codec       = 'libx265',
    mp4_bitrate     = '16M',
    random_seed     = 1000,
    minibatch_size  = 8 ):

    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    print('Loading networks from "%s"...' % network_pkl)
    G, D, Gs = misc.load_pkl(network_pkl)

    mp4 = '%s-lerp.mp4' % network_pkl
    num_frames = int(np.rint(duration_sec * mp4_fps))

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    print('Generating latent vectors...')
    rnd = np.random.RandomState(5)
    shape = [num_frames, Gs.input_shape[1]]
    all_latents = rnd.randn(*shape).astype(np.float32)
    #all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0], mode='wrap')
    #all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    all_dlatents = Gs.components.mapping.run(all_latents, None)
    all_dlatents = scipy.ndimage.gaussian_filter(all_dlatents, [smoothing_sec * mp4_fps] + [0]*2, mode='wrap')
    print (shape, all_latents.shape, all_dlatents.shape)

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    for idx in range(num_frames):
        dlatents = all_dlatents[idx]
        images = Gs.run(dlatents, None, truncation_psi=0.0, randomize_noise=True, output_transform=fmt)
        png_filename = os.path.join(config.result_dir, 'frame_%s.png' % str(idx).zfill(8))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
    #
    # # Frame generation func for moviepy.
    # fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    # def make_frame(t):
    #     frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
    #     dlatents = all_dlatents[frame_idx]
    #     labels = np.zeros([dlatents.shape[0], 0], np.float32)
    #     # images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
    #     images = Gs.run(dlatents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    #     grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
    #     if image_zoom > 1:
    #         grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
    #     if grid.shape[2] == 1:
    #         grid = grid.repeat(3, 2) # grayscale => RGB
    #     return grid
    #
    # # Generate video.
    # import moviepy.editor # pip install moviepy
    # result_subdir = config.result_dir
    # #result_subdir = misc.create_result_subdir(config.result_dir, train.desc)
    # moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, os.path.basename(mp4)), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    # open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

if __name__ == "__main__":
    transitionAtoB_v2()
