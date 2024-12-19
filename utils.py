import argparse
import math
import os
import cv2
import subprocess
from datetime import timedelta
from urllib.parse import urlparse
import re
import numpy as np
import PIL
from PIL import Image, ImageDraw
import datetime
import torch
import torchvision
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
import random
from skimage.metrics import structural_similarity as compare_ssim

from diffusers.utils import load_image




def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=100,
                       loop=0)
    
from PIL import Image
import numpy as np

def export_gif_with_ref(start_image, frames, end_image, reference_image, output_gif_path, fps):
    """
    Export a list of frames into a GIF with columns and an additional version with only frames.

    Args:
    - start_image (PIL.Image): The starting image.
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - end_image (PIL.Image): The ending image.
    - reference_image (PIL.Image): The reference image.
    - output_gif_path (str): Path to save the output GIF.
    - fps (int): Frames per second for the GIF.
    """
    
    # Convert numpy frames to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    
    # Get dimensions of images
    width, height = start_image.size
    
    # Resize the reference image and frames to match the height of start and end images if needed
    reference_image = reference_image.resize((reference_image.width, height))
    resized_frames = [frame.resize((frame.width, height)) for frame in pil_frames]
    
    # Create a new image for each frame with the three columns
    column_frames = []
    for frame in resized_frames:
        # Create an empty image with the total width for all three columns
        new_width = start_image.width + reference_image.width + end_image.width+frame.width
        combined_frame = Image.new('RGB', (new_width, height))
        
        # Paste the start image, reference image, and frame into the new image
        combined_frame.paste(start_image, (0, 0))
        combined_frame.paste(reference_image, (start_image.width, 0))
        combined_frame.paste(end_image, (start_image.width + reference_image.width, 0))
        combined_frame.paste(frame, (start_image.width + reference_image.width+end_image.width, 0))
        
        column_frames.append(combined_frame)
    
    # Calculate frame duration in milliseconds based on fps
    frame_duration = 150
    
    # Save the GIF with columns
    column_frames[0].save(output_gif_path,
                          format='GIF',
                          append_images=column_frames[1:],
                          save_all=True,
                          duration=frame_duration,
                          loop=0)
    
  



def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def map_ssim_distance(dis):
    if dis > 0.95:
        return 1
    elif dis > 0.9: 
        return 2
    elif dis > 0.85: 
        return 3
    elif dis > 0.80: 
        return 4
    elif dis > 0.75: 
        return 5
    elif dis > 0.70: 
        return 6
    elif dis > 0.65: 
        return 7
    elif dis > 0.60: 
        return 8
    elif dis > 0.55: 
        return 9
    else: 
        return 10


def calculate_ssim(frame1, frame2):
    # convert the frames to grayscale images since the compare_ssim function accepts grayscale images
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    # compute SSIM
    ssim = compare_ssim(gray_frame1, gray_frame2)
    
    return ssim


def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err


def calculate_video_motion_distance(frames_data):
    # obtain the number of frames in the video
    frame_count, _, _, _ = frames_data.shape
    
    # init
    similarities = []

    # calculate the similarity between each two frames
    for frame_index in range(1, frame_count):
        prev_frame = frames_data[frame_index - 1, :, :, :]
        current_frame = frames_data[frame_index, :, :, :]

        # calculate the similarity, you can choose to use SSIM or MSE, etc.
        similarity = calculate_ssim(prev_frame, current_frame)
        similarities.append(similarity)

    # calculate the mean similarity as the motion distance of the video
    motion_distance = np.mean(similarities)

    return similarities, motion_distance



def load_images_from_folder_to_pil(folder, target_size=(512, 512)):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    def frame_number(filename):
        # Try the pattern 'frame_x_7fps'
        new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))

        # If the new pattern is not found, use the original digit extraction method
        matches = re.findall(r'\d+', filename)
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    # sorted_files = sorted(os.listdir(folder), key=frame_number)
    sorted_files = sorted(os.listdir(folder))

    # Load, resize, and convert images
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image with original channels
            if img is not None:
                # Resize image
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Convert to uint8 if necessary
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                # Ensure all images are in RGB format
                if len(img.shape) == 2:  # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert the numpy array to a PIL image
                pil_img = Image.fromarray(img)
                images.append(pil_img)

    return images

def extract_frames_from_video(video_path):

    video_capture = cv2.VideoCapture(video_path)
    
    frames = []
    

    if not video_capture.isOpened():

        return frames
    
 
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  
        

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        pil_image = Image.fromarray(frame_rgb)
 
        frames.append(pil_image)

    video_capture.release()
    
    return frames


def export_gif_side_by_side(ref_frame,sketches, frames, output_gif_path, fps):
    """
    Export a list of frames into a GIF with columns and an additional version with only frames.

    Args:
    - start_image (PIL.Image): The starting image.
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - end_image (PIL.Image): The ending image.
    - reference_image (PIL.Image): The reference image.
    - output_gif_path (str): Path to save the output GIF.
    - fps (int): Frames per second for the GIF.
    """
    
    # Convert numpy frames to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    
    # Get dimensions of images
    width, height = pil_frames[0].size
    

    resized_frames = [frame.resize((width, height)) for frame in pil_frames]
    resized_sketches = [sketch.resize((width, height)) for sketch in sketches]
    ref_frame=ref_frame.resize((width, height))
    # Create a new image for each frame with the three columns
    column_frames = []
    for i, frame in enumerate(resized_frames):
        # Create an empty image with the total width for all three columns
        new_width = resized_sketches[0].width + frame.width+frame.width
        combined_frame = Image.new('RGB', (new_width, height))
        
        # Paste the start image, reference image, and frame into the new image

        combined_frame.paste(ref_frame, (0, 0))
        combined_frame.paste(resized_sketches[i], (resized_sketches[0].width, 0))

        combined_frame.paste(frame, (resized_sketches[0].width+resized_sketches[0].width, 0))
        
        column_frames.append(combined_frame)
    
    # Calculate frame duration in milliseconds based on fps
    frame_duration = 150
    
    # Save the GIF with columns
    column_frames[0].save(output_gif_path,
                          format='GIF',
                          append_images=column_frames[1:],
                          save_all=True,
                          duration=frame_duration,
                          loop=0)
    

#shuffle operation

def safe_round(coords, size):
    height, width = size[1], size[2]  
    rounded_coords = np.round(coords).astype(int)
    rounded_coords[:, 0] = np.clip(rounded_coords[:, 0], 0, width - 1)
    rounded_coords[:, 1] = np.clip(rounded_coords[:, 1], 0, height - 1)    
    return rounded_coords
def random_number(num_points,size,coords0,coords1):
    shuffle_indices = np.random.permutation(np.arange(coords0.shape[0]))


    shuffled_coords0 = coords0[shuffle_indices]
    shuffled_coords1 = coords1[shuffle_indices]
    indices = np.random.choice(np.arange(shuffled_coords0.shape[0]), size=num_points, replace=False)

    # selected_coords0 = coords0[indices]
    # selected_coords1 = coords1[indices]
    selected_coords0 = shuffled_coords0[indices]
    selected_coords1 = shuffled_coords1[indices]
    h, w = size[1], size[2]     
    mask0 = np.zeros((h, w), dtype=np.uint8)
    mask1 = np.zeros((h, w), dtype=np.uint8)
    for i, (coord0, coord1) in enumerate(zip(selected_coords0, selected_coords1)):
        x0, y0 = coord0
        x1, y1 = coord1
        # import ipdb;ipdb.set_trace()
        mask0[y0, x0] = i + 1
        mask1[y1, x1] = i + 1
    return mask0,mask1


def split_and_shuffle(image, coordinates):

    assert image.shape[1] % 2 == 0 and image.shape[2] % 2 == 0, "Height and width must be even."
    

    H, W = image.shape[1], image.shape[2]


    patches_img = [
        image[:, :H//2, :W//2],
        image[:, :H//2, W//2:],
        image[:, H//2:, :W//2],
        image[:, H//2:, W//2:]
    ]
    
    patch_coords = [
        (0, H//2, 0, W//2),
        (0, H//2, W//2, W),
        (H//2, H, 0, W//2),
        (H//2, H, W//2, W)
    ]
    

    indices = list(range(4))
    random.shuffle(indices)
    

    new_patch_coords = [
        (0, 0),
        (0, W//2),
        (H//2, 0),
        (H//2, W//2)
    ]
    
   
    new_coordinates = np.zeros_like(coordinates)
    for i, (r, c) in enumerate(coordinates):
        for idx, (r1, r2, c1, c2) in enumerate(patch_coords):
            if r1 <= r < r2 and c1 <= c < c2:
                new_r = r - r1 + new_patch_coords[indices.index(idx)][0]
                new_c = c - c1 + new_patch_coords[indices.index(idx)][1]
                new_coordinates[i] = [new_r, new_c]
                break


    shuffled_img = torch.cat([
        torch.cat([patches_img[indices[0]], patches_img[indices[1]]], dim=2),
        torch.cat([patches_img[indices[2]], patches_img[indices[3]]], dim=2)
    ], dim=1)

    return shuffled_img, new_coordinates


import os
import cv2

def extract_frames_from_videos(video_folder):

    for filename in os.listdir(video_folder):
        if filename.endswith('.mp4'):
            video_path = os.path.join(video_folder, filename)
        
            frames_folder = os.path.join("processed_video", os.path.splitext(filename)[0])
            os.makedirs(frames_folder, exist_ok=True)
            

            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
           
                frame_filename = os.path.join(frames_folder, f'frame_{frame_count:04d}.jpg')

                cv2.imwrite(frame_filename, frame)
                frame_count += 1
            
            cap.release()
            print(f'Extracted {frame_count} frames from {filename} and saved to {frames_folder}')


def create_videos_from_frames(base_folder, output_folder, frame_rate=30):
    
    for root, dirs, files in os.walk(base_folder):
        frames = []
        for file in sorted(files):
            if file.endswith(('.jpg', '.png')):  
                frame_path = os.path.join(root, file)
                frames.append(frame_path)

        if len(frames) == 14:
            video_name = os.path.basename(root) + '.mp4'  
            video_path = os.path.join(output_folder, video_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
            first_frame = cv2.imread(frames[0])
            height, width, layers = first_frame.shape
            video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

            for frame in frames:
                img = cv2.imread(frame)
                video_writer.write(img)

            video_writer.release()
            print(f'Created video: {video_path}')

def random_rotate(image, angle_range=(-60, 60)):
    angle = random.uniform(*angle_range) 
    return image.rotate(angle, fillcolor=(255, 255, 255))

def random_crop(image,ratio=0.9):
    width, height = image.size
    ratio = random.uniform(0.6, 1.0)
    # print('ratio',ratio)
    top = random.randint(0, height - int(height*ratio))
    left = random.randint(0, width - int(width*ratio))
    image=image.crop((left, top, left + int( width*ratio), top + int(height*ratio)))
    image=image.resize((width,height))
    return image

def random_flip(image):
    if random.random() < 0.5:  
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:  
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


def patch_shuffle(image, num_patches):

 
    C, H, W = image.shape
    
    assert H % num_patches == 0 and W % num_patches == 0, "Image dimensions must be divisible by num_patches"
  
    patch_size_h = H // num_patches
    patch_size_w = W // num_patches
    

    patches = image.unfold(1, patch_size_h, patch_size_h).unfold(2, patch_size_w, patch_size_w)
    patches = patches.contiguous().view(C, num_patches * num_patches, patch_size_h, patch_size_w)
    

    shuffle_idx = torch.randperm(num_patches * num_patches)
    shuffled_patches = patches[:, shuffle_idx, :, :]
    
 
    shuffled_patches = shuffled_patches.view(C, num_patches, num_patches, patch_size_h, patch_size_w)
    shuffled_image = shuffled_patches.permute(0, 1, 3, 2, 4).contiguous()
    shuffled_image = shuffled_image.view(C, H, W)
    
    return shuffled_image
def augment_image(image,k):

    image = random_rotate(image)
    image = random_crop(image)  
    image = random_flip(image)
    # torch_image = torchvision.transforms.ToTensor()(image)
    # patch_shuffled_image = patch_shuffle(torch_image, k)
    # to_pil = transforms.ToPILImage()
    # image = to_pil(patch_shuffled_image)

    return image


def load_images_from_folder(folder):
    image_list = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):  
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path)
                image_list.append(img)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return image_list


def get_mask(model, input_img, s=640):
    input_img = (input_img / 255).astype(np.float32)
    h, w = h0, w0 = input_img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    tmpImg = torch.from_numpy(img_input).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():

        pred = model(tmpImg)
        pred = pred.cpu().numpy()[0]
        pred = np.transpose(pred, (1, 2, 0))
        pred = pred[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
        return pred


# code from 

def safe_round(coords, size):
    height, width = size[1], size[2]  
    rounded_coords = np.round(coords).astype(int)
    rounded_coords[:, 0] = np.clip(rounded_coords[:, 0], 0, width - 1)
    rounded_coords[:, 1] = np.clip(rounded_coords[:, 1], 0, height - 1)    
    return rounded_coords
def random_number(num_points,size,coords0,coords1):
    shuffle_indices = np.random.permutation(np.arange(coords0.shape[0]))


    shuffled_coords0 = coords0[shuffle_indices]
    shuffled_coords1 = coords1[shuffle_indices]
    indices = np.random.choice(np.arange(shuffled_coords0.shape[0]), size=num_points, replace=False)

    # selected_coords0 = coords0[indices]
    # selected_coords1 = coords1[indices]
    selected_coords0 = shuffled_coords0[indices]
    selected_coords1 = shuffled_coords1[indices]
    h, w = size[1], size[2]     
    mask0 = np.zeros((h, w), dtype=np.uint8)
    mask1 = np.zeros((h, w), dtype=np.uint8)
    for i, (coord0, coord1) in enumerate(zip(selected_coords0, selected_coords1)):
        x0, y0 = coord0
        x1, y1 = coord1
        # import ipdb;ipdb.set_trace()
        mask0[y0, x0] = i + 1
        mask1[y1, x1] = i + 1
    return mask0,mask1


import torch

def split_and_shuffle(image, keypoints, num_rows, num_cols):
    """
    Split the image into tiles, shuffle them, and update the keypoints accordingly.

    Parameters:
    - image: Tensor of shape (3, H, W)
    - keypoints: Tensor of shape (num_k, 2)
    - num_rows: int, number of rows to split
    - num_cols: int, number of columns to split

    Returns:
    - shuffled_image: Tensor of shape (3, H, W)
    - new_keypoints: Tensor of shape (num_k, 2)
    """
    C, H, W = image.shape

    # Calculate padding to make H and W divisible by num_rows and num_cols
    pad_h = (num_rows - H % num_rows) % num_rows
    pad_w = (num_cols - W % num_cols) % num_cols

    # Pad the image
    H_padded = H + pad_h
    W_padded = W + pad_w
    padded_image = torch.zeros((C, H_padded, W_padded), dtype=image.dtype).to(image.device)
    padded_image[:, :H, :W] = image

    # Compute tile size
    tile_height = H_padded // num_rows
    tile_width = W_padded // num_cols

    # Reshape and permute to get tiles
    tiles = padded_image.reshape(C,
                                 num_rows,
                                 tile_height,
                                 num_cols,
                                 tile_width)
    tiles = tiles.permute(1, 3, 0, 2, 4).contiguous()
    num_tiles = num_rows * num_cols
    tiles = tiles.view(num_tiles, C, tile_height, tile_width)

    # Shuffle the tiles
    idx_shuffle = torch.randperm(num_tiles).to(image.device)
    tiles_shuffled = tiles[idx_shuffle]

    # Reshape back to image
    tiles_shuffled = tiles_shuffled.view(num_rows, num_cols, C, tile_height, tile_width)
    shuffled_image = tiles_shuffled.permute(2, 0, 3, 1, 4).contiguous()
    shuffled_image = shuffled_image.view(C, H_padded, W_padded)
    shuffled_image = shuffled_image[:, :H, :W]  # Crop back to original size

    # Update keypoints
    x = keypoints[:, 0]
    y = keypoints[:, 1]

    # Compute the tile indices where the keypoints are located
    tile_rows = (y / tile_height).long()
    tile_cols = (x / tile_width).long()
    tile_indices = tile_rows * num_cols + tile_cols  # Shape: (num_k,)

    # Create inverse mapping from old tile indices to new tile positions
    idx_unshuffle = torch.argsort(idx_shuffle)  # idx_unshuffle[old_index] = new_index

    # Get new tile indices for each keypoint
    new_tile_indices = idx_unshuffle[tile_indices]
    new_tile_rows = new_tile_indices // num_cols
    new_tile_cols = new_tile_indices % num_cols

    # Compute offsets within the tile
    offset_x = x % tile_width
    offset_y = y % tile_height

    # Compute new keypoints coordinates
    new_x = new_tile_cols * tile_width + offset_x
    new_y = new_tile_rows * tile_height + offset_y

    # Ensure keypoints are within image boundaries
    new_x = new_x.clamp(0, W - 1)
    new_y = new_y.clamp(0, H - 1)

    new_keypoints = torch.stack([new_x, new_y], dim=1)

    return shuffled_image, new_keypoints

def generate_point_map(size, coords0, coords1):

    h, w = size[1], size[2]
    mask0 = np.zeros((h, w), dtype=np.uint8)
    mask1 = np.zeros((h, w), dtype=np.uint8)
    for i, (coord0, coord1) in enumerate(zip(coords0, coords1)):
        x0, y0 = coord0
        x1, y1 = coord1
    
        x0, y0 = int(round(x0)), int(round(y0))
        x1, y1 = int(round(x1)), int(round(y1))
       
        if 0 <= x0 < w and 0 <= y0 < h:
            mask0[y0, x0] = i + 1
        if 0 <= x1 < w and 0 <= y1 < h:
            mask1[y1, x1] = i + 1
    return mask0, mask1


def select_multiple_points(points0, points1, num_points):
    
    N = len(points0)
    num_points = min(num_points, N)
    indices = np.random.choice(N, size=num_points, replace=False)
    selected_points0 = points0[indices]
    selected_points1 = points1[indices]
    return selected_points0, selected_points1

def generate_point_map_frames(size, coords0, coords1,visibility):

    h, w = size[1], size[2]
    mask0 = np.zeros((h, w), dtype=np.uint8)
    num_frames = coords1.shape[0]
    mask1 = np.zeros((num_frames, h, w), dtype=np.uint8)

    for i, coord0 in enumerate(coords0):
        x0, y0 = coord0
        x0, y0 = int(round(x0)), int(round(y0))
        if 0 <= x0 < w and 0 <= y0 < h:
            mask0[y0, x0] = i + 1

    for frame_idx in range(num_frames):
        coords_frame = coords1[frame_idx]
        for i, coord1 in enumerate(coords_frame):
            x1, y1 = coord1
            x1, y1 = int(round(x1)), int(round(y1))
            if 0 <= x1 < w and 0 <= y1 < h and visibility[frame_idx,i]==True:
                mask1[frame_idx, y1, x1] = i + 1

    return mask0, mask1



import numpy as np

def extract_patches(image, coords, patch_size):

    N = coords.shape[0]
    channels, H, W = image.shape
    patches = np.zeros((N, channels, patch_size, patch_size), dtype=image.dtype)
    half_size = patch_size // 2

    for i in range(N):
        x0, y0 = coords[i]
        x0 = int(round(x0))
        y0 = int(round(y0))

        # Define the patch region in the image
        x_start_img = x0 - half_size
        x_end_img = x0 + half_size + 1
        y_start_img = y0 - half_size
        y_end_img = y0 + half_size + 1

        # Define the region in the patch to fill
        x_start_patch = 0
        y_start_patch = 0
        x_end_patch = patch_size
        y_end_patch = patch_size

        # Adjust for boundaries
        if x_start_img < 0:
            x_start_patch = -x_start_img
            x_start_img = 0
        if y_start_img < 0:
            y_start_patch = -y_start_img
            y_start_img = 0
        if x_end_img > W:
            x_end_patch -= (x_end_img - W)
            x_end_img = W
        if y_end_img > H:
            y_end_patch -= (y_end_img - H)
            y_end_img = H

        # Calculate the actual sizes
        patch_height = y_end_patch - y_start_patch
        patch_width = x_end_patch - x_start_patch
        img_height = y_end_img - y_start_img
        img_width = x_end_img - x_start_img

        # Ensure the sizes match
        if patch_height != img_height or patch_width != img_width:
            min_height = min(patch_height, img_height)
            min_width = min(patch_width, img_width)
            y_end_patch = y_start_patch + min_height
            y_end_img = y_start_img + min_height
            x_end_patch = x_start_patch + min_width
            x_end_img = x_start_img + min_width

        # Assign the image patch to the patches array
        patches[i, :, y_start_patch:y_end_patch, x_start_patch:x_end_patch] = \
            image[:, y_start_img:y_end_img, x_start_img:x_end_img]

    return patches

def generate_point_feature_map_frames_naive(image, size, coords0, coords1, visibility, patch_size):

    channels, H, W = size
    num_frames = coords1.shape[0]
    N = coords0.shape[0]

    # Extract patches from the reference image at coords0
    patches = extract_patches(image, coords0, patch_size)
    half_size = patch_size // 2

    # Initialize the feature maps
    feature_maps = np.zeros((num_frames, channels, H, W), dtype=image.dtype)

    for frame_idx in range(num_frames):
        feature_map = np.zeros((channels, H, W), dtype=image.dtype)
        coords_frame = coords1[frame_idx]

        for i in range(N):
            if visibility[frame_idx, i]:
                x1, y1 = coords_frame[i]
                x1 = int(round(x1))
                y1 = int(round(y1))

                # Define the patch region in the feature map
                x_start_map = x1 - half_size
                x_end_map = x1 + half_size + 1
                y_start_map = y1 - half_size
                y_end_map = y1 + half_size + 1

                # Define the region in the patch to use
                x_start_patch = 0
                y_start_patch = 0
                x_end_patch = patch_size
                y_end_patch = patch_size

                # Adjust for boundaries
                if x_start_map < 0:
                    x_start_patch = -x_start_map
                    x_start_map = 0
                if y_start_map < 0:
                    y_start_patch = -y_start_map
                    y_start_map = 0
                if x_end_map > W:
                    x_end_patch -= (x_end_map - W)
                    x_end_map = W
                if y_end_map > H:
                    y_end_patch -= (y_end_map - H)
                    y_end_map = H

                # Calculate the actual sizes
                patch_height = y_end_patch - y_start_patch
                patch_width = x_end_patch - x_start_patch
                map_height = y_end_map - y_start_map
                map_width = x_end_map - x_start_map

                # Ensure the sizes match
                if patch_height != map_height or patch_width != map_width:
                    min_height = min(patch_height, map_height)
                    min_width = min(patch_width, map_width)
                    y_end_patch = y_start_patch + min_height
                    y_end_map = y_start_map + min_height
                    x_end_patch = x_start_patch + min_width
                    x_end_map = x_start_map + min_width

                # Place the patch into the feature map
                feature_map[:, y_start_map:y_end_map, x_start_map:x_end_map] = \
                    patches[i, :, y_start_patch:y_end_patch, x_start_patch:x_end_patch]

        feature_maps[frame_idx] = feature_map

    return feature_maps


import os
from PIL import Image
import numpy as np
from moviepy.editor import ImageSequenceClip

def export_gif_side_by_side_complete(ref_frame, sketches, frames, output_gif_path, supp_dir,fps):
    """
    Export frames into a GIF and an MP4 video with columns, and save individual frames and sketches.

    Args:
    - ref_frame (PIL.Image or np.ndarray): The reference image.
    - sketches (list): List of sketch images (as numpy arrays or PIL Image objects).
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - fps (int): Frames per second for the GIF and MP4.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_gif_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the base name of the output file (without extension)
    base_name = os.path.splitext(os.path.basename(output_gif_path))[0]

    # Create subdirectories for sketches and frames
    sketch_dir = os.path.join(supp_dir,"sketches")
    frame_dir = os.path.join(supp_dir,"frames")
    os.makedirs(sketch_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    pil_sketches = [Image.fromarray(sketch) if isinstance(sketch, np.ndarray) else sketch for sketch in sketches]
    ref_frame = Image.fromarray(ref_frame) if isinstance(ref_frame, np.ndarray) else ref_frame
    
    # Get dimensions of images
    width, height = pil_frames[0].size
    
    # Resize images
    resized_frames = [frame.resize((width, height)) for frame in pil_frames]
    resized_sketches = [sketch.resize((width, height)) for sketch in pil_sketches]
    ref_frame = ref_frame.resize((width, height))
    
    # Save each sketch frame
    for i, sketch in enumerate(resized_sketches):
        sketch_filename = os.path.join(sketch_dir, f"{base_name}_sketch_{i:04d}.png")
        sketch.save(sketch_filename)
    
    # Save each frame
    for i, frame in enumerate(resized_frames):
        frame_filename = os.path.join(frame_dir, f"{base_name}_frame_{i:04d}.png")
        frame.save(frame_filename)
    
    # Save reference frame
    ref_filename = os.path.join(supp_dir, f"{base_name}_reference.png")
    ref_frame.save(ref_filename)
    
    # Create a new image for each frame with the three columns
    column_frames = []
    for i, frame in enumerate(resized_frames):
        # Create an empty image with the total width for all three columns
        new_width = ref_frame.width + resized_sketches[i].width + frame.width
        combined_frame = Image.new('RGB', (new_width, height))
        
        # Paste the reference image, sketch, and frame into the new image
        combined_frame.paste(ref_frame, (0, 0))
        combined_frame.paste(resized_sketches[i], (ref_frame.width, 0))
        combined_frame.paste(frame, (ref_frame.width + resized_sketches[i].width, 0))
        
        column_frames.append(combined_frame)
    
    # Calculate frame duration in milliseconds based on fps
    frame_duration = int(1000 / fps)
    
    # Save the GIF with columns
    column_frames[0].save(output_gif_path,
                          format='GIF',
                          append_images=column_frames[1:],
                          save_all=True,
                          duration=frame_duration,
                          loop=0)
    
    # Save the MP4 video with the same content
    output_mp4_path = os.path.join(supp_dir , 'result.mp4')
    # Convert PIL Images to numpy arrays for moviepy
    video_frames = [np.array(frame) for frame in column_frames]
    clip = ImageSequenceClip(video_frames, fps=fps)
    clip.write_videofile(output_mp4_path, codec='libx264')



def export_gif_with_ref_complete(start_image, frames, end_image, reference_image, output_gif_path, supp_dir, fps):
    """
    Export a list of frames into a GIF with columns, save individual images and frames,
    and create an MP4 video, following the storage method of 'export_gif_side_by_side_complete'.

    Args:
    - start_image (PIL.Image or np.ndarray): The starting image.
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - end_image (PIL.Image or np.ndarray): The ending image.
    - reference_image (PIL.Image or np.ndarray): The reference image.
    - output_gif_path (str): Path to save the output GIF.
    - supp_dir (str): Directory to save supplementary files.
    - fps (int): Frames per second for the GIF and MP4.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_gif_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the base name of the output file (without extension)
    base_name = os.path.splitext(os.path.basename(output_gif_path))[0]

    # Create subdirectories for images and frames
    start_end_dir = os.path.join(supp_dir, "start_end_images")
    frame_dir = os.path.join(supp_dir, "frames")
    reference_dir = os.path.join(supp_dir, "reference")
    os.makedirs(start_end_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)
    
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    start_image = Image.fromarray(start_image) if isinstance(start_image, np.ndarray) else start_image
    end_image = Image.fromarray(end_image) if isinstance(end_image, np.ndarray) else end_image
    reference_image = Image.fromarray(reference_image) if isinstance(reference_image, np.ndarray) else reference_image
    
    # Get dimensions of images
    width, height = start_image.size
    
    # Resize images to match the height
    reference_image = reference_image.resize((reference_image.width, height))
    resized_frames = [frame.resize((frame.width, height)) for frame in pil_frames]
    
    # Save start_image, end_image, and reference_image
    start_image_filename = os.path.join(start_end_dir, f"{base_name}_start.png")
    start_image.save(start_image_filename)
    end_image_filename = os.path.join(start_end_dir, f"{base_name}_end.png")
    end_image.save(end_image_filename)
    reference_image_filename = os.path.join(reference_dir, f"{base_name}_reference.png")
    reference_image.save(reference_image_filename)
    
    # Save each frame
    for i, frame in enumerate(resized_frames):
        frame_filename = os.path.join(frame_dir, f"{base_name}_frame_{i:04d}.png")
        frame.save(frame_filename)
    
    # Create a new image for each frame with the columns
    column_frames = []
    for i, frame in enumerate(resized_frames):
        # Calculate the total width for all columns
        new_width = start_image.width + reference_image.width + end_image.width + frame.width
        combined_frame = Image.new('RGB', (new_width, height))
        
        # Paste the images into the combined frame
        combined_frame.paste(start_image, (0, 0))
        combined_frame.paste(reference_image, (start_image.width, 0))
        combined_frame.paste(end_image, (start_image.width + reference_image.width, 0))
        combined_frame.paste(frame, (start_image.width + reference_image.width + end_image.width, 0))
        
        column_frames.append(combined_frame)
    
    # Calculate frame duration in milliseconds based on fps
    frame_duration = int(1000 / fps)
    
    # Save the GIF with columns
    column_frames[0].save(output_gif_path,
                          format='GIF',
                          append_images=column_frames[1:],
                          save_all=True,
                          duration=frame_duration,
                          loop=0)
    
    # Save the MP4 video with the same content
    output_mp4_path = os.path.join(supp_dir, 'result.mp4')
    # Convert PIL Images to numpy arrays for moviepy
    video_frames = [np.array(frame) for frame in column_frames]
    clip = ImageSequenceClip(video_frames, fps=fps)
    clip.write_videofile(output_mp4_path, codec='libx264')


def export_gif_side_by_side_complete_ablation(ref_frame, sketches, frames, output_gif_path, supp_dir,fps):
    """
    Export frames into a GIF and an MP4 video with columns, and save individual frames and sketches.

    Args:
    - ref_frame (PIL.Image or np.ndarray): The reference image.
    - sketches (list): List of sketch images (as numpy arrays or PIL Image objects).
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - fps (int): Frames per second for the GIF and MP4.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_gif_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the base name of the output file (without extension)
    base_name = os.path.splitext(os.path.basename(output_gif_path))[0]

    # Create subdirectories for sketches and frames
    sketch_dir = os.path.join(supp_dir,"sketches")
    frame_dir = os.path.join(supp_dir,"frames")
    os.makedirs(sketch_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    pil_sketches = [Image.fromarray(sketch) if isinstance(sketch, np.ndarray) else sketch for sketch in sketches]
    ref_frame = Image.fromarray(ref_frame) if isinstance(ref_frame, np.ndarray) else ref_frame
    
    # Get dimensions of images
    width, height = pil_frames[0].size
    
    # Resize images
    resized_frames = [frame.resize((width, height)) for frame in pil_frames]
    resized_sketches = [sketch.resize((width, height)) for sketch in pil_sketches]
    ref_frame = ref_frame.resize((width, height))
    
    # Save each sketch frame
    for i, sketch in enumerate(resized_sketches):
        sketch_filename = os.path.join(sketch_dir, f"{base_name}_sketch_{i:04d}.png")
        sketch.save(sketch_filename)
    
    # Save each frame
    for i, frame in enumerate(resized_frames):
        frame_filename = os.path.join(frame_dir, f"{base_name}_frame_{i:04d}.png")
        frame.save(frame_filename)
    
    # Save reference frame
    ref_filename = os.path.join(supp_dir, f"{base_name}_reference.png")
    ref_frame.save(ref_filename)
    
    # Create a new image for each frame with the three columns
    column_frames = []
    rgb_frames = []
    for i, frame in enumerate(resized_frames):
        # Create an empty image with the total width for all three columns
        new_width = ref_frame.width + resized_sketches[i].width + frame.width
        combined_frame = Image.new('RGB', (new_width, height))
        
        # Paste the reference image, sketch, and frame into the new image
        combined_frame.paste(ref_frame, (0, 0))
        combined_frame.paste(resized_sketches[i], (ref_frame.width, 0))
        combined_frame.paste(frame, (ref_frame.width + resized_sketches[i].width, 0))
        
        column_frames.append(combined_frame)
        rgb_frames.append(frame)
    
    # Calculate frame duration in milliseconds based on fps
    frame_duration = int(1000 / fps)
    
    # Save the GIF with columns
    column_frames[0].save(output_gif_path,
                          format='GIF',
                          append_images=column_frames[1:],
                          save_all=True,
                          duration=frame_duration,
                          loop=0)
    
    # Save the MP4 video with the same content
    output_mp4_path = supp_dir+'.mp4'
    # Convert PIL Images to numpy arrays for moviepy
    video_frames = [np.array(frame) for frame in column_frames]
    rgb_frames = [np.array(frame) for frame in rgb_frames]
    clip = ImageSequenceClip(rgb_frames, fps=fps)
    clip.write_videofile(output_mp4_path, codec='libx264')