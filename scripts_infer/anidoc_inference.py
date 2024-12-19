import sys

from pyparsing import col
sys.path.insert(0,".")

import argparse
from packaging import version
import glob
import os
from LightGlue.lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from LightGlue.lightglue.utils import load_image, rbd
from cotracker.predictor import CoTrackerPredictor, sample_trajectories, generate_gassian_heatmap, sample_trajectories_with_ref
import torch
from diffusers.utils.import_utils import is_xformers_available

from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from pipelines.AniDoc import AniDocPipeline
from models_diffusers.controlnet_svd import ControlNetSVDModel
from diffusers.utils import load_image, export_to_video, export_to_gif
import time
from lineart_extractor.annotator.lineart import LineartDetector
import numpy as np
from PIL import Image
from utils import load_images_from_folder,export_gif_with_ref,export_gif_side_by_side,extract_frames_from_video,safe_round,select_multiple_points,generate_point_map,generate_point_map_frames,export_gif_side_by_side_complete,export_gif_side_by_side_complete_ablation
import random
import torchvision.transforms as T
from LightGlue.lightglue import viz2d
import matplotlib.pyplot as plt
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from torchvision.transforms import PILToTensor



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="pretrained_weights/stable-video-diffusion-img2vid-xt", help="Path to the input image.")

    parser.add_argument(
        "--pretrained_unet", type=str, help="Path to the input image.",
 
        default="pretrained_weights/anidoc"

    )
    parser.add_argument(
        "--controlnet_model_name_or_path", type=str, help="Path to the input image.",
      default="pretrained_weights/anidoc/controlnet"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output video.")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")

    parser.add_argument("--noise_aug", type=float, default=0.02)

    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--all_sketch",action="store_true",help="all_sketch")
    parser.add_argument("--not_quant_sketch",action="store_true",help="not_quant_sketch")
    parser.add_argument("--repeat_sketch",action="store_true",help="not_quant_sketch")
    parser.add_argument("--matching",action="store_true",help="add keypoint matching")
    parser.add_argument("--tracking",action="store_true",help="tracking keypoint")
    parser.add_argument("--repeat_matching",action="store_true",help="not tracking, but just simply repeat")
    parser.add_argument("--tracker_point_init", type=str, default='gaussion', choices=['dift', 'gaussion', 'both'], help="Regular grid size")
    parser.add_argument(
        "--tracker_shift_grid",
        type=int, default=0, choices=[0, 1],
        help="shift the grid for the tracker")
    parser.add_argument("--tracker_grid_size", type=int, default=8, help="Regular grid size")
    parser.add_argument(
        "--tracker_grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--tracker_backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument("--control_image", type=str, default=None, help="Path to the output video.")
    parser.add_argument("--ref_image", type=str, default=None, help="Path to the output video.")
    parser.add_argument("--max_points", type=int, default=10)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()
    dtype = torch.float16


    unet = UNetSpatioTemporalConditionModel.from_pretrained(

            args.pretrained_unet,
            subfolder="unet",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            custom_resume=True,
        )

    unet.to("cuda",dtype)
 
    if args.controlnet_model_name_or_path:

        controlnet = ControlNetSVDModel.from_pretrained(
            args.controlnet_model_name_or_path,
        )
    else:

        controlnet = ControlNetSVDModel.from_unet(
            unet,
            conditioning_channels=8
        )
    controlnet.to("cuda",dtype)
    if is_xformers_available():
        import xformers
        xformers_version = version.parse(xformers.__version__)
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError(
            "xformers is not available. Make sure it is installed correctly")

    pipe = AniDocPipeline.from_pretrained(
 
        args.pretrained_model_name_or_path,
        unet=unet,
        controlnet=controlnet,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to("cuda")
    device = "cuda"
    detector = LineartDetector(device) 
    extractor = SuperPoint(max_num_keypoints=2000).eval().to(device)  # load the extractor
    matcher = LightGlue(features='superpoint').eval().to(device)  # load the matcher

    tracker = CoTrackerPredictor(
        checkpoint="pretrained_weights/cotracker2.pth",
        shift_grid=args.tracker_shift_grid,
    )
    tracker.requires_grad_(False)
    tracker.to(device, dtype=torch.float32)


    width, height = args.width, args.height

    # image = load_image('dalle3_cat.jpg')
    if args.output_dir is None:
            args.output_dir = "results"
    os.makedirs(args.output_dir, exist_ok=True)

    image_folder_list=[
        'data_test/sample1.mp4',
    ]

    ref_image_list=[
        "data_test/sample1.png",
    ]
    if args.ref_image is not None and args.control_image is not None:
        ref_image_list=[args.ref_image]
        image_folder_list=[args.control_image]

                    

    for val_id ,each_sample in enumerate(image_folder_list):
        if os.path.isdir(each_sample):
  
            control_images=load_images_from_folder(each_sample)
        elif each_sample.endswith(".mp4"):
            control_images = extract_frames_from_video(each_sample)
        ref_image=load_image(ref_image_list[val_id]).resize((width, height))


    #resize:
        for j, each in enumerate(control_images):
            control_images[j]=control_images[j].resize((width, height))

    # load image from folder
        if args.all_sketch:
                controlnet_image=[]
                for k in range(len(control_images)):
                    sketch=control_images[k]
                    sketch = np.array(sketch)
                    sketch=detector(sketch,coarse=False)
                    sketch=np.repeat(sketch[:, :, np.newaxis], 3, axis=2)
                    if args.not_quant_sketch:
                        pass
                    else:
                        sketch= (sketch > 200).astype(np.uint8)*255
                    sketch = Image.fromarray(sketch).resize((width, height))
           
                    controlnet_image.append(sketch)

                controlnet_sketch_condition = [T.ToTensor()(img).unsqueeze(0) for img in controlnet_image]
                controlnet_sketch_condition = torch.cat(controlnet_sketch_condition, dim=0).unsqueeze(0).to(device, dtype=torch.float16)
                controlnet_sketch_condition = (controlnet_sketch_condition - 0.5) / 0.5  #(1,14,3,h,w)
                # matching condition
                with torch.no_grad():
                    ref_img_value = T.ToTensor()(ref_image).to(device, dtype=torch.float16)  #(0,1)
                    
                    ref_img_value = ref_img_value.to(torch.float32)
                    current_img= T.ToTensor()(controlnet_image[0]).to(device, dtype=torch.float16)  #(0,1)
                    current_img = current_img.to(torch.float32)
                    feats0 = extractor.extract(ref_img_value)  
                    feats1 = extractor.extract(current_img)
                    matches01 = matcher({'image0': feats0, 'image1': feats1})
                    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  
                    matches = matches01['matches']  
                    points0 = feats0['keypoints'][matches[..., 0]]  
                    points1 = feats1['keypoints'][matches[..., 1]]
                    points0 = points0.cpu().numpy()
                    # points0_org=points0.copy()
                    points1 = points1.cpu().numpy()
                
                    points0 = safe_round(points0, current_img.shape)
                    points1 = safe_round(points1, current_img.shape)

                    num_points = min(50, points0.shape[0])
                    points0,points1 = select_multiple_points(points0, points1, num_points)
                    mask1, mask2 = generate_point_map(size=current_img.shape, coords0=points0, coords1=points1)
                    # import ipdb;ipdb.set_trace()
                    point_map1=torch.from_numpy(mask1)
                    point_map2=torch.from_numpy(mask2)
                    point_map1 = point_map1.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)
                    point_map2 = point_map2.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)
                    point_map=torch.cat([point_map1,point_map2],dim=2)
                    conditional_pixel_values=ref_img_value.unsqueeze(0).unsqueeze(0)
                    conditional_pixel_values = (conditional_pixel_values - 0.5) / 0.5
                   
                    point_map_with_ref= torch.cat([point_map,conditional_pixel_values],dim=2)
                    original_shape = list(point_map_with_ref.shape)
                    new_shape = original_shape.copy()
                    new_shape[1] = args.num_frames-1

                    if args.repeat_matching:
                        matching_controlnet_image=point_map_with_ref.repeat(1,args.num_frames,1,1,1)
                        controlnet_condition=torch.cat([controlnet_sketch_condition, matching_controlnet_image], dim=2)
                    elif args.tracking:
                        with torch.no_grad():
                            video_for_tracker = (controlnet_sketch_condition * 0.5 + 0.5) * 255.
                            queries = np.insert(points1,0,0,axis=1)
                            queries =torch.from_numpy(queries).to(device,torch.float).unsqueeze(0)
                           
                            if queries.shape[1]==0:
                                pred_tracks_sampled=None
                                points0_sampled = None
                            else:
                                pred_tracks, pred_visibility = tracker(
                                    video_for_tracker.to(dtype=torch.float32),
                                    queries=queries,
                                    grid_size=args.tracker_grid_size,  # 8
                                    grid_query_frame=args.tracker_grid_query_frame,  # 0
                                    backward_tracking=args.tracker_backward_tracking,  # False
                                    # segm_mask=segm_mask,
                                )
                                pred_tracks_sampled, pred_visibility_sampled,points0_sampled = sample_trajectories_with_ref(
                                    pred_tracks.cpu(), pred_visibility.cpu(), torch.from_numpy(points0).unsqueeze(0).cpu(),
                                    max_points=args.max_points,
                                    motion_threshold=1,
                                    vis_threshold=3,
                                )  
                            if pred_tracks_sampled is None:
                                mask1 = np.zeros((args.height, args.width), dtype=np.uint8)
                                mask2 = np.zeros((args.num_frames,args.height, args.width), dtype=np.uint8)
                            else:
                                pred_tracks_sampled = pred_tracks_sampled.squeeze(0).cpu().numpy()
                                pred_visibility_sampled =pred_visibility_sampled.squeeze(0).cpu().numpy()
                                points0_sampled =points0_sampled.squeeze(0).cpu().numpy()
                                for frame_id in range(args.num_frames):
                                        pred_tracks_sampled[frame_id] = safe_round(pred_tracks_sampled[frame_id],current_img.shape)
                                points0_sampled = safe_round(points0_sampled,current_img.shape)
                                
                                mask1, mask2 = generate_point_map_frames(size=current_img.shape, coords0=points0_sampled,coords1=pred_tracks_sampled,visibility=pred_visibility_sampled)
                             
                            point_map1=torch.from_numpy(mask1)
                            point_map2=torch.from_numpy(mask2)
                            point_map1 = point_map1.unsqueeze(0).unsqueeze(0).repeat(1,args.num_frames,1,1,1).to(device, dtype=torch.float16)
                            point_map2 = point_map2.unsqueeze(0).unsqueeze(2).to(device, dtype=torch.float16)
                            point_map=torch.cat([point_map1,point_map2],dim=2)   

                            conditional_pixel_values_repeat=conditional_pixel_values.repeat(1,14,1,1,1)
                        
                            point_map_with_ref= torch.cat([point_map,conditional_pixel_values_repeat],dim=2)  
                            controlnet_condition= torch.cat([controlnet_sketch_condition, point_map_with_ref], dim=2)   
                    else:
                        zero_tensor = torch.zeros(new_shape).to(device, dtype=torch.float16)
                        matching_controlnet_image=torch.cat((point_map_with_ref,zero_tensor),dim=1)
                        controlnet_condition = torch.cat([controlnet_sketch_condition, matching_controlnet_image], dim=2)
                            

                    ref_base_name=os.path.splitext(os.path.basename(ref_image_list[val_id]))[0]  
                    sketch_base_name=os.path.splitext(os.path.basename(each_sample))[0] 
                    supp_dir=os.path.join(args.output_dir,ref_base_name+"_"+sketch_base_name)  
                    os.makedirs(supp_dir, exist_ok=True)  
          
        elif args.repeat_sketch:
            controlnet_image=[]
            for i_2 in range(int(len(control_images)/2)):
                sketch=control_images[0]
                sketch = np.array(sketch)
                sketch=detector(sketch,coarse=False)
                sketch=np.repeat(sketch[:, :, np.newaxis], 3, axis=2)
               
                if args.not_quant_sketch:
                    pass
                else:
                    sketch= (sketch > 200).astype(np.uint8)*255
                sketch = Image.fromarray(sketch)
                controlnet_image.append(sketch)
            for i_3 in range(int(len(control_images)/2)):
                sketch=control_images[-1]


                
                sketch = np.array(sketch)
                sketch=detector(sketch,coarse=False)
                sketch=np.repeat(sketch[:, :, np.newaxis], 3, axis=2)
                
                if args.not_quant_sketch:
                    pass
                else:
                    sketch= (sketch > 200).astype(np.uint8)*255
                sketch = Image.fromarray(sketch)
               
                controlnet_image.append(sketch)


                
        generator = torch.manual_seed(args.seed)


        with torch.inference_mode():
            video_frames = pipe(
                ref_image, 
                controlnet_condition,
                height=args.height,
                width=args.width,
                num_frames=14,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=7,
                noise_aug_strength=0.02,
                generator=generator,
            ).frames[0]   




        out_file = supp_dir+'.mp4'


        if args.all_sketch:
        

            export_gif_side_by_side_complete_ablation(ref_image,controlnet_image,video_frames,out_file.replace('.mp4','.gif'),supp_dir,6)
  
        elif args.repeat_sketch:
            export_gif_with_ref(control_images[0],video_frames,controlnet_image[-1],controlnet_image[0],out_file.replace('.mp4','.gif'),6)











