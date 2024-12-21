# AniDoc: Animation Creation Made Easier
<a href="https://yihao-meng.github.io/AniDoc_demo/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/pdf/2412.14173"><img src="https://img.shields.io/badge/arXiv-2404.12.14173-b31b1b.svg"></a>



https://github.com/user-attachments/assets/99e1e52a-f0e1-49f5-b81f-e787857901e4




> <a href="https://yihao-meng.github.io/AniDoc_demo">**AniDoc: Animation Creation Made Easier**</a>
>

[Yihao Meng](https://yihao-meng.github.io/)<sup>1,2</sup>, [Hao Ouyang](https://ken-ouyang.github.io/)<sup>2</sup>, [Hanlin Wang](https://openreview.net/profile?id=~Hanlin_Wang2)<sup>3,2</sup>, [Qiuyu Wang](https://github.com/qiuyu96)<sup>2</sup>, [Wen Wang](https://github.com/encounter1997)<sup>4,2</sup>, [Ka Leong Cheng](https://felixcheng97.github.io/)<sup>1,2</sup> , [Zhiheng Liu](https://johanan528.github.io/)<sup>5</sup>, [Yujun Shen](https://shenyujun.github.io/)<sup>2</sup>, [Huamin Qu](http://www.huamin.org/index.htm/)<sup>†,2</sup><br>
<sup>1</sup>HKUST <sup>2</sup>Ant Group <sup>3</sup>NJU <sup>4</sup>ZJU <sup>5</sup>HKU <sup>†</sup>corresponding author

> AniDoc colorizes a sequence of sketches based on a character design reference with high fidelity, even when the sketches significantly differ in pose and scale.  
</p>

**Strongly recommend seeing our [demo page](https://yihao-meng.github.io/AniDoc_demo).**


## Showcases:
<p style="text-align: center;">
  <img src="figure/showcases/image1.gif" alt="GIF" />
</p>
<p style="text-align: center;">
  <img src="figure/showcases/image2.gif" alt="GIF" />
</p>
<p style="text-align: center;">
  <img src="figure/showcases/image3.gif" alt="GIF" />
</p>
<p style="text-align: center;">
  <img src="figure/showcases/image4.gif" alt="GIF" />
</p>

## Flexible Usage:
### Same Reference with Varying Sketches
<div style="display: flex; flex-direction: column; align-items: center; gap: 20px;">
<img src="figure/showcases/image29.gif" alt="GIF Animation">
<img src="figure/showcases/image30.gif" alt="GIF Animation">
<img src="figure/showcases/image31.gif" alt="GIF Animation"  style="margin-bottom: 40px;"> 
<div style="text-align:center; margin-top: -50px; margin-bottom: 70px;font-size: 18px; letter-spacing: 0.2px;">
        <em>Satoru Gojo from Jujutsu Kaisen</em>
</div>
</div>

### Same Sketch with Different References.

<div style="display: flex; flex-direction: column; align-items: center; gap: 20px;">
<img src="figure/showcases/image33.gif" alt="GIF Animation" >

<img src="figure/showcases/image34.gif" alt="GIF Animation" >
<img src="figure/showcases/image35.gif" alt="GIF Animation" style="margin-bottom: 40px;"> 
<div style="text-align:center; margin-top: -50px; margin-bottom: 70px;font-size: 18px; letter-spacing: 0.2px;">
        <em>Anya Forger from Spy x Family</em>
</div>
</div>

## TODO List

- [x] Release the paper and demo page. Visit [https://yihao-meng.github.io/AniDoc_demo/](https://yihao-meng.github.io/AniDoc_demo/) 
- [x] Release the inference code.
- [ ] Build Gradio Demo
- [ ] Release the training code.
- [ ] Release the sparse sketch setting interpolation code.


## Requirements:
The training is conducted on 8 A100 GPUs (80GB VRAM), the inference is tested on RTX 5000 (32GB VRAM). In our test, the inference requires about 14GB VRAM.
## Environment
All the tests are conducted in Linux. We suggest running our code in Linux. To set up our environment in Linux, please run:
```
sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg
conda create -n anidoc python=3.8 -y
conda activate anidoc
pip install ipykernel
python -m ipykernel install --user --name anidoc --display-name "anidoc"
```

```
git clone https://huggingface.co/spaces/svjack/AniDoc && cd AniDoc
pip install -r requirements.txt
python gradio_app.py
```
- OR
```
git clone https://github.com/svjack/AniDoc.git
cd AniDoc
bash install.sh
```

## Checkpoints
1. please download the pre-trained stable video diffusion (SVD) checkpoints from [here](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/tree/main), and put the whole folder under `pretrained_weight`, it should look like `./pretrained_weights/stable-video-diffusion-img2vid-xt`
2. please download the checkpoint for our Unet and ControlNet from [here](https://huggingface.co/Yhmeng1106/anidoc/tree/main), and put the whole folder as `./pretrained_weights/anidoc`.
3. please download the co_tracker checkpoint from [here](https://huggingface.co/facebook/cotracker/blob/main/cotracker2.pth) and put it as  `./pretrained_weights/cotracker2.pth`.
   
```bash
mkdir pretrained_weights

git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
cp -r stable-video-diffusion-img2vid-xt pretrained_weights

git clone https://huggingface.co/Yhmeng1106/anidoc
cp -r anidoc/anidoc pretrained_weights

#git clone https://huggingface.co/facebook/cotracker
#cp -r cotracker/cotracker2.pth pretrained_weights
wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth?download=true -O cotracker2.pth
cp cotracker2.pth pretrained_weights
```

## Video prepare 
```python
from moviepy.editor import VideoFileClip, ImageSequenceClip
import os
import shutil
import numpy as np

def extract_frames_and_save(input_video_path, output_video_path, target_frames=14):
    """
    将视频的总帧数调整为指定的帧数，并保存到本地。

    :param input_video_path: 输入视频文件的路径
    :param output_video_path: 输出视频文件的路径
    :param target_frames: 目标帧数（默认为 14）
    """
    try:
        # 创建临时路径
        temp_dir = "temp_frames"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # 加载视频文件
        video_clip = VideoFileClip(input_video_path)

        # 提取所有帧并保存到临时路径
        total_frames = int(video_clip.fps * video_clip.duration)
        for i, frame in enumerate(video_clip.iter_frames()):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            if i < total_frames:  # 只保存有效帧
                video_clip.save_frame(frame_path, t=i / video_clip.fps)

        # 选取 14 帧
        frame_files = sorted(os.listdir(temp_dir))
        selected_frames = np.linspace(0, len(frame_files) - 1, target_frames, dtype=int)
        selected_frame_files = [frame_files[i] for i in selected_frames]

        # 读取选取的帧
        frames = [os.path.join(temp_dir, frame) for frame in selected_frame_files]

        # 合并为新视频
        new_clip = ImageSequenceClip(frames, fps=video_clip.fps)
        new_clip.write_videofile(output_video_path, codec="libx264")

        print(f"视频已成功保存到: {output_video_path}")
    except Exception as e:
        print(f"处理视频时出错: {e}")
    finally:
        # 关闭视频对象
        if 'video_clip' in locals():
            video_clip.close()
        if 'new_clip' in locals():
            new_clip.close()

        # 删除临时路径
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# 示例调用
input_video = "刻晴摇_short.mp4"  # 替换为你的输入视频文件路径
output_video = "刻晴摇_short_14fps.mp4"  # 替换为你的输出视频文件路径
extract_frames_and_save(input_video, output_video, target_frames=14)
```

## Generate Your Animation!
To colorize the target lineart sequence with a specific character design, you can run the following command:
```python
python scripts_infer/anidoc_inference.py --all_sketch --matching --tracking --control_image '刻晴摇_short_14fps.mp4' --ref_image '刻晴白背景.png' --output_dir 'results' --max_point 10
```
- OR
```
bash  scripts_infer/anidoc_inference.sh
```


We provide some test cases in  `data_test` folder. You can also try our model with your own data. You can change the lineart sequence and corresponding character design in the script `anidoc_inference.sh`, where `--control_image` refers to the lineart sequence and `--ref_image` refers to the character design. 



## Citation:
Don't forget to cite this source if it proves useful in your research!
```bibtex
@article{meng2024anidoc,
      title={AniDoc: Animation Creation Made Easier},
      author={Yihao Meng and Hao Ouyang and Hanlin Wang and Qiuyu Wang and Wen Wang and Ka Leong Cheng and Zhiheng Liu and Yujun Shen and Huamin Qu},
      journal={arXiv preprint arXiv:2412.14173},
      year={2024}
}

```
