from torchvision import transforms
from datasets import video_transforms
from .ucf101_datasets import UCF101
from .dummy_datasets import DummyDataset
from .webvid_datasets import WebVid10M
from .videoswap_datasets import VideoSwapDataset
from .dl3dv_datasets import DL3DVDataset
from .pair_datasets import PairDataset
from .metric_datasets import MetricDataset
from .sakuga_ref_datasets import SakugaRefDataset

def get_dataset(args):
    if args.dataset not in ["encdec_images", "pair_dataset"]:
        temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1
    if args.dataset == 'sakuga_ref':
        temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval+args.ref_jump_frames) # 16 1
    if args.dataset == 'ucf101':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        ])
        dataset = UCF101(args, transform=transform_ucf101, temporal_sample=temporal_sample)
        return dataset

    elif args.dataset == 'dummy':
        size = (args.height, args.width)
        transform = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            # video_transforms.RandomHorizontalFlipVideo(),  # NOTE
            video_transforms.UCFCenterCropVideo(size=size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        ])

        dataset = DummyDataset(
            sample_frames=args.num_frames,
            base_folder=args.base_folder,
            temporal_sample=temporal_sample,
            transform=transform,
            seed=args.seed,
            file_list=args.file_list,
        )
        return dataset
    elif args.dataset == 'sakuga_ref':
        size = (args.height, args.width)
        transform = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            # video_transforms.RandomHorizontalFlipVideo(),  # NOTE
            video_transforms.UCFCenterCropVideo(size=size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        ])

        dataset = SakugaRefDataset(
                video_frames=args.num_frames,
                ref_jump_frames=args.ref_jump_frames,
                base_folder=args.base_folder,
                temporal_sample=temporal_sample,
                transform=transform,
                seed=args.seed,
                file_list=args.file_list,
        )
        return dataset     
    elif args.dataset == 'webvid':
        size = (args.height, args.width)
        transform = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            # video_transforms.RandomHorizontalFlipVideo(),  # NOTE
            video_transforms.UCFCenterCropVideo(size=size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        ])

        dataset = WebVid10M(
            sample_frames=args.num_frames,
            base_folder=args.base_folder,
            temporal_sample=temporal_sample,
            transform=transform,
            seed=args.seed,
        )
        return dataset

    elif args.dataset == 'videoswap':
        size = (args.height, args.width)
        transform = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            # video_transforms.RandomHorizontalFlipVideo(),
            # video_transforms.UCFCenterCropVideo(size=size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        ])

        dataset = VideoSwapDataset(
            width=args.width,
            height=args.height,
            sample_frames=args.num_frames,
            base_folder=args.base_folder,
            temporal_sample=temporal_sample,
            transform=transform,
            seed=args.seed
        )
        return dataset

    elif args.dataset == 'dl3dv':
        size = (args.height, args.width)
        # transform = transforms.Compose([
        #     video_transforms.ToTensorVideo(), # TCHW
        #     # video_transforms.RandomHorizontalFlipVideo(),
        #     # video_transforms.UCFCenterCropVideo(size=size),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        # ])

        dataset = DL3DVDataset(
            width=args.width,
            height=args.height,
            sample_frames=args.num_frames,
            base_folder=args.base_folder,
            file_list=args.file_list,
            temporal_sample=temporal_sample,
            # transform=transform,
            seed=args.seed,
        )
        return dataset

    elif args.dataset == "pair_dataset":
        # size = (args.height, args.width)
        # transform = transforms.Compose([
        #     video_transforms.ToTensorVideo(), # TCHW
        #     # video_transforms.RandomHorizontalFlipVideo(),
        #     video_transforms.UCFCenterCropVideo(size=size),
        #     # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        # ])

        dataset = PairDataset(
            # width=args.width,
            # height=args.height,
            # sample_frames=args.num_frames,
            base_folder=args.base_folder,
            # temporal_sample=temporal_sample,
            # transform=transform,
            # seed=args.seed,
            with_pair=args.with_pair,
        )
        return dataset

    elif args.dataset == "metric_dataset":

        dataset = MetricDataset(
            base_folder=args.base_folder,
        )
        return dataset

    else:
        raise NotImplementedError(args.dataset)
