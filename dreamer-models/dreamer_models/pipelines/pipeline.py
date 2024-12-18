import os
from importlib import import_module

from .. import utils


class BasePipeline:
    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LazyPipeline(BasePipeline):
    def __init__(self, pipeline, pipeline_info):
        self.pipeline = pipeline
        self.pipeline_info = pipeline_info
        self.device = None
        self.is_init = False

    def init_pipeline(self):
        if not self.is_init:
            self.pipeline = self.pipeline(**self.pipeline_info)
            if self.device is not None:
                self.pipeline.to(self.device)
            self.is_init = True

    def to(self, device):
        self.device = device
        if self.is_init:
            self.pipeline.to(device)
        return self

    def __call__(self, *args, **kwargs):
        self.init_pipeline()
        return self.pipeline(*args, **kwargs)


def get_pipelines():
    model_dir = utils.get_model_dir()
    models = {
        'depth_estimation/dpt/large': {
            '_class_name': 'vision.depth_estimation.pipeline_dpt.DPTForDepthEstimationPipeline',
            'model_name': 'Intel/dpt-large',
        },
        'depth_estimation/dpt/hybrid_midas': {
            '_class_name': 'vision.depth_estimation.pipeline_dpt.DPTForDepthEstimation2Pipeline',
            'model_name': os.path.join(model_dir, 'huggingface/models--Intel--dpt-hybrid-midas'),
        },
        'detection/grounding_dino/swint_ogc': {
            '_class_name': 'vision.detection.pipeline_grounding_dino.GroundingDINOPipeline',
            'config_path': os.path.join(model_dir, 'others/GroundingDINO/GroundingDINO_SwinT_OGC.py'),
            'model_path': os.path.join(model_dir, 'others/GroundingDINO/groundingdino_swint_ogc.pth'),
        },
        'edge_detection/canny': {
            '_class_name': 'vision.edge_detection.pipeline_canny.CannyPipeline',
        },
        'edge_detection/hed/bsds500': {
            '_class_name': 'vision.edge_detection.pipeline_hed.HEDPipeline',
            'model_name': 'lllyasviel/Annotators',
        },
        'edge_detection/lineart/sk_model': {
            '_class_name': 'vision.edge_detection.pipeline_lineart.LineartPipeline',
            'model_name': 'lllyasviel/Annotators',
        },
        'edge_detection/mlsd/mobilev2_large_512_fp32': {
            '_class_name': 'vision.edge_detection.pipeline_mlsd.MLSDPipeline',
            'model_name': 'lllyasviel/ControlNet',
        },
        'edge_detection/pidinet/table5': {
            '_class_name': 'vision.edge_detection.pipeline_pidinet.PidiNetPipeline',
            'model_name': 'lllyasviel/Annotators',
        },
        'face_restoration/codeformer/v0.1.0': {
            '_class_name': 'vision.face_restoration.pipeline_codeformer.CodeFormerPipeline',
            'model_path': os.path.join(model_dir, 'others/Codeformer/codeformer-v0.1.0.pth'),
        },
        'face_swap/insightface/inswapper_128': {
            '_class_name': 'vision.face_swap.pipeline_insightface.InsightFaceSwapPipeline',
            'model_path': os.path.join(model_dir, 'others/InsightFace/inswapper_128.onnx'),
        },
        'keypoints/openpose/body_hand_face': {
            '_class_name': 'vision.keypoints.pipeline_openpose.OpenPosePipeline',
            'model_name': 'lllyasviel/ControlNet',
        },
        'lane_detection/laneaf/dla34_640x288_batch2_v023': {
            '_class_name': 'vision.lane_detection.pipeline_laneaf.LaneAFPipeline',
            'model_path': os.path.join(model_dir, 'others/lane/dla34-640x288_batch2-v023.pth'),
        },
        'segmentation/segment_anything/vit_h_4b8939': {
            '_class_name': 'vision.segmentation.pipeline_segment_anything.SegmentAnythingPipeline',
            'model_path': os.path.join(model_dir, 'others/segment_anything/sam_vit_h_4b8939.pth'),
            'model_name': 'vit_h',
        },
        'segmentation/upernet/convnext_small': {
            '_class_name': 'vision.segmentation.pipeline_upernet.UperNetForSemanticSegmentationPipeline',
            'model_name': 'openmmlab/upernet-convnext-small',
        },
        'super_resolution/real_esrgan/x2plus': {
            '_class_name': 'vision.super_resolution.pipeline_real_esrgan.RealESRGANPipeline',
            'model_paths': [os.path.join(model_dir, 'others/RealESRGAN/RealESRGAN_x2plus.pth')],
            'model_names': ['x2plus'],
        },
        'super_resolution/real_esrgan/x4plus': {
            '_class_name': 'vision.super_resolution.pipeline_real_esrgan.RealESRGANPipeline',
            'model_paths': [os.path.join(model_dir, 'others/RealESRGAN/RealESRGAN_x4plus.pth')],
            'model_names': ['x4plus'],
        },
        'super_resolution/real_esrgan/x4plus_anime_6B': {
            '_class_name': 'vision.super_resolution.pipeline_real_esrgan.RealESRGANPipeline',
            'model_paths': [os.path.join(model_dir, 'others/RealESRGAN/RealESRGAN_x4plus_anime_6B.pth')],
            'model_names': ['x4plus_anime_6B'],
        },
        'super_resolution/real_esrgan/x2plus_x4plus': {
            '_class_name': 'vision.super_resolution.pipeline_real_esrgan.RealESRGANPipeline',
            'model_paths': [
                os.path.join(model_dir, 'others/RealESRGAN/RealESRGAN_x2plus.pth'),
                os.path.join(model_dir, 'others/RealESRGAN/RealESRGAN_x4plus.pth'),
            ],
            'model_names': ['x2plus', 'x4plus'],
        },
        'others/normal_bae/scannet': {
            '_class_name': 'vision.others.pipeline_normal_bae.NormalBaePipeline',
            'model_name': 'lllyasviel/Annotators',
        },
        'others/shuffle/content': {
            '_class_name': 'vision.others.pipeline_shuffle.ContentShufflePipeline',
        },
    }
    return models


def load_pipeline(pipeline_name, lazy=False, **kwargs):
    pipelines = get_pipelines()
    pipeline_info = pipelines[pipeline_name]
    pipeline_info.update(kwargs)
    parts = pipeline_info.pop('_class_name').split('.')
    module_name = '.'.join(parts[:-1])
    module = import_module('giga_models.pipelines.' + module_name)
    pipeline = getattr(module, parts[-1])
    if lazy:
        return LazyPipeline(pipeline, pipeline_info)
    else:
        return pipeline(**pipeline_info)


def list_pipelines():
    pipelines = get_pipelines()
    return list(pipelines.keys())
