import torch
import time
import os
import folder_paths
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, T2IAdapter
from .pipeline_t2i_adapter import PhotoMakerStableDiffusionXLAdapterPipeline
from huggingface_hub import hf_hub_download
from .style_template import styles
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import gradio as gr
from .face_utils import FaceAnalysis2, analyze_faces
from torchvision import transforms
import torchvision.transforms.v2 as T

# global variable
# photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
device = "cuda" if torch.cuda.is_available() else "mps"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"
face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_detector.prepare(ctx_id=0, det_size=(640, 640))
#torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch_dtype = torch.float16
adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch_dtype, variant="fp16"
).to(device)
tensor = torch.rand(3, 256, 256)


def tensor_to_image(first_image):
    # 提取第一个图像，现在它是 [C, H, W]
    if first_image.dtype != torch.float32:
        first_image = first_image.to(torch.float32)

    if first_image.max() > 1.0:
        first_image = first_image / 255.0
    first_image = first_image.permute(2, 0, 1)
    # 转换为 PIL 图像并转换为 numpy 数组
    try:
        pil_image = T.ToPILImage()(first_image)
        pil_image_rgb = pil_image.convert('RGB')
        image_np = np.array(pil_image_rgb)
        print(image_np.shape)  # Outputs shape (1104, 828, 3) if successful
    except Exception as e:
        print(f'Error converting image: {e}')
    return T.ToPILImage()(first_image).convert('RGB')

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative


class BaseModelLoader_fromhub_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "SG161222/RealVisXL_V3.0"})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "PhotoMaker"

    def load_model(self, base_model_path):
        # Code to load the base model
        pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        return [pipe]


class BaseModelLoader_local_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),)
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "PhotoMaker"

    def load_model(self, ckpt_name):
        # Code to load the base model
        if not ckpt_name:
            raise ValueError("Please provide the ckpt_name parameter with the name of the checkpoint file.")

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")

        pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_single_file(
            pretrained_model_link_or_path=ckpt_path,
            adapter = adapter,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        return [pipe]


class PhotoMakerAdapterLoader_fromhub_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "TencentARC/PhotoMaker-V2"}),
                "filename": ("STRING", {"default": "photomaker-v2.bin"}),
                "pipe": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_photomaker_adapter"
    CATEGORY = "PhotoMaker"

    def load_photomaker_adapter(self, repo_id, filename, pipe):
        # 使用hf_hub_download方法获取PhotoMaker文件的路径
        photomaker_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model"
        )

        # 加载PhotoMaker检查点
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img",
            pm_version="v2",
        )
        pipe.id_encoder.to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora()
        pipe.to(device)
        return [pipe]


class PhotoMakerAdapterLoader_local_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pm_model_path": ("STRING", {"default": "enter your photomaker model path"}),
                "filename": ("STRING", {"default": "photomaker-v1.bin"}),
                "pipe": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_photomaker_adapter"
    CATEGORY = "PhotoMaker"

    def load_photomaker_adapter(self, pm_model_path, filename, pipe):
        # 拼接完整的模型路径
        photomaker_path = os.path.join(pm_model_path, filename)

        # 加载PhotoMaker检查点
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img"
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        return [pipe]


class LoRALoader_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "lora_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "display": "slider"}),
                "pipe": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "PhotoMaker"

    def load_lora(self, lora_name, lora_weight, pipe):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_name_processed = os.path.basename(lora_path).replace(".safetensors", "")

        # 解融合之前的 LoRA
        pipe.unfuse_lora()

        # 卸载之前加载的 LoRA 权重
        pipe.unload_lora_weights()

        # 重新加载新的 LoRA 权重
        unique_adapter_name = f"photomaker_{int(time.time())}"
        pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path),
                               adapter_name=unique_adapter_name)

        # 设置适配器和权重
        adapter_weights = [1.0, lora_weight]
        pipe.set_adapters(["photomaker", unique_adapter_name], adapter_weights=adapter_weights)

        # 融合 LoRA
        pipe.fuse_lora()

        return [pipe]


class ImagePreprocessingNode:
    def __init__(self, ref_image=None, ref_images_path=None, mode="direct_Input"):
        self.ref_image = ref_image
        self.ref_images_path = ref_images_path
        self.mode = mode

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_images_path": ("STRING", {"default": "path/to/images"}),  # 图像文件夹路径
                "mode": (["direct_Input", "path_Input"], {"default": "direct_Input"})  # 选择模式
            },
            "optional": {
                "ref_image": ("IMAGE",)  # 直接输入图像（可选）
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_image"
    CATEGORY = "PhotoMaker"

    def preprocess_image(self, ref_image=None, ref_images_path=None, mode="direct_Input"):
        # 使用传入的参数更新类属性
        ref_image = ref_image if ref_image is not None else ref_image
        ref_images_path = ref_images_path if ref_images_path is not None else ref_images_path
        mode = mode

        if mode == "direct_Input" and ref_image is not None:
            # 直接图像处理
            pil_images = []
            for image in ref_image:
                image_np = (255. * image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                pil_images.append(pil_image)
            return pil_images
        elif mode == "path_Input":
            # 路径输入图像
            image_basename_list = os.listdir(ref_images_path)
            image_path_list = [
                os.path.join(ref_images_path, basename)
                for basename in image_basename_list
                if
                not basename.startswith('.') and basename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
            ]
            return [load_image(image_path) for image_path in image_path_list]
        else:
            raise ValueError("Invalid mode. Choose 'direct_Input' or 'path_Input'.")


'''
class CompositeImageGenerationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "sci-fi, closeup portrait photo of a man img wearing the sunglasses in Iron man suit, face, slim body, high quality, film grain", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth", "multiline": True}),
                "style_name": (STYLE_NAMES, {"default": DEFAULT_STYLE_NAME}),
                "style_strength_ratio": ("INT", {"default": 20, "min": 1, "max": 50, "display": "slider"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0, "max": 10}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}), 
                "pipe": ("MODEL",),
                "pil_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PhotoMaker"

    def generate_image(self, style_name, style_strength_ratio, steps, seed, prompt, negative_prompt, guidance_scale, batch_size, pil_image, pipe, width, height):
        # Code for the remaining process including style template application, merge step calculation, etc.
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
        
        start_merge_step = int(float(style_strength_ratio) / 100 * steps)
        if start_merge_step > 30:
            start_merge_step = 30

        generator = torch.Generator(device=device).manual_seed(seed)

        output = pipe(
            prompt=prompt,
            input_id_images=[pil_image],
            negative_prompt=negative_prompt,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=guidance_scale, 
            width=width,
            height=height,
            return_dict=False
        )

        # 检查输出类型并相应处理
        if isinstance(output, tuple):
            # 当返回的是元组时，第一个元素是图像列表
            images_list = output[0]
        else:
            # 如果返回的是 StableDiffusionXLPipelineOutput，需要从中提取图像
            images_list = output.images

        # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
        images_tensors = []
        for img in images_list:
            # 将 PIL.Image 转换为 numpy.ndarray
            img_array = np.array(img)
            # 转换 numpy.ndarray 为 torch.Tensor
            img_tensor = torch.from_numpy(img_array).float() / 255.
            # 转换图像格式为 CHW (如果需要)
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            # 添加批次维度并转换为 NHWC
            img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            images_tensors.append(img_tensor)

        if len(images_tensors) > 1:
            output_image = torch.cat(images_tensors, dim=0)
        else:
            output_image = images_tensors[0]

        return (output_image,)
'''


# 拆分生成块
class Prompt_Style:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "sci-fi, closeup portrait photo of a man img wearing the sunglasses in Iron man suit, face, slim body, high quality, film grain",
                    "multiline": True}),
                "negative_prompt": ("STRING", {
                    "default": "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
                    "multiline": True}),
                "style_name": (STYLE_NAMES, {"default": DEFAULT_STYLE_NAME})
            }
        }

    RETURN_TYPES = ('STRING', 'STRING',)
    RETURN_NAMES = ('positive_prompt', 'negative_prompt',)
    FUNCTION = "prompt_style"
    CATEGORY = "PhotoMaker"

    def prompt_style(self, style_name, prompt, negative_prompt):
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        return prompt, negative_prompt


class NEWCompositeImageGenerationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "forceInput": True}),
                "negative": ("STRING", {"multiline": True, "forceInput": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "display": "slider"}),
                "style_strength_ratio": ("INT", {"default": 20, "min": 1, "max": 50, "display": "slider"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0, "max": 10, "display": "slider"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "use_doodle": ("INT",{"default": 0, "min": 0, "max": 1, "display": "slider"}),
                "sketch_image": ("IMAGE",),
                "adapter_conditioning_scale": ("INT",{"default": 1, "min": 1, "max": 4, "display": "slider"}),
                "adapter_conditioning_factor": ("INT",{"default": 1, "min": 1, "max": 4, "display": "slider"}),
                "pipe": ("MODEL",),
                "pil_image": ("IMAGE",)

            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PhotoMaker"

    def generate_image(self, steps, seed, positive, negative, style_strength_ratio, guidance_scale, batch_size,
                        width, height, use_doodle
                       , sketch_image, adapter_conditioning_scale, adapter_conditioning_factor,pipe,pil_image):
        # Code for the remaining process including style template application, merge step calculation, etc.

        if use_doodle == 1:
            sketch_image = sketch_image["composite"]
            r, g, b, a = sketch_image.split()
            sketch_image = a.convert("RGB")
            sketch_image = TF.to_tensor(sketch_image) > 0.5  # Inversion
            sketch_image = TF.to_pil_image(sketch_image.to(torch.float32))
            adapter_conditioning_scale = adapter_conditioning_scale
            adapter_conditioning_factor = adapter_conditioning_factor
        else:
            adapter_conditioning_scale = 0.
            adapter_conditioning_factor = 0.
            sketch_image = None
        input_id_images = []
        for v in range(pil_image.shape[0]):
            img_tensor = pil_image[v]
            pil_image_new = tensor_to_image(img_tensor)
            input_id_images.append(load_image(pil_image_new))
        id_embed_list = []
        for img in input_id_images:
            img = np.array(img)
            img = img[:, :, ::-1]
            faces = analyze_faces(face_detector, img)
            if len(faces) > 0:
                id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))

        if len(id_embed_list) == 0:
            raise gr.Error(f"No face detected, please update the input face image(s)")

        id_embeds = torch.stack(id_embed_list)

        start_merge_step = int(float(style_strength_ratio) / 100 * steps)
        if start_merge_step > 30:
            start_merge_step = 30

        generator = torch.Generator(device=device).manual_seed(seed)

        output = pipe(
            prompt=positive,
            input_id_images=input_id_images,
            negative_prompt=negative,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            id_embeds=id_embeds,
            image=sketch_image,
            adapter_conditioning_scale=adapter_conditioning_scale,
            adapter_conditioning_factor=adapter_conditioning_factor,
            return_dict=False
        )

        # 检查输出类型并相应处理
        if isinstance(output, tuple):
            # 当返回的是元组时，第一个元素是图像列表
            images_list = output[0]
        else:
            # 如果返回的是 StableDiffusionXLPipelineOutput，需要从中提取图像
            images_list = output.images

        # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
        images_tensors = []
        for img in images_list:
            # 将 PIL.Image 转换为 numpy.ndarray
            img_array = np.array(img)
            # 转换 numpy.ndarray 为 torch.Tensor
            img_tensor = torch.from_numpy(img_array).float() / 255.
            # 转换图像格式为 CHW (如果需要)
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            # 添加批次维度并转换为 NHWC
            img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            images_tensors.append(img_tensor)

        if len(images_tensors) > 1:
            output_image = torch.cat(images_tensors, dim=0)
        else:
            output_image = images_tensors[0]

        return (output_image,)


NODE_CLASS_MAPPINGS = {
    "BaseModel_Loader_fromhub": BaseModelLoader_fromhub_Node,
    "BaseModel_Loader_local": BaseModelLoader_local_Node,
    "PhotoMakerAdapter_Loader_fromhub": PhotoMakerAdapterLoader_fromhub_Node,
    "PhotoMakerAdapter_Loader_local": PhotoMakerAdapterLoader_local_Node,
    "LoRALoader": LoRALoader_Node,
    "Ref_Image_Preprocessing": ImagePreprocessingNode,
    "Prompt_Styler": Prompt_Style,
    "NEW_PhotoMaker_Generation": NEWCompositeImageGenerationNode,
    # "PhotoMaker_Generation": CompositeImageGenerationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BaseModel_Loader_fromhub": "Base Model Loader from hub🤗",
    "BaseModel_Loader_local": "Base Model Loader locally",
    "PhotoMakerAdapter_Loader_fromhub": "PhotoMaker Adapter Loader from hub🤗",
    "PhotoMakerAdapter_Loader_local": "PhotoMaker Adapter Loader locally",
    "LoRALoader": "LoRALoader",
    "Ref_Image_Preprocessing": "Ref Image Preprocessing",
    "Prompt_Styler": "Prompt_Styler",
    "NEW_PhotoMaker_Generation": "NEW PhotoMaker Generation",
    # "PhotoMaker_Generation": "PhotoMaker Generation"
}
