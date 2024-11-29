import base64
import io
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os
import torch
import hashlib

class LoadBase64Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base64_string": ("STRING", {"multiline": True, "default": ""})}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, base64_string):
        if base64_string.startswith("data:image/") or base64_string.startswith(
            "data:application/octet-stream;base64,"
        ):
            base64_string = base64_string.split(";")[1].split(",")[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(image):
            i = ImageOps.exif_transpose(i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and image.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(cls, base64_string):
        m = hashlib.sha256()
        m.update(base64_string.encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, base64_string):
        try:
            if base64_string.startswith("data:image/") or base64_string.startswith(
                "data:application/octet-stream;base64,"
            ):
                base64_string = base64_string.split(";")[1].split(",")[1]
            base64.b64decode(base64_string)
            return True
        except Exception as e:
            return f"Invalid base64 string: {str(e)}"

class LoadBase64Mask:
    _color_channels = ["alpha", "red", "green", "blue"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_string": ("STRING", {"multiline": True, "default": ""}),
                "channel": (cls._color_channels, ),
            }
        }

    CATEGORY = "mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"

    def load_image(self, base64_string, channel):
        if base64_string.startswith("data:image/") or base64_string.startswith(
            "data:application/octet-stream;base64,"
        ):
            base64_string = base64_string.split(";")[1].split(",")[1]
        print(base64_string)
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image = ImageOps.exif_transpose(image)

        if image.getbands() != ("R", "G", "B", "A"):
            if image.mode == 'I':
                image = image.point(lambda i: i * (1 / 255))
            image = image.convert("RGBA")

        mask = None
        c = channel[0].upper()
        if c in image.getbands():
            mask = np.array(image.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (mask.unsqueeze(0),)

    @classmethod
    def IS_CHANGED(cls, base64_string, channel):
        m = hashlib.sha256()
        m.update(base64_string.encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, base64_string):
        try:
            base64.b64decode(base64_string)
            return True
        except Exception as e:
            return f"Invalid base64 string: {str(e)}"

class ImageToBase64:
    def __init__(self):
        self.output_dir = "output_images"  # Specify your output directory here
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"})
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base64_image",)
    FUNCTION = "convert_to_base64"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def convert_to_base64(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        results = list()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            results.append(base64_image)

            metadata = None
            if prompt or extra_pnginfo:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", prompt)
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, value)

            filename_with_batch_num = f"{filename_prefix}_{batch_number:05}.png"
            file_path = os.path.join(self.output_dir, filename_with_batch_num)
            img.save(file_path, pnginfo=metadata, compress_level=self.compress_level)

        return (results,)
# Register the node class with ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadBase64Image": LoadBase64Image,
    "LoadBase64Mask": LoadBase64Mask,
    "ImageToBase64": ImageToBase64

}

# Provide a friendly name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBase64Image": "Load Base64 Image Node",
    "LoadBase64Mask": "Load Base64 Mask Node",
    "ImageToBase64": "Image to Base64 Node"
}
