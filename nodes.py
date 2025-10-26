import base64
import io
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os
import torch
import hashlib
import folder_paths
from PIL.PngImagePlugin import PngInfo
import json
import cv2

class LoadBase64Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base64_string": ("STRING", {"multiline": True, "default": ""})}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, base64_string):
        # Skip execution if string is empty or None
        if not base64_string or base64_string.strip() == "":
            print("[LoadBase64Image] Skipped — empty base64 string")
            return (None, None)

        try:
            # Clean up header if needed
            if base64_string.startswith("data:image/") or base64_string.startswith(
                "data:application/octet-stream;base64,"
            ):
                base64_string = base64_string.split(";")[1].split(",")[1]

            # Try decoding base64
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
                image_rgb = i.convert("RGB")

                if len(output_images) == 0:
                    w, h = image_rgb.size

                if image_rgb.size != (w, h):
                    continue

                image_tensor = torch.from_numpy(
                    np.array(image_rgb).astype(np.float32) / 255.0
                )[None,]

                if 'A' in i.getbands():
                    mask = 1.0 - torch.from_numpy(
                        np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    )
                else:
                    mask = torch.zeros((64, 64), dtype=torch.float32)

                output_images.append(image_tensor)
                output_masks.append(mask.unsqueeze(0))

            if not output_images:
                print("[LoadBase64Image] Skipped — no valid image frames")
                return (None, None)

            if len(output_images) > 1 and image.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]

            return (output_image, output_mask)

        except Exception as e:
            print(f"[LoadBase64Image] Skipped — invalid or corrupt base64 ({e})")
            return (None, None)

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

# MASKS SUBTRACT

class WAS_Mask_Subtract:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks_a": ("MASK",),
                        "masks_b": ("MASK",),
                    }
                }

    CATEGORY = "WAS Suite/Image/Masking"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "subtract_masks"

    def subtract_masks(self, masks_a, masks_b):
        subtracted_masks = torch.clamp(masks_a - masks_b, 0, 255)
        return (subtracted_masks,)

# MASKS ADD

class WAS_Mask_Add:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks_a": ("MASK",),
                        "masks_b": ("MASK",),
                    }
                }

    CATEGORY = "WAS Suite/Image/Masking"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "add_masks"

    def add_masks(self, masks_a, masks_b):
        if masks_a.ndim > 2 and masks_b.ndim > 2:
            added_masks = masks_a + masks_b
        else:
            added_masks = torch.clamp(masks_a.unsqueeze(1) + masks_b.unsqueeze(1), 0, 255)
            added_masks = added_masks.squeeze(1)
        return (added_masks,)

## IMAGE SAVE METADATA
class saveImageLazy():
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")  # Include LIST for full file paths
    RETURN_NAMES = ("images", "filenames")  # Match the expected input for the next step
    OUTPUT_NODE = False
    FUNCTION = "save"
    CATEGORY = "EasyUse/Image"

    def save(self, images, save_metadata, prompt=None, extra_pnginfo=None):
        extension = 'png'

        # Derive filename_prefix internally
        default_prefix = "AnimateDiff"
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            default_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        complete_file_paths = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            filename_with_batch_num = filename.replace(
                "%batch_num%", str(batch_number)
            )

            counter = 1

            if os.path.exists(full_output_folder) and os.listdir(full_output_folder):
                filtered_filenames = list(filter(
                    lambda filename: filename.startswith(
                        filename_with_batch_num + "_"
                    ) and filename[len(filename_with_batch_num) + 1:-4].isdigit(),
                    os.listdir(full_output_folder)
                ))

                if filtered_filenames:
                    max_counter = max(
                        int(filename[len(filename_with_batch_num) + 1:-4])
                        for filename in filtered_filenames
                    )
                    counter = max_counter + 1

            file = f"{filename_with_batch_num}_{counter:05}.{extension}"

            save_path = os.path.join(full_output_folder, file)  # Full file path

            if save_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            img.save(save_path, pnginfo=metadata)

            complete_file_paths.append(save_path)  # Add full path to the list

        # Return images and the list of full file paths
        return (images, complete_file_paths)

# gaussian blur a tensor image batch in format [B x H x W x C] on H/W (spatial, per-image, per-channel)
def cv_blur_tensor(images, dx, dy):
    if min(dx, dy) > 100:
        np_img = torch.nn.functional.interpolate(images.detach().clone().movedim(-1,1), scale_factor=0.1, mode='bilinear').movedim(1,-1).cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(image, (dx // 20 * 2 + 1, dy // 20 * 2 + 1), 0)
        return torch.nn.functional.interpolate(torch.from_numpy(np_img).movedim(-1,1), size=(images.shape[1], images.shape[2]), mode='bilinear').movedim(1,-1)
    else:
        np_img = images.detach().clone().cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(image, (dx, dy), 0)
        return torch.from_numpy(np_img)

# guided filter a tensor image batch in format [B x H x W x C] on H/W (spatial, per-image, per-channel)
def guided_filter_tensor(ref, images, d, s):
    if d > 100:
        np_img = torch.nn.functional.interpolate(images.detach().clone().movedim(-1,1), scale_factor=0.1, mode='bilinear').movedim(1,-1).cpu().numpy()
        np_ref = torch.nn.functional.interpolate(ref.detach().clone().movedim(-1,1), scale_factor=0.1, mode='bilinear').movedim(1,-1).cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = guidedFilter(np_ref[index], image, d // 20 * 2 + 1, s)
        return torch.nn.functional.interpolate(torch.from_numpy(np_img).movedim(-1,1), size=(images.shape[1], images.shape[2]), mode='bilinear').movedim(1,-1)
    else:
        np_img = images.detach().clone().cpu().numpy()
        np_ref = ref.cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = guidedFilter(np_ref[index], image, d, s)
        return torch.from_numpy(np_img)

class RestoreDetail:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "detail": ("IMAGE",),
                "mode": (["add", "multiply"],),
                "blur_type": (["blur", "guidedFilter"],),
                "blur_size": ("INT", {"default": 1, "min": 1, "max": 1023}),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "round": 0.01}),
            },
            "optional": {
                "mask": ("IMAGE",),  # Optional mask
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_normalize"
    CATEGORY = "image/filters"

    def batch_normalize(self, images, detail, mode, blur_type, blur_size, factor, mask=None):
        t = images.detach().clone() + 0.1
        ref = detail.detach().clone() + 0.1

        if ref.shape[0] < t.shape[0]:
            ref = ref[0].unsqueeze(0).repeat(t.shape[0], 1, 1, 1)

        d = blur_size * 2 + 1

        if blur_type == "blur":
            blurred = cv_blur_tensor(t, d, d)
            blurred_ref = cv_blur_tensor(ref, d, d)
        elif blur_type == "guidedFilter":
            blurred = guided_filter_tensor(t, t, d, 0.01)
            blurred_ref = guided_filter_tensor(ref, ref, d, 0.01)

        if mode == "multiply":
            t = (ref / blurred_ref) * blurred
        else:
            t = (ref - blurred_ref) + blurred

        t = t - 0.1
        t = torch.clamp(torch.lerp(images, t, factor), 0, 1)

        # Apply mask if provided
        if mask is not None:
            mask = mask.expand_as(t)  # Ensure mask has the same shape
            t = t * mask + images * (1 - mask)

        return (t,)

# Register the node class with ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadBase64Image": LoadBase64Image,
    "LoadBase64Mask": LoadBase64Mask,
    "ImageToBase64": ImageToBase64,
    "WAS_Mask_Add": WAS_Mask_Add,
    "WAS_Mask_Subtract": WAS_Mask_Subtract,
    "easy saveImageLazy": saveImageLazy,
    "RestoreDetail": RestoreDetail,

}

# Provide a friendly name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBase64Image": "Load Base64 Image Node",
    "LoadBase64Mask": "Load Base64 Mask Node",
    "ImageToBase64": "Image to Base64 Node",
    "WAS_Mask_Add": "mask add victor",
    "WAS_Mask_Subtract": "mask subtract victor",
    "easy saveImageLazy": "Save Image (Lazy)",
    "RestoreDetail": "Restore Detail",

}
