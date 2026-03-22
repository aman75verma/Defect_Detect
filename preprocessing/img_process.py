# Single image, save as PNG:
# python preprocessing/img_process.py --image "path\img.jpg" --output_dir "data\processed_images" --format img --img_ext png --crop_mode center
# python preprocessing/img_process.py --input_dir "data\raw" --output_dir "data\processed_images_npy_batch" --format npy --npy_dtype float32 --crop_mode pad
# python preprocessing/img_process.py --input_dir "data\raw" --output_dir "data\processed_images_batch" --format img --img_ext jpg --crop_mode center


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TARGET_SIZE = 224
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_phone_image(image_path: str | Path) -> Image.Image:
    """Load image, correct phone EXIF rotation, convert to RGB (3 channels)."""
    with Image.open(image_path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def center_crop_square(image: Image.Image) -> Image.Image:
    """Center-crop to square so object geometry is not stretched."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def pad_to_square(image: Image.Image, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Pad to square without removing border content.
    Useful when defects can appear near image edges.
    """
    width, height = image.size
    side = max(width, height)
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2
    pad_right = side - width - pad_left
    pad_bottom = side - height - pad_top
    return ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)


def preprocess_image(
    image_path: str | Path,
    size: int = TARGET_SIZE,
    crop_mode: str = "center",
) -> np.ndarray:
    """
    Return a preprocessed image as uint8 numpy array with shape (224, 224, 3).
    """
    image = load_phone_image(image_path)
    if crop_mode == "center":
        image = center_crop_square(image)
    elif crop_mode == "pad":
        image = pad_to_square(image)
    else:
        raise ValueError("--crop_mode must be either 'center' or 'pad'")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.shape != (size, size, 3):
        raise ValueError(f"Unexpected output shape {arr.shape}; expected {(size, size, 3)}")
    return arr


def save_processed_array(
    arr: np.ndarray,
    src_path: str | Path,
    output_dir: str | Path,
    output_format: str,
    image_ext: str = "jpg",
    npy_dtype: str = "uint8",
) -> Path:
    """
    Save preprocessed array as:
    - image format (jpg/png)
    - npy format
    """
    src_path = Path(src_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "npy":
        if npy_dtype == "float32":
            arr_to_save = (arr.astype(np.float32) / 255.0).astype(np.float32)
        elif npy_dtype == "uint8":
            arr_to_save = arr
        else:
            raise ValueError("--npy_dtype must be one of: uint8, float32")
        save_path = output_dir / f"{src_path.stem}.npy"
        np.save(save_path, arr_to_save)
        return save_path

    if output_format == "img":
        image_ext = image_ext.lower().lstrip(".")
        if image_ext not in {"jpg", "jpeg", "png"}:
            raise ValueError("For --format img, --img_ext must be one of: jpg, jpeg, png")
        suffix = ".jpg" if image_ext == "jpeg" else f".{image_ext}"
        save_path = output_dir / f"{src_path.stem}{suffix}"
        Image.fromarray(arr).save(save_path)
        return save_path

    raise ValueError("--format must be either 'img' or 'npy'")


def process_one(
    image_path: str | Path,
    output_dir: str | Path,
    output_format: str,
    image_ext: str,
    crop_mode: str,
    npy_dtype: str,
) -> Path:
    arr = preprocess_image(image_path, crop_mode=crop_mode)
    return save_processed_array(
        arr=arr,
        src_path=image_path,
        output_dir=output_dir,
        output_format=output_format,
        image_ext=image_ext,
        npy_dtype=npy_dtype,
    )


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    output_format: str,
    image_ext: str,
    crop_mode: str,
    npy_dtype: str,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    image_files = [p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in VALID_EXTS]
    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    for image_path in image_files:
        relative_parent = image_path.parent.relative_to(input_dir)
        dest_dir = output_dir / relative_parent
        save_path = process_one(
            image_path=image_path,
            output_dir=dest_dir,
            output_format=output_format,
            image_ext=image_ext,
            crop_mode=crop_mode,
            npy_dtype=npy_dtype,
        )
        print(f"Saved: {save_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess raw phone images to 224x224x3 and save as image or .npy"
    )
    parser.add_argument("--image", type=str, help="Path to one input image")
    parser.add_argument("--input_dir", type=str, help="Path to input folder of images")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Folder where processed files will be saved",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="img",
        choices=["img", "npy"],
        help="Output format: img or npy",
    )
    parser.add_argument(
        "--img_ext",
        type=str,
        default="jpg",
        help="When --format img, extension to save: jpg/jpeg/png",
    )
    parser.add_argument(
        "--crop_mode",
        type=str,
        default="center",
        choices=["center", "pad"],
        help="center: follow report pipeline, pad: preserve edge defects",
    )
    parser.add_argument(
        "--npy_dtype",
        type=str,
        default="uint8",
        choices=["uint8", "float32"],
        help="When --format npy: uint8 (0-255) or float32 (0-1)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("Pass either --image (single file) or --input_dir (folder)")

    if args.image and args.input_dir:
        parser.error("Use only one mode: --image OR --input_dir")

    if args.image:
        save_path = process_one(
            image_path=args.image,
            output_dir=args.output_dir,
            output_format=args.format,
            image_ext=args.img_ext,
            crop_mode=args.crop_mode,
            npy_dtype=args.npy_dtype,
        )
        print(f"Saved: {save_path}")
        return

    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_format=args.format,
        image_ext=args.img_ext,
        crop_mode=args.crop_mode,
        npy_dtype=args.npy_dtype,
    )


if __name__ == "__main__":
    main()
