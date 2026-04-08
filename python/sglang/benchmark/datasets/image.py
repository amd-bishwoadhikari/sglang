import io
import warnings
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pybase64
from PIL import Image
from transformers import AutoProcessor

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_mm_prompt,
)
from sglang.benchmark.utils import get_processor


@dataclass
class ImageDataset(BaseDataset):
    num_requests: int
    image_count: int
    input_len: int
    output_len: int
    range_ratio: float
    image_content: str
    image_format: str
    image_resolution: str
    backend: str
    random_image_count: bool

    @classmethod
    def from_args(cls, args: Namespace) -> "ImageDataset":
        return cls(
            num_requests=args.num_prompts,
            image_count=args.image_count,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            image_content=args.image_content,
            image_format=args.image_format,
            image_resolution=args.image_resolution,
            backend=args.backend,
            random_image_count=args.random_image_count,
        )

    def load(self, tokenizer=None, model_id=None) -> List[DatasetRow]:
        processor = get_processor(model_id)
        return sample_image_requests(
            num_requests=self.num_requests,
            image_count=self.image_count,
            input_len=self.input_len,
            output_len=self.output_len,
            range_ratio=self.range_ratio,
            processor=processor,
            image_content=self.image_content,
            image_format=self.image_format,
            image_resolution=self.image_resolution,
            backend=self.backend,
            random_image_count=self.random_image_count,
        )


def parse_image_resolution(image_resolution: str) -> Tuple[int, int]:
    """Parse image resolution into (width, height).

    Supports presets '1080p', '720p', '360p' and custom 'heightxwidth' format
    (e.g., '1080x1920' means height=1080, width=1920).
    """
    resolution_to_size = {
        "4k": (3840, 2160),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "360p": (640, 360),
    }
    if image_resolution in resolution_to_size:
        return resolution_to_size[image_resolution]

    res = image_resolution.strip().lower()
    if "x" in res:
        parts = res.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            height = int(parts[0])
            width = int(parts[1])
            if height > 0 and width > 0:
                return (width, height)

    raise ValueError(
        f"Unsupported image resolution: {image_resolution}. "
        "Choose from 4k, 1080p, 720p, 360p, or provide custom 'heightxwidth' (e.g., 1080x1920)."
    )


_SUPPORTED_BACKENDS = ("sglang", "sglang-native", "sglang-oai-chat")


def resolve_prompt_mode(backend: str) -> bool:
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Image dataset only supports backends: {list(_SUPPORTED_BACKENDS)}, "
            f"got '{backend}'."
        )
    # sglang-oai-chat: server's chat handler applies chat template, so send raw text.
    # sglang/sglang-native: /generate does not apply chat template, so send prompt_str
    # which contains image placeholder tokens needed by the multimodal processor.
    return backend == "sglang-oai-chat"


def _print_image_stats(
    dataset,
    total_images,
    image_counts,
    random_image_count,
    image_count,
    image_content,
    image_format,
    total_image_bytes,
):
    """Print aggregated image benchmark dataset token statistics.

    Metrics are read from either DatasetRow fields or from
    ``extra_request_body["image_metrics"]`` for backward compatibility with
    slimmer DatasetRow schemas.
    """

    def _get_metric(row: DatasetRow, attr: str) -> int:
        """Fetch one metric from row fields or image_metrics payload."""
        val = getattr(row, attr, None)
        if val is not None:
            return int(val)
        extra = getattr(row, "extra_request_body", None) or {}
        return int(extra.get("image_metrics", {}).get(attr, 0))

    _ROWS = [
        ("Raw text prompt tokens (w/o overhead)", "input_len"),
        ("Text prompt tokens (w overhead)", "text_prompt_len"),
        ("Text prompt overhead", "text_prompt_overhead"),
        ("Raw vision prompt tokens (w/o overhead)", "raw_vision_prompt_len"),
        ("Vision prompt tokens (w overhead)", "vision_prompt_len"),
        ("Vision overhead", "vision_prompt_overhead"),
        ("Total input tokens", "prompt_len"),
        ("Total output tokens", "output_len"),
    ]
    fmt = []
    for lb, attr in _ROWS:
        a = np.array([_get_metric(r, attr) for r in dataset])
        fmt.append(
            (
                lb,
                f"{int(a.sum()):,}",
                f"{a.mean():,.1f}",
                f"{int(a.min()):,}",
                f"{int(a.max()):,}",
            )
        )
    w = [max(len(r[i]) for r in fmt) for i in range(5)]

    print("\n===== Image Dataset Statistics =====")
    print(f"  Number of requests: {len(dataset)}")
    print(f"  Total images:       {total_images}")
    if random_image_count:
        print(
            f"  Images per request: min={np.min(image_counts)}, "
            f"max={np.max(image_counts)}, mean={np.mean(image_counts):.2f}"
        )
    else:
        print(f"  Images per request: {image_count} (fixed)")
    print()
    for lb, s, m, mn, mx in fmt:
        print(
            f"  {lb:<{w[0]}s}  sum={s:>{w[1]}}  mean={m:>{w[2]}}"
            f"  min={mn:>{w[3]}}  max={mx:>{w[4]}}"
        )
    print(
        f"\n  Image payload: {image_content} {image_format}, "
        f"avg {total_image_bytes // max(len(dataset), 1):,} bytes/request"
    )
    print("====================================\n")


def create_mm_data_row(
    text_prompt: str,
    images: List[Image.Image],
    images_base64: List[str],
    output_len: int,
    processor: AutoProcessor,
    backend: str,
    input_len: int = None,
) -> DatasetRow:
    """Create a multimodal data row for an image dataset."""

    use_raw_prompt = resolve_prompt_mode(backend)
    try:
        if type(processor).__name__ == "Phi4MMProcessor":
            # <|endoftext10|> is the image token used in the phi-4-multimodal model.
            content_items = text_prompt.replace("image 1", "|endoftext10|")
        else:
            content_items = [
                {"type": "image", "image": {"url": image_base64}}
                for image_base64 in images_base64
            ]
            content_items.append({"type": "text", "text": text_prompt})
        prompt_str = processor.apply_chat_template(
            [{"role": "user", "content": content_items}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        # Note (Xinyuan): This is a workaround for an issue where some tokenizers do not support content as a list. (e.g. InternVL)
        print(f"Error applying chat template: {e}, fallback to <image> tag")
        # Some tokenizers do not support list content; fall back to a placeholder in the text
        prompt_str = f"<image>{text_prompt}"

    # Total sequence length (text + vision + templates + overheads)
    processed = processor(
        text=[prompt_str],
        images=images,
        padding=False,
        return_tensors="pt",
    )
    input_ids = processed["input_ids"][0]
    prompt_len = input_ids.numel()

    # Text tokens after chat template (no images)
    try:
        text_only_str = processor.apply_chat_template(
            [{"role": "user", "content": text_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        text_prompt_len = processor(
            text=[text_only_str],
            padding=False,
            return_tensors="pt",
        )["input_ids"].numel()
    except Exception:
        # Fallback: just tokenize the text prompt directly
        tokenizer_to_use = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        text_prompt_len = len(tokenizer_to_use.encode(text_prompt))

    # Raw vision tokens (image pad tokens only)
    if hasattr(processor, "image_token_id") and processor.image_token_id is not None:
        raw_vision_prompt_len = (input_ids == processor.image_token_id).sum().item()
    else:
        raw_vision_prompt_len = 0
    vision_prompt_len = prompt_len - text_prompt_len

    # If input_len is not provided, use text_prompt_len as input_len
    if input_len is None:
        input_len = text_prompt_len
    # Calculate metrics for the image dataset
    image_metrics = {
        "input_len": input_len,
        "raw_vision_prompt_len": raw_vision_prompt_len,
        "text_prompt_overhead": max(prompt_len - input_len - vision_prompt_len, 0),
        "vision_prompt_overhead": max(vision_prompt_len - raw_vision_prompt_len, 0),
    }

    return DatasetRow(
        prompt=text_prompt if use_raw_prompt else prompt_str,
        prompt_len=prompt_len,
        output_len=output_len,
        text_prompt_len=text_prompt_len,
        vision_prompt_len=vision_prompt_len,
        image_data=images_base64,
        extra_request_body={"image_metrics": image_metrics},
    )


def sample_image_requests(
    num_requests: int,
    image_count: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    processor: AutoProcessor,
    image_content: str,
    image_format: str,
    image_resolution: str,
    backend: str,
    random_image_count: bool = False,
) -> List[DatasetRow]:
    """Generate requests with images.

    - If ``random_image_count`` is True, each request includes a random number of images between 1 and ``image_count``.
    - If ``random_image_count`` is False, each request includes exactly ``image_count`` images.
    - Supported resolutions: 4k (3840x2160), 1080p (1920x1080), 720p (1280x720), 360p (640x360),
      or custom 'heightxwidth' (e.g., 1080x1920).
    - Text lengths follow the 'random' dataset sampling rule. ``prompt_len``
      is the full multimodal sequence length (text + vision + templates + overheads).
    """

    # Parse resolution (supports presets and 'heightxwidth')
    width, height = parse_image_resolution(image_resolution)

    # Determine image counts for each request
    if random_image_count:
        # Random number of images per request
        image_counts = np.random.randint(1, image_count + 1, size=num_requests)
        total_images = np.sum(image_counts)
    else:
        # Fixed number of images per request
        image_counts = np.full(num_requests, image_count)
        total_images = image_count * num_requests

    # Check for potentially problematic combinations and warn user
    if width * height >= 1920 * 1080 and total_images >= 100:
        warnings.warn(
            f"High resolution ({width}x{height}) with {total_images} total images "
            f"may take a long time. Consider reducing resolution or image count.",
            UserWarning,
            stacklevel=2,
        )

    # Sample text lengths
    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_requests,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_requests,
    )

    def _gen_random_image_data_uri(
        width: int = width, height: int = height
    ) -> Tuple[Image.Image, str, int]:
        if image_content == "blank":
            # Generate blank white image
            arr = np.full((height, width, 3), 255, dtype=np.uint8)
        else:
            # Generate random colored image
            arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format=image_format, quality=85)
        encoded = pybase64.b64encode(buf.getvalue()).decode("utf-8")
        image_data = f"data:image/{image_format};base64,{encoded}"
        image_bytes = len(image_data.encode("utf-8"))
        return img, image_data, image_bytes

    dataset: List[DatasetRow] = []
    total_image_bytes = 0
    for i in range(num_requests):
        # Get the number of images for this request
        request_image_count = int(image_counts[i])

        # Generate text prompt
        target_len = int(input_lens[i])
        text_prompt = gen_mm_prompt(
            processor.tokenizer,
            getattr(processor, "image_token_id", None),
            target_len,
        )

        # Generate image list
        images, images_base64, images_bytes = zip(
            *[_gen_random_image_data_uri() for _ in range(request_image_count)]
        )
        total_image_bytes += sum(images_bytes)

        data_row = create_mm_data_row(
            text_prompt,
            list(images),
            list(images_base64),
            int(output_lens[i]),
            processor,
            backend,
            input_len=target_len,
        )
        dataset.append(data_row)

    # Print statistics
    _print_image_stats(
        dataset,
        total_images,
        image_counts,
        random_image_count,
        image_count,
        image_content,
        image_format,
        total_image_bytes,
    )
    return dataset
