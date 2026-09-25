"""Synthetic batched-transform regression test for the ImageNet preflight."""

from __future__ import annotations

from PIL import Image
from torchvision import transforms

from imagenet100_gpu_preflight import preprocess_batch


def test_preprocess_batch_handles_heterogeneous_images() -> None:
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    batch = {
        "image": [
            Image.new("RGB", (320, 240), color=(255, 0, 0)),
            Image.new("RGB", (487, 301), color=(0, 255, 0)),
        ],
        "label": [3, 7],
        "sample_id": ["a", "b"],
    }
    result = preprocess_batch(batch, transform, {3: 13, 7: 17})
    assert result["sample_id"] == ["a", "b"]
    assert result["labels"] == [13, 17]
    assert len(result["pixel_values"]) == 2
    assert all(tuple(tensor.shape) == (3, 224, 224) for tensor in result["pixel_values"])


if __name__ == "__main__":
    test_preprocess_batch_handles_heterogeneous_images()
    print("BATCHED PREPROCESS HETEROGENEOUS IMAGE TEST: PASS")
