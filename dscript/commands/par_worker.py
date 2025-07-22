from __future__ import annotations

import sys

import torch
from loguru import logger

from ..models.interaction import DSCRIPTModel
from ..utils import log


# Worker process function for parallel prediction
def _predict(
    device,
    modelPath,
    input_queue,
    output_queue,
    store_cmaps=False,
    use_fs=False,
    block_queue=None,
):
    device = torch.device(f"cuda:{device}" if device != "cpu" else "cpu")
    if device.type == "cpu":
        log(
            "Using CPU for predictions. This may be slow for large datasets.",
            file=None,  # If None, will be printed
            print_also=True,
        )
        use_cuda = False
    else:
        log(
            f"Using CUDA device {device.index} - {torch.cuda.get_device_name(device)}",
            file=None,  # If None, will be printed
            print_also=True,
        )
        use_cuda = True
    # Load Model
    try:
        if modelPath.endswith(".sav") or modelPath.endswith(".pt"):
            model = torch.load(
                modelPath, map_location=torch.device(device), weights_only=False
            )  # Check moved to main
            model.use_cuda = use_cuda
        else:
            logger.debug(f"Loading model from {modelPath} on device {device}.")
            # Safe to call concurrently - see https://github.com/huggingface/huggingface_hub/pull/2534
            # Prefer to download here (will only download once) for concurrency
            model = DSCRIPTModel.from_pretrained(modelPath, use_cuda=True)
            model = model.to(device=device)
            model.use_cuda = use_cuda
    except Exception as e:
        log(f"Model {modelPath} failed: {e}", file=None, print_also=True)
        sys.exit(7)

    if (
        dict(model.named_parameters())["contact.hidden.conv.weight"].shape[1] == 242
    ) and (use_fs):
        raise ValueError(
            "A TT3D model has been provided, but no foldseek_fasta has been provided"
        )
        sys.exit(8)

    model.eval()
    old_i0 = -1
    log("Making Predictions...", file=None, print_also=True)

    with torch.no_grad():
        for tup in iter(input_queue.get, None):
            # Record that all pairs in a pair of blocks have been taken off the queue,
            # as indicated by the presence of a flag of the form (None, i)
            if tup[0] is None:
                # If we still get flags, even if there is no block_queue in use, we ignore them
                # This shouldn't happen anymore.
                if block_queue is not None:
                    block_queue.put(tup[1])
                continue
            i0 = tup[0]
            i1 = tup[1]
            # Check for repeat seq - Assumes inputs may be sorted by first pair element
            if old_i0 != i0:
                p0 = tup[2].to(device=device)
                old_i0 = i0
            p1 = tup[3].to(device=device)

            # Load foldseek one-hot
            if use_fs:
                fs0 = tup[4].to(device=device)
                fs1 = tup[5].to(device=device)

            # Clear tup to remove references to tensors in shared CPU
            tup = None

            try:
                if use_fs:
                    try:
                        cm, p = model.map_predict(p0, p1, True, fs0, fs1)
                    except TypeError as e:
                        log(e)
                        log(
                            "Loaded model does not support foldseek. Please retrain with --allow_foldseek or download a pre-trained TT3D model.",
                            file=None,
                            print_also=True,
                        )
                        raise e
                else:
                    cm, p = model.map_predict(p0, p1)

                p = p.item()
                if store_cmaps:
                    cm = cm.squeeze().cpu()
                    cm.share_memory_()
                    res = (i0, i1, p, cm)
                else:
                    res = (i0, i1, p)
                output_queue.put(res)
            except RuntimeError as e:
                # Don't have seq names to print here
                log(e, file=None, printAlso=True)
                # An error arising in any process will be indicated by the presense of -1 in the output.
                # (We always have to put something  so the writer process will finish)
                output_queue.put(
                    i0, i1, -1
                )  # the contact map will only be queried if p=-1 > threshold
