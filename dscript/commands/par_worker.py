from __future__ import annotations
import sys
import torch
from ..utils import log
from ..models.interaction import DSCRIPTModel

#Worker process function for parallel prediction
def _predict(device, modelPath, input_queue, output_queue, store_cmaps=False, use_fs=False, logFile=None, block_queue=None):
    log(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}",
            file=logFile, #If None, will be printed
            print_also=True,
        )
    # Load Model
    log(f"Loading model from {modelPath}", file=logFile, print_also=True)
    if modelPath.endswith(".sav") or modelPath.endswith(".pt"):
        try:
            model = torch.load(modelPath).cuda(device=device)
            model.use_cuda = True
        except FileNotFoundError:
            log(f"Model {modelPath} not found", file=logFile, print_also=True)
            #logFile.close()
            sys.exit(1) #Is it bad to call this from multiple processes?
    else:
        try:
            model = DSCRIPTModel.from_pretrained(
                modelPath, use_cuda=True
            )
            model = model.cuda(device=device)
            model.use_cuda = True
        except Exception as e:
            #print(e)
            log(f"Model {modelPath} failed: {e}", file=logFile, print_also=True)
            #logFile.close()
            sys.exit(1)
    if (
        dict(model.named_parameters())["contact.hidden.conv.weight"].shape[1]
        == 242
    ) and (use_fs):
        raise ValueError(
            "A TT3D model has been provided, but no foldseek_fasta has been provided"
        )
    model.eval()

    old_i0 = -1
    log("Making Predictions...", file=logFile, print_also=True)

    with torch.no_grad():
        for tup in iter(input_queue.get, None):
            if block_queue is not None and tup[0] is None:
                block_queue.put(tup[1])
                continue
            i0 = tup[0]
            i1 = tup[1]
            #Check for repeat seq - Assumes inputs may be sorted by first pair element
            if old_i0 != i0:
                p0 = tup[2].cuda(device=device)
                old_i0 = i0
            p1 = tup[3].cuda(device=device)

            # Load foldseek one-hot
            if use_fs:
                fs0 = tup[4].cuda(device=device)
                fs1 = tup[5].cuda(device=device)
            
            #Clear tup to remove references to tensors in shared CPU
            tup = None
            
            try:
                if use_fs:
                    try:
                        cm, p = model.map_predict(
                            p0, p1, True, fs0, fs1
                        )
                    except TypeError as e:
                        log(e)
                        log(
                            "Loaded model does not support foldseek. Please retrain with --allow_foldseek or download a pre-trained TT3D model."
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
                #No longer have seq names n0, n1 and only first proc can log anyways
                #if logFile: 
                #    log(
                #        f"{n0} x {n1} skipped ({e})",
                #        file=logFile,
                #    )
                log(e)
                #An error arising in any process will stil be indicated by the presense of -1 in the output.
                output_queue.put(i0, i1, -1) #the contact map will only be queried if p=-1 > threshold

