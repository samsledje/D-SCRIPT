import argparse

import torch

from dscript.models.interaction import DSCRIPTModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Push a model to HuggingFace")
    parser.add_argument("model_pt", help="Path to the model.pt file")
    parser.add_argument("hf_user", help="HuggingFace user")
    parser.add_argument("hf_model_name", help="HuggingFace model name")
    args = parser.parse_args()

    model_pt = args.model_pt
    hf_user = args.hf_user
    hf_model_name = args.hf_model_name

    # Load the state dict
    model_old = torch.load(model_pt, map_location=torch.device("cpu"))
    state_dict = model_old.state_dict()

    # Load the model
    model = DSCRIPTModel(
        emb_nin=6165,
        emb_nout=100,
        emb_dropout=0.5,
        con_embed_dim=121,
        con_hidden_dim=50,
        con_width=7,
        use_cuda=False,
        do_w=model_old.do_w,
        pool_size=9,
        do_pool=model_old.do_pool,
        do_sigmoid=model_old.do_sigmoid,
    )

    # Load the state dict into the model
    model.load_state_dict(state_dict)
    model.eval()

    model.push_to_hub(f"{hf_user}/{hf_model_name}")
