"""
D-SCRIPT: Structure Aware PPI Prediction
"""

import argparse
import sys

from .commands import (
    embed,
    evaluate,
    extract_3di,
    predict_bipartite,
    predict_block,
    predict_serial,
)
from .commands.embed import EmbeddingArguments
from .commands.evaluate import EvaluateArguments
from .commands.extract_3di import Extract3DiArguments
from .commands.predict_bipartite import BipartitePredictionArguments
from .commands.predict_block import BlockedPredictionArguments
from .commands.predict_serial import PredictionArguments

DScriptArguments = (
    EmbeddingArguments
    | EvaluateArguments
    | PredictionArguments
    | BlockedPredictionArguments
    | BipartitePredictionArguments
    | Extract3DiArguments
)


class CitationAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from . import __citation__

        print(__citation__)
        setattr(namespace, self.dest, values)
        sys.exit(0)


def main():
    from . import __version__

    # The new training commands parse their own settings (YAML + flags); hand
    # them everything after the command name. This is also how Fabric's
    # launched processes re-enter.
    if len(sys.argv) > 1 and sys.argv[1] == "train_serial":
        from .commands.train_serial import main as train_main

        return train_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "train_batched":
        from .commands.train_batched import main as train_main

        return train_main(sys.argv[2:])

    # Imported here, not at the top, so the new commands above never load it.
    from .legacy.tt3d_train import train  # upstream tt3d_new training, frozen

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-v", "--version", action="version", version="D-SCRIPT " + __version__
    )
    parser.add_argument(
        "-c",
        "--citation",
        action=CitationAction,
        nargs=0,
        help="show program's citation and exit",
    )

    subparsers = parser.add_subparsers(title="D-SCRIPT Commands", dest="cmd")
    subparsers.required = True

    subparsers.add_parser("train_serial", help="train the original model (see `dscript train_serial --help`)")
    subparsers.add_parser("train_batched", help="train the batched model (see `dscript train_batched --help`)")
    modules = {
        "train": train,
        "embed": embed,
        "evaluate": evaluate,
        "predict_serial": predict_serial,
        "predict": predict_block,
        "predict_bipartite": predict_bipartite,
        "extract-3di": extract_3di,
    }

    for name, module in modules.items():
        sp = subparsers.add_parser(name, description=module.__doc__)
        module.add_args(sp)
        sp.set_defaults(func=module.main)

    args: DScriptArguments = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

