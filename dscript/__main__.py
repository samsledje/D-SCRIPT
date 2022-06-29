"""
D-SCRIPT: Structure Aware PPI Prediction
"""
import argparse
import os
import sys
from typing import Union

from .commands.embed import EmbeddingArguments
from .commands.evaluate import EvaluateArguments
from .commands.predict import PredictionArguments
from .commands.train import TrainArguments

DScriptArguments = Union[
    EmbeddingArguments, EvaluateArguments, PredictionArguments, TrainArguments
]


class CitationAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(CitationAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from . import __citation__

        print(__citation__)
        setattr(namespace, self.dest, values)
        sys.exit(0)


def main():
    from . import __version__

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

    from .commands import train, embed, evaluate, predict

    modules = {
        "train": train,
        "embed": embed,
        "evaluate": evaluate,
        "predict": predict,
    }

    for name, module in modules.items():
        sp = subparsers.add_parser(name, description=module.__doc__)
        module.add_args(sp)
        sp.set_defaults(func=module.main)

    args: DScriptArguments = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
