"""
D-SCRIPT: Structure Aware PPI Prediction
"""
import argparse, os, sys


class CitationAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(CitationAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        import dscript

        print(dscript.__citation__)
        setattr(namespace, self.dest, values)
        sys.exit(0)


def main():
    import dscript

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-v", "--version", action="version", version="D-SCRIPT " + dscript.__version__)
    parser.add_argument(
        "-c",
        "--citation",
        action=CitationAction,
        nargs=0,
        help="show program's citation and exit",
    )

    subparsers = parser.add_subparsers(title="D-SCRIPT Commands", dest="cmd")
    subparsers.required = True

    import dscript.commands.train
    import dscript.commands.eval
    import dscript.commands.embed
    import dscript.commands.predict

    modules = {
        "train": dscript.commands.train,
        "eval": dscript.commands.eval,
        "embed": dscript.commands.embed,
        "predict": dscript.commands.predict,
    }

    for name, module in modules.items():
        sp = subparsers.add_parser(name, description=module.__doc__)
        module.add_args(sp)
        sp.set_defaults(func=module.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
