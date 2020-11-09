"""
D-SCRIPT: Structure Aware PPI Prediction
"""

def main():
    import argparse, os
    parser = argparse.ArgumentParser(description=__doc__)

    import dscript
    parser.add_argument("--version", action="version", version="dscript "+dscript.__version__)

    subparsers = parser.add_subparsers(title="How to use D-SCRIPT", dest="cmd")
    subparsers.required = True

    import dscript.commands.train
    import dscript.commands.eval
    #import dscript.commands.embed
    import dscript.commands.predict

    modules = {
            "train": dscript.commands.train,
            "eval": dscript.commands.eval,
            #"embed": dscript.commands.embed,
            "predict": dscript.commands.predict}

    for name, module in modules.items():
        sp = subparsers.add_parser(name, description=module.__doc__)
        module.add_args(sp)
        sp.set_defaults(func=module.main)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
