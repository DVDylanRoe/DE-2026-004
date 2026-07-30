import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=int)  
    parser.add_argument("--serve", action="store_true")
    return parser
