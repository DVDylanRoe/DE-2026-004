import argparse

def build_parser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=int)
    return parser