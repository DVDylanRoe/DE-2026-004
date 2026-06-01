from cli import build_parser

def test_build_parser():

    parser = build_parser()
    args = parser.parse_args(["--uid", "123"])
    assert args.uid == 123