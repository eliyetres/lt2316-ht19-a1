import argparse

parser = argparse.ArgumentParser(description="Tests the model.")

parser.add_argument("-m", "--model", metavar="m", dest="model", type=str, help="The network model.")

args = parser.parse_args()