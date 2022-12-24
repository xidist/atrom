import os
import socket
import pprint
import json
import inspect
import config.config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_config_file", action="store_true")
    parser.add_argument("--hacky_write_data_files", action="store_true")
    args = parser.parse_args()

    if args.make_config_file:
        config.config.make_config_file()
        exit()

    if args.hacky_write_data_files:
        config.config.hacky_write_data_files()
        exit()

    

    

