import argparse

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mc",
        "--model_conf",
        metavar='model_conf',
        type=str,
        default='resnet50'
    )
    parser.add_argument(
        "-lr",
        "--lr_conf",
        metavar='lr_conf',
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        metavar='batch_size',
        type=int,
        default=32
    )
    parser.add_argument(
        "-opt",
        "--optimizer",
        metavar='optimizer',
        type=str,
        default='Adam'
    )
    parser.add_argument(
        "-l",
        "--loss_conf",
        metavar='loss_conf',
        type=str,
        default='CrossEntropyLoss'
    )
    # add int argument for epochs
    parser.add_argument(
        "-e",
        "--epochs",
        metavar='epochs',
        type=int,
        default=13
    )
    # add float argument for sample
    parser.add_argument(
        "-s",
        "--sample",
        metavar='sample',
        type=float,
        default=1
    )
    parser.add_argument(
        "-f",
        "--folder",
        metavar='folder_config',
        type=str,
        default='0'
    )
    parser.add_argument(
        "-an",
        "--artists_number",
        metavar='artists_number',
        type=int,
        default='0'
    )
    return parser