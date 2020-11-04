import tensorflow as tf
from models.io import load_h5, upsample_wav
from models.model import default_opt
import models
import numpy as np
import argparse
import matplotlib
import os

os.sys.path.append(os.path.abspath("."))
os.sys.path.append(os.path.dirname(os.path.abspath(".")))


matplotlib.use("Agg")


# ----------------------------------------------------------------------------


def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="Commands")

    # train

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)

    train_parser.add_argument(
        "--train", required=True, help="path to h5 archive of training patches"
    )
    train_parser.add_argument(
        "--val", required=True, help="path to h5 archive of validation set patches"
    )
    train_parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="number of epochs to train"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=128, help="training batch size"
    )
    train_parser.add_argument(
        "--logname", default="tmp-run", help="folder where logs will be stored"
    )
    train_parser.add_argument(
        "--layers",
        default=4,
        type=int,
        help="number of layers in each of the D and U halves of the network",
    )
    train_parser.add_argument("--alg", default="adam",
                              help="optimization algorithm")
    train_parser.add_argument("--lr", default=1e-3,
                              type=float, help="learning rate")

    # eval

    eval_parser = subparsers.add_parser("eval")
    eval_parser.set_defaults(func=eval)

    eval_parser.add_argument(
        "--logname", required=True, help="path to training checkpoint"
    )
    eval_parser.add_argument(
        "--out-label", default="", help="append label to output samples"
    )
    eval_parser.add_argument(
        "--wav-file-list", help="list of audio files for evaluation"
    )
    eval_parser.add_argument("--r", help="upscaling factor", type=int)
    eval_parser.add_argument(
        "--sr", help="high-res sampling rate", type=int, default=16000
    )

    return parser


# ----------------------------------------------------------------------------


def train(args):
    # get data
    X_train, Y_train, Z_train = load_h5(args.train)
    X_val, Y_val, Z_val = load_h5(args.val)
    #   X_train, Y_train = load_h5(args.train)
    #   X_val, Y_val = load_h5(args.val)

    print("++++++++++++++++++++++")
    print(Z_train.shape)
    print(Z_train[0])
    print(Z_train[3])
    print("++++++++++++++++++++++")

    # determine super-resolution level
    n_dim, n_chan = Y_train[0].shape
    r = Y_train[0].shape[1] / X_train[0].shape[1]
    assert n_chan == 1

    # create model
    model = get_model(args, n_dim, r, from_ckpt=False, train=True)

    # load model
    # model = get_model(args, n_dim, r, from_ckpt=True, train=True)
    # model.load(args.logname) # from default checkpoint

    # train model
    model.fit(X_train, Y_train, Z_train, X_val,
              Y_val, Z_val, n_epoch=args.epochs)


#   model.fit(X_train, Y_train, X_val, Y_val, n_epoch=args.epochs)


def eval(args):
    print("------eval---------")
    # load model
    model = get_model(args, 0, args.r, from_ckpt=True, train=False)
    model.load(args.logname)  # from default checkpoint
    upsample_wav(args.wav_file_list, args, model)


def get_model(args, n_dim, r, from_ckpt=False, train=True):
    """Create a model based on arguments"""
    if train:
        opt_params = {
            "alg": args.alg,
            "lr": args.lr,
            "b1": 0.9,
            "b2": 0.999,
            "batch_size": args.batch_size,
            "layers": args.layers,
        }
    else:
        opt_params = default_opt

    # create model
    model = models.AudioUNet(
        from_ckpt=from_ckpt,
        n_dim=n_dim,
        r=r,
        opt_params=opt_params,
        log_prefix=args.logname,
    )
    return model


def main():
    print("----main start----")
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)
    print("----main end----")


if __name__ == "__main__":
    main()
