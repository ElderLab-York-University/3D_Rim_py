import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="SAIRL")

    #General
    parser.add_argument("--DataPath",
                        type=str,
                        default='../Thingi10K',
                        help="Path of the dataset")

    parser.add_argument("--NpzOutPath",
                        type=str,
                        default='../Thingi10KData',
                        help="Path of the output result(in raw npz file)")

    parser.add_argument("--SampleStep",
                        type=int,
                        default=4,
                        help="number of sampling from given vertex")
    parser.add_argument("--NumOfObject",
                        type=int,
                        default=500,
                        help="number of object sampled from dataset")
    parser.add_argument("--maxNumSamplePreObj",
                        type=int,
                        default=10,
                        help="number of view angle sampled for one object")
    return parser.parse_args()