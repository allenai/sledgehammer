#!/usr/bin/env python

import subprocess
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main():
    parser = arg_parser()

    args = parser.parse_args()
    
    serialization_dir = "/".join(args.model_file.split("/")[:-1])

    cmd = "python allennlp_overrides/commands/evaluate.py {} {} ".format(serialization_dir, args.dev_file)+\
        "--include-package allennlp_overrides --cuda-device {} ".format(args.cuda_device)+\
        " -o "+'"{'+" iterator: "+'{'+"batch_size: 1"+'}'+", model: "+'{'+\
            "temperature_threshold: 1, scaling_temperature: '{}'".format(args.temperatures)+\
            '}}"'+" --weights-file {} -t \"{}\" --output-file {};".format(
            args.model_file, args.confidence_threshold, args.output_file)

    print (cmd)
    return_value = subprocess.call(cmd, shell=True)

    if return_value != 0:
        print("Evaluation failed (return value {})".format(return_value))
        return -2

    return 0



def arg_parser():
    """Extracting CLI arguments"""
    p = ArgumentParser(add_help=False)

    p.add_argument("-t", "--temperatures", help="Calibration temperatures", type=str, required=True)
    p.add_argument("-c", "--confidence_threshold", help="Confidence threshold value", type=float, default=1)
    p.add_argument("-u", "--cuda_device", help="CUDA device (or -1 for CPU)", type=int, default=0)
    p.add_argument('-v', '--dev_file', help="Development set file",  type=str, required=True)
    p.add_argument('-m', '--model_file', help="Model file",  type=str, required=True)
    p.add_argument('-o', '--output_file', help="Output file",  type=str, required=True)


    return  ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[p])


if __name__ == '__main__':
    sys.exit(main())

