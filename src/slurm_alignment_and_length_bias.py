import sys
from os.path import dirname,realpath,join
dname = join(dirname(realpath(__file__)),'PopCOGenT')
sys.path.insert(0,dname)
from length_bias_functions import *
import argparse

def main():
    parser = argparse.ArgumentParser(
        description=('Align contigs in a job array'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--genome1',
                        default=None,
                        type=str,
                        help='Genome 1 file.')
    parser.add_argument('--genome2',
                        default=None,
                        type=str,
                        help='Genome 2 file.')
    parser.add_argument('--alignment_dir',
                        default=None,
                        type=str,
                        help='Directory for alignments.')
    parser.add_argument('--mugsy_path',
                        default=None,
                        type=str,
                        help='Path to mugsy.')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='Random seed.')

    parser.add_argument('--keep_alignments',
                        default=False,
                        action='store_true',
                        help='Whether to discard alignment files after length bias is calculated.')
    args = parser.parse_args()
    align_and_calculate_length_bias(args.genome1, args.genome2, args.alignment_dir, args.mugsy_path, args.seed,args.keep_alignments)

if __name__ == '__main__':
    main()