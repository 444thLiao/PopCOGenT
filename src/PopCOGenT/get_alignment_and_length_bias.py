import argparse
from os import system, path, remove, makedirs
from os.path import join,exists
import random
import string
from Bio import SeqIO
from length_bias_functions import *
from joblib import Parallel, delayed
import glob
from itertools import combinations
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description=('Align contigs in a job array'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


    parser.add_argument('--genome_dir',
                        default=None,
                        type=str,
                        help='Directory containing genome files.')
    parser.add_argument('--genome_ext',
                        default=None,
                        type=str,
                        help='Extension for genome files (e.g., .fasta')
    parser.add_argument('--alignment_dir',
                        default=None,
                        type=str,
                        help='Directory for alignments. Please provide absolute path.')
    parser.add_argument('--mugsy_path',
                        default=None,
                        type=str,
                        help='Path to mugsy. Please provide absolute path.')
    parser.add_argument('--mugsy_env',
                        default=None,
                        type=str,
                        help='Path to mugsyenv.sh. Please provide absolute path.')

    parser.add_argument('--base_name',
                        default='output',
                        type=str,
                        help='base output file name')
    parser.add_argument('--final_output_dir',
                        default='./',
                        type=str,
                        help='Directory for final output.')

    parser.add_argument('--num_threads',
                        default=1,
                        type=int,
                        help='number of threads to run in parallel for single-machine use (i.e., not slurm)')

    parser.add_argument('--keep_alignments',
                        default=False,
                        action='store_true',
                        help='Whether to discard alignment files after length bias is calculated.')
    
    parser.add_argument('--slurm',
                        default=False,
                        action='store_true')
    parser.add_argument('--script_dir',
                        default='./',
                        type=str,
                        help='Directory for run scripts. Please provide absolute path. Required for slurm scripts.')
    parser.add_argument('--source_path',
                        default=None,
                        type=str,
                        help='Path to source scripts. Please provide absolute path. Required for slurm scripts.')

    args = parser.parse_args()
    check_inputs(args)

    if args.slurm:
        make_scripts(args.genome_dir,
                     args.genome_ext,
                     args.alignment_dir,
                     args.mugsy_env,
                     args.mugsy_path,
                     args.script_dir,
                     args.source_path)
    else:
        length_bias_files = run_on_single_machine(args.num_threads,
                                                  args.genome_dir,
                                                  args.genome_ext,
                                                  args.alignment_dir,
                                                  args.mugsy_path,
                                                  args.keep_alignments)
        header = ['Strain 1',
                 'Strain 2',
                 'Initial divergence',
                 'Alignment size',
                 'Genome 1 size',
                 'Genome 2 size',
                 'Observed SSD',
                 'SSD 95 CI low',
                 'SSD 95 CI high']
        rows = [open(f).read().strip().split() for f in length_bias_files]
        df = pd.DataFrame(rows, columns=header)
        outfile_name = '{final_output_dir}/{base_name}.length_bias.txt'.format(final_output_dir=args.final_output_dir, base_name=args.base_name)
        df.to_csv(outfile_name, sep='\t', index=False)

def check_inputs(args):

    # Check that contig files exist in the directory
    contig_list = glob.glob('{contigdir}/*{extension}'.format(contigdir=args.genome_dir,
                                                              extension=args.genome_ext))
    #print(contig_list)
    if len(contig_list) == 0:
        raise FileNotFoundError('Files with contig extension not found in directory.')

    # Check for alignment directory. Makes it if it isn't there.
    if not path.exists(args.alignment_dir):
        print('Alignment output directory does not exist. Creating new directory.')
        makedirs(args.alignment_dir)

    # Check for final ouput_directory. Makes it if it isn't there.
    if not path.exists(args.alignment_dir):
        print('Final output directory does not exist. Creating new directory.')
        makedirs(args.final_output_dir)

    # Checks mugsy path
    if not path.exists(args.mugsy_path):
        raise FileNotFoundError('Invalid mugsy path.')

    # Make sure that script directory is specified if you're using slurm.
    if args.slurm:
        if not args.script_dir:
            raise ValueError('Slurm specified without directory for scripts. Terminating.')
        if not path.exists(args.script_dir):
            print('Temporary script directory does not exist. Creating new directory.')
            makedirs(args.script_dir)

def run_on_single_machine(threads,
                          genome_directory,
                          genome_extension,
                          alignment_dir,
                          mugsy_path,
                          keep_alignments):
    tqdm.write('renaming the input genomes')
    renamed_genomes = [rename_for_mugsy(g) for g in tqdm(glob.glob(join(genome_directory,'*' + genome_extension)))]
    pairs_and_seeds = [(g1, g2, random.randint(1, int(1e9))) for g1, g2 in combinations(renamed_genomes, 2)]
    maf_files = [maf_name(_[0],_[1],alignment_dir) for _ in pairs_and_seeds]
    remaining_files = [_ for _ in maf_files if not exists(_)]
    tqdm.write(f'{len(maf_files)} in total needed. {len(remaining_files)} would be runned')
    #pairs_and_seeds = [p for m,p in tqdm(zip(maf_files,pairs_and_seeds)) if not exists(m+'.length_bias.txt')]
    length_bias_files = Parallel(n_jobs=threads)(delayed(align_and_calculate_length_bias)(g1, g2, alignment_dir, mugsy_path, seed, keep_alignments) 
                                                 for g1, g2, seed in tqdm(pairs_and_seeds))
    return length_bias_files

def make_scripts(genome_directory,
                 genome_extension,
                 alignment_dir,
                 mugsy_env,
                 mugsy_path,
                 script_dir,
                 source_path):
    tqdm.write('renaming the input genomes')
    renamed_genomes = [rename_for_mugsy(g) for g in tqdm(glob.glob(join(genome_directory,'*' + genome_extension)))]
    pairs_and_seeds = [(g1, g2, random.randint(1, int(1e9))) for g1, g2 in combinations(renamed_genomes, 2)]
    maf_files = [maf_name(_[0],_[1],alignment_dir) for _ in pairs_and_seeds]
    tqdm.write(f'Counting the pre-calculated files.')
    pairs_and_seeds = [p 
                       for m,p in tqdm(zip(maf_files,pairs_and_seeds)) 
                       if not exists(m+'.length_bias.txt')]
    tqdm.write(f'{len(pairs_and_seeds)} were lefted')
    with open('./cmds','w') as f1:
        for g1, g2, seed in tqdm(pairs_and_seeds):
            cmd = f'source {mugsy_env}'.format(mugsy_env=mugsy_env) +f'; /home-user/thliao/anaconda3/envs/PopCOGenT/bin/python3.6 {script_dir}/slurm_alignment_and_length_bias.py --genome1 {g1} --genome2 {g2} --alignment_dir {alignment_dir} --mugsy_path {mugsy_path} --seed {seed} --keep_alignments'.format(g1=g1, g2=g2, alignment_dir=alignment_dir, mugsy_path=mugsy_path, seed=str(seed), source_path=source_path)
            f1.write(cmd+'\n')
if __name__ == '__main__':
    main()