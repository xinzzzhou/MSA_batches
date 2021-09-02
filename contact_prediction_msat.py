'''
Unsupervised Contact Prediction (MSA Transformer)
Changed Author: Xin Zhou
Date:           18/5/2021
Input:          'xx.pt': Pretrained MSA Transformer model
                'xx.aln','xx.a3m': MSA
                'batch_size': Batch size of the sequences
                'seq_number': number of sequences would be sampled in a MSA
                'root_path':
Output:         Contact map of the first sequence
Function:       Using the pretrained model to compute contact map for each gene with MSA.
                Divide MSAs with batch, and process batches MSAs
Attention(*):
                (1) sampled sequence: Keeping the first sequence, and random sample certain number of sequences.
                (2) result shape: if batch size >1, then the shape of saved matrix (contact map) is padded;
                                  else, the shape equals to the real.
                (3) sequences longer than 1024 cannot be computed.
'''

import matplotlib.pyplot as plt
import esm
import torch
import os
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
from random import sample
import numpy as np
import argparse
import time

def params_parser():
    parser = argparse.ArgumentParser(description="Hyper-parameters fpr unsupervised contact prediction (with MSA Transformer)")
    parser.add_argument(
        "--batch_size", default=3, type=int, help="Batch size of the sequences,considering the memory limitation, usually 3 or 4")
    parser.add_argument(
        "--seq_number", default=64, type=int, help="number of sequences would be sampled in a MSA")
    parser.add_argument(
        "--root_path", default="/mnt/a22896c0-1177-4b70-9625-fdbdf3e9e159/PDB/", type=str, help="root path")
    parser.add_argument(
        "--result_path", default="/mnt/a22896c0-1177-4b70-9625-fdbdf3e9e159/PDB/msaT_result/test/", type=str, help="result path")
    parser.add_argument(
        "--long_files", default="['5B2PA','1OFDA','4LGYA','5B2RB']", type=str, help="long files (seq_len > 1024)")
    return parser.parse_args()

torch.set_grad_enabled(False)

## Data loading
# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)
#long files (seq_len > 1024)
# long_files = ['5B2PA','1OFDA','4LGYA','5B2RB']

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def read_msa1(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    tmp=[]
    a = SeqIO.parse(filename, "fasta")
    c= itertools.islice(a, nseq)
    for record in c:
        b = remove_insertions(str(record.seq))
        tmp.append((record,b))
    return tmp

def read_msa2(filename: str, nseq: int) -> List[Tuple[str, str]]:
    '''Reads random nseq sequences from an MSA file (always include the original one), automatically removes insertinos.'''
    return [(list(SeqIO.parse(filename, "fasta"))[0].description, remove_insertions(str(list(SeqIO.parse(filename, "fasta"))[0].seq)))] + [(record.description, remove_insertions(str(record.seq)))
                           for record in sample(list(SeqIO.parse(filename, "fasta")), nseq - 1)]

'''get computed files'''
def get_computed_files(result_path):
    computed_files = []
    for _, _, fs in os.walk(result_path):
        for f in fs:
            computed_files.append(f)
    return computed_files

def get_tail(s):
    return s.split(".")[-1]

'''process file with head'''
def get_depthAlen(file):
    lines = file.readlines()
    seq_number = int(len(lines)/2)
    seq_len = int(len(lines[1]))
    file.close()
    return seq_number, seq_len

def get_path(root_path, name, type,  extend_name = ""):
    if extend_name == "":
        if type == "aln":
            return f"{root_path}data1/{name[1:3]}/{name}.aln"
        else:
                return f"{root_path}data/{name[1:3]}/{name}.{type}"
    else:
        if "msaT" not in extend_name:
            return f"{root_path}data/{name[1:3]}/{name}_{extend_name}.{type}"
        else:
            return f"{root_path}data1/{name[1:3]}/{name}_{extend_name}.{type}"

'''cut the padding results'''
def cutoffpadding(matrix, length):
    a= matrix[:length-1,:length-1]
    return a

'''get all files and divide them according to the seq length'''
def get_all_files(batch_size, long_files, root_path, result_path, seq_number):
    normal_ii = 0
    low_depth_msas = []  # record all files(depth < seq_number)
    normal_depth_msas = []  # record all files (depth >= seq_number)
    normal_tmp = []    # record batch files
    msas_depth = dict()
    msas_length = dict()
    dir_path = f"{root_path}data1/"
    computed_files = get_computed_files(result_path)
    for dir in os.listdir(dir_path):    # iterate all the directoryies
        for _, _, fs in os.walk(dir_path+dir):    # iterate all the files under the specified directory
            for f in fs:
                gene_name = f[-9:-4]
                if (get_tail(f)=='aln') and (gene_name not in computed_files) and (gene_name not in long_files):
                    full_path = os.path.join(dir_path+dir, f)
                    depth, length = get_depthAlen(open(full_path,'r'))
                    msas_depth[gene_name] = depth
                    msas_length[gene_name] = length
                    if depth < seq_number:
                        low_depth_msas.append(gene_name)
                    else:
                        normal_ii += 1
                        if (normal_ii % batch_size) != 0:
                            normal_tmp.append(gene_name)
                        else:
                            normal_tmp.append(gene_name)
                            normal_depth_msas.append(normal_tmp)
                            normal_tmp = []
    if len(normal_tmp) != 0:
        normal_depth_msas.append(normal_tmp)
    return low_depth_msas, normal_depth_msas, msas_depth, msas_length

if __name__ == '__main__':
    args = params_parser()
    # Run MSA Transformer Contact Prediction
    msa_transformer, msa_alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval().cuda(0)
    msa_batch_converter = msa_alphabet.get_batch_converter()
    # get all computed files
    low_depth_msas, normal_depth_msas, msas_depth, msas_length = get_all_files(args.batch_size, args.long_files, args.root_path, args.result_path, args.seq_number)

    '''process low files'''
    start_time = time.time()
    for low_msa in low_depth_msas:
        msa_data = [read_msa2(get_path(args.root_path, low_msa, 'aln'), msas_depth.get(low_msa))]
        msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
        msa_batch_tokens = msa_batch_tokens.cuda()
        print(msa_batch_tokens.size(), msa_batch_tokens.dtype)  # Should be a 3D tensor with dtype torch.int64.

        msa_contacts = msa_transformer.predict_contacts(msa_batch_tokens).cpu()
        msa_contacts.numpy()
        np.savetxt(get_path(args.root_path, low_msa, 'mtx', 'msaT'), msa_contacts[0], fmt="%f", delimiter="\t")

        # fig, axes = plt.subplots(figsize=(18, 6), ncols=3)
        # for ax, contact, msa in zip(axes, msa_contacts, msa_batch_strs):
        #     seqlen = len(msa[0])
        #     ax.imshow(contact[:seqlen, :seqlen], cmap="Blues")
        # plt.savefig("a.png")
        # plt.show()
    end_time = time.time()
    print("average running time: %.2f" % ((end_time - start_time) / len(low_depth_msas)))

    '''process normal files'''
    start_time = time.time()
    for batches_msas in normal_depth_msas:
        msa_data = []
        for batch_msa in batches_msas:
            msa_data.append(read_msa2(get_path(args.root_path, batch_msa, 'aln'), args.seq_number))
        msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
        msa_batch_tokens = msa_batch_tokens.cuda()
        print(msa_batch_tokens.size(), msa_batch_tokens.dtype)  # Should be a 3D tensor with dtype torch.int64.

        msa_contacts = msa_transformer.predict_contacts(msa_batch_tokens).cpu()
        msa_contacts.numpy()
        i=0
        for map in msa_contacts:
            np.savetxt(get_path(args.root_path, batches_msas[i], 'mtx', 'msaT'), cutoffpadding(map,msas_length.get(batches_msas[i])), fmt="%f", delimiter="\t")
            i += 1

            # fig, axes = plt.subplots(figsize=(18, 6), ncols=3)
            # for ax, contact, msa in zip(axes, msa_contacts, msa_batch_strs):
            #     seqlen = len(msa[0])
            #     ax.imshow(contact[:seqlen, :seqlen], cmap="Blues")
            # plt.savefig("a.png")
            # plt.show()
    end_time = time.time()
    print("average running time: %.2f" % ((end_time - start_time) / len(normal_depth_msas)))

