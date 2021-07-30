from io import TextIOWrapper
import argparse
from typing import TextIO
import os
import pickle
import numpy as np

from collections import defaultdict
from operator import itemgetter


class EmptySequenceError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class NonAlphabetSymbolsException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class NotSeenSymbolException(Exception):
    def __init__(self, message):
        self.__message = message
        super().__init__(self.__message)


class UnknownNucleotideException(Exception):
    def __init__(self, message):
        self.__message = message
        super().__init__(self.__message)


class Sequence:
    def __init__(self, sequence: str, accept_non_alpha: bool = False):
        self.sequence = sequence.strip().upper()
        if not self.sequence:
            raise EmptySequenceError
        if not self.isalpha() and not accept_non_alpha:
            raise NonAlphabetSymbolsException
        self.flavor_pp_dict = {"A": "R", "G": "R", "T": "Y", "C": "Y", "U": "Y"}
        self.flavor_ak_dict = {"A": "M", "G": "K", "T": "K", "C": "M", "U": "K"}
        self.flavor_sw_dict = {"A": "W", "G": "S", "T": "W", "C": "S", "U": "W"}
        self.flavor_ct_dict = {"A": "1", "G": "1", "V": "1",
                               "I": "2", "L": "2", "F": "2", "P": "2",
                               "Y": "3", "M": "3", "T": "3", "S": "3",
                               "H": "4", "N": "4", "Q": "4", "W": "4",
                               "R": "5", "K": "5",
                               "D": "6", "E": "6",
                               "C": "7"}
        self.flavor_hp_dict = {"A": "n", "G": "n", "S": "n", "T": "n", "P": "n", "H": "n", "Y": "n",  # neutral amino acids
                               "R": "p", "K": "p", "E": "p", "D": "p", "Q": "p", "N": "p",  # polar
                               "C": "h", "L": "h", "V": "h", "I": "h", "M": "h", "F": "h", "W": "h"}  # hydrophobic
        self.blosum62 = {"A": [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4, -10],
                         "R": [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4, -10],
                         "N": [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4, -10],
                         "D": [2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4,  1, -1, -4, -10],
                         "C": [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4, -10],
                         "Q": [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4, -10],
                         "E": [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4, -10],
                         "G": [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4, -10],
                         "H": [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4, -10],
                         "I": [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4, -10],
                         "L": [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4, -10],
                         "K": [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4, -10],
                         "M": [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4, -10],
                         "F": [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4, -10],
                         "P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4, -10],
                         "S": [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4, -10],
                         "T": [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -1, -1, 0, -4, -10],
                         "W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4, -10],
                         "Y": [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4, -10],
                         "V": [0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1, 4, -3, -2, -1, -4, -10],
                         "B": [-2, -1, 3, 4, -3,  0,  1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4, -10],
                         "Z": [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3,  1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4, -10],
                         "X": [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4, -10],
                         "*": [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1, -10],
                         "-": [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 1]}
        self.hivb = {
             "A": [7, -5, -4, -2, -7, -8, -2, -1, -8, -2, -6, -5, -4, -9, -1, -1, 2, -13, -12, 1, -3, -4, -2, -13, -19],
             "R": [-5, 6, -2, -7, -6, 0, -4, 0, 1, -4, -4, 3, -1, -9, -3, 0, -1, -4, -6, -7, -4, -1, -1, -13, -19],
             "N": [-4, -2, 7, 3, -6, -3, -3, -4, 0, -4, -8, 0, -7, -8, -5, 2, 0, -13, -2, -7, 6, -3, -1, -13, -19],
             "D": [-2, -7, 3, 8, -10, -6, 2, -1, -1, -8, -12, -4, -10, -11, -9, -2, -4, -12, -4, -4, 6, 0, -2, -13, -19],
             "C": [-7, -6, -6, -10, 11, -12, -12, -3, -4, -7, -6, -9, -10, 2, -8, 0, -4, 0, 1, -6, -7, -12, -3, -13, -19],
             "Q": [-8, 0, -3, -6, -12, 7, -1, -7, 1, -8, -2, 0, -6, -8, 0, -6, -5, -11, -6, -10, -5, 5, -2, -13, -19],
             "E": [-2, -4, -3, 2, -12, -1, 7, 0, -6, -8, -11, 0, -7, -13, -9, -7, -5, -12, -9, -3, 0, 5, -2, -13, -19],
             "G": [-1, 0, -4, -1, -3, -7, 0, 7, -7, -8, -10, -3, -9, -7, -8, 0, -4, -3, -10, -4, -2, -1, -3, -13, -19],
             "H": [-8, 1, 0, -1, -4, 1, -6, -7, 9, -7, -2, -2, -7, -3, -1, -3, -4, -8, 3, -10, 0, -1, -1, -13, -19],
             "I": [-2, -4, -4, -8, -7, -8, -8, -8, -7, 6, 0, -5, 2, 0, -7, -3, 0, -11, -7, 3, -5, -8, -1, -13, -19],
             "L": [-6, -4, -8, -12, -6, -2, -11, -10, -2, 0, 6, -7, 0, 1, -1, -4, -5, -4, -5, -1, -10, -5, -3, -13, -19],
             "K": [-5, 3, 0, -4, -9, 0, 0, -3, -2, -5, -7, 6, -3, -11, -6, -2, 0, -9, -9, -6, 0, 0, -1, -13, -19],
             "M": [-4, -1, -7, -10, -10, -6, -7, -9, -7, 2, 0, -3, 10, -4, -7, -6, 0, -9, -10, 1, -8, -6, -1, -13, -19],
             "F": [-9, -9, -8, -11, 2, -8, -13, -7, -3, 0, 1, -11, -4, 9, -7, -4, -7, -3, 3, -3, -9, -10, -2, -13, -19],
             "P": [-1, -3, -5, -9, -8, 0, -9, -8, -1, -7, -1, -6, -7, -7, 8, 0, -1, -11, -8, -8, -7, -2, -3, -13, -19],
             "S": [-1, 0, 2, -2, 0, -6, -7, 0, -3, -3, -4, -2, -6, -4, 0, 7, 1, -9, -4, -6, 0, -6, -1, -13, -19],
             "T": [2, -1, 0, -4, -4, -5, -5, -4, -4, 0, -5, 0, 0, -7, -1, 1, 6, -12, -7, -1, 0, -5, -1, -13, -19],
             "W": [-13, -4, -13, -12, 0, -11, -12, -3, -8, -11, -4, -9, -9, -3, -11, -9, -12, 10, -2, -12, -12, -11, -6, -13, -19],
             "Y": [-12, -6, -2, -4, 1, -6, -9, -10, 3, -7, -5, -9, -10, 3, -8, -4, -7, -2, 9, -9, -3, -7, -3, -13, -19],
             "V": [1, -7, -7, -4, -6, -10, -3, -4, -10, 3, -1, -6, 1, -3, -8, -6, -1, -12, -9, 6, -5, -5, -1, -13, -19],
             "B": [-3, -4, 6, 6, -7, -5, 0, -2, 0, -5, -10, 0, -8, -9, -7, 0, 0, -12, -3, -5, 7, 0, -2, -13, -19],
             "Z": [-4, -1, -3, 0, -12, 5, 5, -1, -1, -8, -5, 0, -6, -10, -2, -6, -5, -11, -7, -5, 0, 6, -3, -13, -19],
             "X": [-2, -1, -1, -2, -3, -2, -2, -3, -1, -1, -3, -1, -1, -2, -3, -1, -1, -6, -3, -1, -2, -3, -2, -13, -19],
             "*": [-13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, 1, -19],
             "-": [-19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, 1]}
        self._embedding = None
        self.__accept_non_alpha = accept_non_alpha

    def isalpha(self):
        return self.sequence.isalpha()

    def colour(self, colouring_type: int):
        if colouring_type == 0:
            pass
        elif colouring_type == 1:
            for base, change in self.flavor_pp_dict.items():
                self.sequence = self.sequence.replace(base, change)
        elif colouring_type == 2:
            for base, change in self.flavor_ak_dict.items():
                self.sequence = self.sequence.replace(base, change)
        elif colouring_type == 3:
            for base, change in self.flavor_sw_dict.items():
                self.sequence = self.sequence.replace(base, change)
        elif colouring_type == 4:
            for base, change in self.flavor_ct_dict.items():
                self.sequence = self.sequence.replace(base, change)
        elif colouring_type == 5:
            for base, change in self.flavor_hp_dict.items():
                self.sequence = self.sequence.replace(base, change)

    def get_charset(self) -> set:
        return set(self.sequence)

    def embed(self, charset: list, embedding_type: int):
        if embedding_type == 7:
            self._embedding = np.zeros((self.length, len(charset)), dtype=np.int8)
            for i, char in enumerate(self.sequence):
                if char not in charset:
                    raise NotSeenSymbolException(f'{char} symbol is not in the given header!')
                self._embedding[i, charset.index(char)] = 1
        elif embedding_type == 8:
            self._embedding = np.zeros((self.length, len(self.blosum62.keys())), dtype=np.int8)
            for i, char in enumerate(self.sequence):
                if char not in charset:
                    raise NotSeenSymbolException(f'{char} symbol is not in the given header!')
                self._embedding[i] += self.blosum62[char]
        elif embedding_type == 9:
            self._embedding = np.zeros((self.length, len(self.hivb.keys())), dtype=np.int8)
            for i, char in enumerate(self.sequence):
                if char not in charset:
                    raise NotSeenSymbolException(f'{char} symbol is not in the given header!')
                self._embedding[i] += self.hivb[char]

    def seq2kmers(self, min_length: int, max_length: int, shift: int = 1, normalize: bool = False,
                  binarized: bool = False) -> defaultdict:
        """
        Function for generation of simple k-mers
        :param min_length: minlength of k-mer
        :param max_length: maxlength of k-mer
        :param shift: sequence shift for k-mer calculation. 1 by default
        :param normalize: normalize k-mers
        :param binarized: to binarize the output
        :return: generator
        """
        kmers = defaultdict(int)
        for length in range(min_length, max_length + 1):
            n = 0
            if binarized:
                for x in range(0, self.length - length + 1, shift):
                    kmers[self.sequence[x:x + length]] = 1
            else:
                for x in range(0, self.length - length + 1, shift):
                    kmers[self.sequence[x:x + length]] += 1
                    n += 1
            if normalize and not binarized:
                for x in range(0, self.length - length + 1, shift):
                    kmers[self.sequence[x:x + length]] /= n
                    kmers[self.sequence[x:x + length]] = round(kmers[self.sequence[x:x + length]], 3)
        return kmers

    def seq2pckmers(self, min_length: int, max_length: int, length_step: int = 1, level: int = 2,
                    shift: int = 1, normalize: bool = False, binarized: bool = False) -> defaultdict:
        """
        Function for generation of position-centered k-mers
        :param min_length: mininal length of k-mer
        :param max_length: maximal length of k-mer
        :param length_step: step between min- and maxlength k-mer
        :param level: level of MNN descriptors (see Tarasova et al., doi:10.3390/ijms21030748)
        :param shift: sequence shift for k-mer calculation. 1 by default
        :param normalize: normalize k-mers
        :param binarized: to binarize the output
        :return: generator
        """
        kmers = defaultdict(int)
        if level == 1:
            for length in range(min_length, max_length + 1, length_step):
                n = 0
                if binarized:
                    for x in range(length - 1, self.length - length + 1, shift):
                        kmers[self.sequence[x + 1 - length:x + 1]] = 1
                        kmers[self.sequence[x:x + length]] = 1
                else:
                    for x in range(length - 1, self.length - length + 1, shift):
                        kmers[self.sequence[x + 1 - length:x + 1]] += 1
                        kmers[self.sequence[x:x + length]] += 1
                        n += 2
                if normalize and not binarized:
                    for x in range(length - 1, self.length - length + 1, shift):
                        kmers[self.sequence[x + 1 - length:x + 1]] = round(kmers[self.sequence[x + 1 - length:x + 1]] / n, 3)
                        kmers[self.sequence[x:x + length]] = round(kmers[self.sequence[x:x + length]] / n, 3)
        elif level == 2:
            for length in range(min_length, max_length + 1, length_step):
                n = 0
                if binarized:
                    for x in range(length - 1, self.length - length + 1, shift):
                        kmers[self.sequence[x + 1 - length:x + length]] = 1
                else:
                    for x in range(length - 1, self.length - length + 1, shift):
                        kmers[self.sequence[x + 1 - length:x + length]] += 1
                        n += 1
                if normalize and not binarized:
                    for x in range(length - 1, self.length - length + 1, shift):
                        kmers[self.sequence[x + 1 - length:x + length]] = round(kmers[self.sequence[x + 1 - length:x + length]] / n, 3)
        elif level == 12:
            for length in range(min_length, max_length + 1, length_step):
                n = 0
                if binarized:
                    for x in range(length - 1, self.length - length + 1, shift):
                        kmers[self.sequence[x + 1 - length:x + 1]] = 1
                        kmers[self.sequence[x:x + length]] = 1
                        kmers[self.sequence[x + 1 - length:x + length]] = 1
                else:
                    for x in range(length - 1, self.length - length + 1, shift):
                        kmers[self.sequence[x + 1 - length:x + 1]] += 1
                        kmers[self.sequence[x:x + length]] += 1
                        kmers[self.sequence[x + 1 - length:x + length]] += 1
                        n += 3
                if normalize and not binarized:
                    for x in range(length - 1, self.length - length + 1, shift):
                        kmers[self.sequence[x + 1 - length:x + 1]] = round(kmers[self.sequence[x + 1 - length:x + 1]] / n, 3)
                        kmers[self.sequence[x:x + length]] = round(kmers[self.sequence[x:x + length]] / n, 3)
                        kmers[self.sequence[x + 1 - length:x + length]] = round(kmers[self.sequence[x + 1 - length:x + length]] / n, 3)

        return kmers

    def copy(self):
        sequence_copy = Sequence(self.sequence, self.__accept_non_alpha)
        return sequence_copy

    @property
    def embedding(self):
        if isinstance(self._embedding, np.ndarray):
            return self._embedding.flatten()
        else:
            return None

    @property
    def length(self):
        return len(self.sequence)


def check_inputs(args: argparse.Namespace):
    #for desc_type in [7, 8]:  # TODO rewrite this part in a proper way
    #    if len(args.type) > 1 and 0 < args.type.count(desc_type) < len(args.type):
    #        raise Exception("It is not allowed to generate fingerprints together with non-binary fragments.")
    #    if desc_type not in args.type and (len(args.type) != len(args.min_length) or len(args.type) != len(args.max_length) or
    #                               len(args.type) != len(args.shift) or len(args.type) != len(args.revcomp) or
    #                               len(args.type) != len(args.flavor)):
    #        raise Exception(f'There is an inconvenience in non-binary fragmentation setups: '
    #                        f'{len(args.type)} non-binary fragmentations, {len(args.min_length)} minimal lengths, '
    #                        f'{len(args.max_length)} maximal lengths, {len(args.shift)} shifts, '
    #                        f'{len(args.step)} steps, {len(args.revcomp)} reverse compliments and '
    #                        f'{len(args.flavor)} Reduced/Expanded alphabets.')
    # if desc_type in args.type and args.revcomp:  # TODO correct sanity check for reverse complement
    #    raise Exception('Reverse complement can not be used for sequence type "protein". The program will be aborted.')
    nt_2_4_6 = (args.type.count(2) + args.type.count(4) + args.type.count(6))
    if nt_2_4_6 != len(args.level):
        raise Exception(f'Number of position-dependent fragmentation types does not coincide with the number of '
                        f'given levels: {args.type.count(2) + args.type.count(4) + args.type.count(6)} against '
                        f'{len(args.level)}!')
    if nt_2_4_6 != len(args.step):
        raise Exception(f'Number of position-dependent fragmentation types does not coincide with the number of '
                        f'given steps: {args.type.count(2) + args.type.count(4) + args.type.count(6)} against '
                        f'{len(args.step)}!')


def print_arguments(args: argparse.Namespace):
    message = f'{"Input file:": <16} {args.input_file.name}\n' \
              f'{"Output file:": <16} {args.output}.{args.format}\n' \
              f'{"Header file:": <16} {"taken as " + args.hdr.name if args.hdr else args.output + ".hdr"}\n' \
              f'{"Types:": <16} {args.type}\n' \
              f'{"Minimal lengths:": <16} {args.min_length}\n' \
              f'{"Maximal lengths:": <16} {args.max_length}\n' \
              f'{"Shifts:": <16} {args.shift}\n' \
              f'{"Steps:": <16} {args.step}\n' \
              f'{"Levels:": <16} {args.level}\n' \
              f'{"Reverse compl.:": <16} {args.revcomp}\n' \
              f'{"Flavor:": <16} {args.flavor}\n' \
              f'{"*"*80}'
    print(message)


def read_hdr(header: TextIOWrapper) -> list:
    """
    Reads the given header file.
    :param header: File with precomputed k-mers.
    :return: k-mers list.
    """
    kmers = []
    for line in header:
        if not line.strip():
            continue
        kmer = line.split('.')[1].strip()
        kmers.append(kmer)

    return kmers


def write_hdr(kmers: list, header: TextIO):
    """
    Writes the header file.
    :param kmers: Dictionary of k-mers, where each k-mer is a key, and there is its index starting with zero as a value.
    :param header: ioStream to write the file.
    """
    for index, kmer in enumerate(kmers, 1):
        header.write(f'{str(index)+".": <9}{kmer: >119}\n')


def dump_fingerprint_buffer(filename: str, file_format: str, buffer: list):
    buffer = np.array(buffer, dtype=np.int8)
    buffer = np.array(buffer, dtype=np.str)
    ids = np.array([['?'] for _ in range(len(buffer))], dtype=np.str)
    buffer = np.concatenate((ids, buffer), axis=1)
    if file_format == 'csv':
        with open(filename + '.csv', 'ab') as out:
            np.savetxt(out, buffer, delimiter=',', fmt='%s')
    elif file_format == 'svm':
        with open(filename + '.svm', 'a') as out:
            buffer_str = []
            for seq in buffer:
                buffer_str.append(f'{seq[0]} {" ".join([str(j)+":"+v for j, v in enumerate(seq[1:-1], 1) if int(v) != 0])} '
                                  f'{len(seq)-1}:{seq[-1]}\n')
            out.writelines(buffer_str)
            del buffer_str


def compute_revcomp(kmers: defaultdict, binary: bool = False) -> defaultdict:
    """
    Transforms the k-mers dictionary according to reverse compliments.
    :param kmers: input k-mers dictionary, where a k-mer is a key, and its frequency in a sequence is a value.
    :param binary: represent data in a binary mode.
    :return dict: new k-mers dictionary where each k-mer is represented by both complement and reverse compliment.
    """
    compliment_na = {"A": "T", "T": "A", "C": "G", "G": "C"}  # dictionary of bases' complements
    complimented_kmers = defaultdict(int)
    compliments = {}
    for kmer in kmers:
        compliment = []
        for char in kmer:
            if char not in compliment_na:
                print(f'Warning: {char} is not in compliments dictionary!')
                compliment.append(char)
            else:
                compliment.append(compliment_na[char])
        compliment.reverse()
        compliments[''.join(compliment)] = kmer

    for revcomp_kmer, kmer in compliments.items():
        n1, n2 = 0, 0
        if revcomp_kmer in kmers:
            n1 = kmers[revcomp_kmer]
        if kmer in kmers:
            n2 = kmers[kmer]
        n = n1 + n2
        if binary and n:
            n = 1
        complimented_kmers['/'.join(sorted([revcomp_kmer, kmer]))] = n

    return complimented_kmers


def update(original_dict: defaultdict, new_dict: defaultdict):
    """
    Updates the given k-mers dictionary with a newly obtained k-mers dictionary.
    :param original_dict:
    :param new_dict:
    :return: New dictionary.
    """
    for kmer, value in new_dict.items():
        if kmer in original_dict:
            original_dict[kmer] += value
            print(f'WARNING: duplicated k-mer {kmer} has been found!')
        else:
            original_dict[kmer] = value


def postprocess_onehot_header(kmers: list, mode: int = 0, seq_length: int = 0) -> list:
    """
    This is an attempt to adapt one-hot header to SVM format specifics.
    :param kmers: List of tokens (generated or imported).
    :param mode: Usage mode: 0 - transform the header to dump it, 1 - transform the header to upload and use it.
    :param seq_length: Length of sequences.
    :return: Resulting k-mers.
    """
    if mode == 0:
        new_kmers = []
        for i in range(1, seq_length+1):
            s = [f'P{i}_{token}' for token in kmers]
            new_kmers.extend(s)
        return new_kmers
    else:
        new_kmers = []
        line_index = 0
        while 'P1_' in kmers[line_index]:
            new_kmers.append(kmers[line_index].split('_')[-1])
            line_index += 1
        return new_kmers


def main(input_file: TextIOWrapper, output_file: str, header_file: TextIOWrapper, types: list, min_lengths: list,
         max_lengths: list, shifts: list, steps: list, levels: list, revcomps: list, flavors: list,
         file_format: str = 'svm', buffer_size: int = 1000, accept_non_alpha: bool = False):
    kmers = []
    if header_file:
        print('Read the header file..')
        kmers = read_hdr(header_file)

    if os.path.exists(output_file+'.'+file_format):
        print(f'{output_file}.{file_format} exists. It will be removed!')
        os.remove(f'{output_file}.{file_format}')

    if 7 in types or 8 in types or 9 in types:
        if not kmers:
            print('Gathering of the header..')
            kmers_set = set()
            seq_length = 0
            if types[0] == 7:
                for i, sequence in enumerate(input_file, 1):
                    try:
                        sequence_obj = Sequence(sequence, accept_non_alpha=True)
                    except EmptySequenceError:
                        print(f'Line {i}: Empty sequence was found..')
                        continue
                    if not seq_length:
                        seq_length = sequence_obj.length
                    sequence_obj.colour(flavors[0])
                    kmers_set.update(sequence_obj.get_charset())
            elif types[0] == 8:
                sequence = input_file.readline().strip()
                sequence_obj = Sequence(sequence, accept_non_alpha=True)
                seq_length = sequence_obj.length
                kmers_set = {"A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L",
                             "K", "M", "F", "P", "S", "T", "W", "Y", "V", "B", "Z", "X", "*", "-"}
            elif types[0] == 9:
                sequence = input_file.readline().strip()
                sequence_obj = Sequence(sequence, accept_non_alpha=True)
                seq_length = sequence_obj.length
                kmers_set = {"A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L",
                             "K", "M", "F", "P", "S", "T", "W", "Y", "V", "B", "Z", "X", "*", "-"}
            kmers = sorted(list(kmers_set))
            del kmers_set
            print('Write the header..')
            tmp_kmers = postprocess_onehot_header(kmers, mode=0, seq_length=seq_length)
            write_hdr(tmp_kmers, open(output_file + '.hdr', 'w'))
            del tmp_kmers
        else:
            kmers = postprocess_onehot_header(kmers, mode=1)

        buffer = []
        print('Run fragmentation..')
        file_name = input_file.name
        input_file.close()
        input_file = open(file_name)
        for i, sequence in enumerate(input_file, 1):
            if i % buffer_size == 0:
                print(f'{i} line passed..')
            try:
                sequence = Sequence(sequence, accept_non_alpha=True)
            except EmptySequenceError:
                print(f'Line {i}: Empty sequence was found..')
                continue
            sequence.colour(flavors[0])
            try:
                sequence.embed(kmers, types[0])
            except NotSeenSymbolException:
                print(f'Line {i}: Unknown symbol has been found in a sequence. This sequence will be omitted..')
                continue
            buffer.append(sequence.embedding)
            if len(buffer) == buffer_size:
                dump_fingerprint_buffer(output_file, file_format, buffer)
                del buffer
                buffer = []
        if buffer:
            dump_fingerprint_buffer(output_file, file_format, buffer)
            del buffer
    else:
        buffer = []
        if not header_file:
            kmers = set()
        print('Run fragmentation..')
        for i, sequence in enumerate(input_file, 1):
            if i % buffer_size == 0:
                print(f'{i} sequences passed..')
            try:
                sequence_obj = Sequence(sequence, accept_non_alpha=accept_non_alpha)
            except EmptySequenceError:
                print(f'Line {i}: Empty sequence was found..')
                continue
            except NonAlphabetSymbolsException:
                print(f'Line {i}: Non alphabet symbols have been found in a sequence. This sequence will be '
                      f'omitted..')
                continue
            result = defaultdict(int)
            levels_copy = levels.copy()
            steps_copy = steps.copy()
            for frag_type, min_length, max_length, shift, revcomp, flavor in zip(types, min_lengths, max_lengths,
                                                                                 shifts, revcomps, flavors):
                sequence_copy = sequence_obj.copy()
                sequence_copy.colour(flavor)
                if frag_type in [1, 3, 5]:
                    seq_kmers = sequence_copy.seq2kmers(min_length, max_length, shift, normalize=(frag_type == 3),
                                                        binarized=(frag_type == 5))
                else:
                    seq_kmers = sequence_copy.seq2pckmers(min_length, max_length, steps_copy.pop(0), levels_copy.pop(0),
                                                          shift, normalize=(frag_type == 4),
                                                          binarized=(frag_type == 6))

                if revcomp:
                    seq_kmers = compute_revcomp(seq_kmers, binary=(frag_type == 6))

                update(result, seq_kmers)
                del seq_kmers
                del sequence_copy
            buffer.append(result)
            if not header_file:
                kmers.update(result.keys())
            if len(buffer) == buffer_size:
                with open(output_file+'.tmp', 'ab') as out:
                    pickle.dump(buffer, out)
                    del buffer
                    buffer = []
        if buffer:
            with open(output_file + '.tmp', 'ab') as out:
                pickle.dump(buffer, out)
                del buffer
        if not header_file:
            print('Write the header..')
            kmers = sorted(list(kmers))
            write_hdr(kmers, open(output_file + '.hdr', 'w'))
        print('Prepare output..')
        with open(output_file + '.tmp', 'rb') as pickle_input, open(f'{output_file}.{file_format}', 'w') as file_out:
            try:
                kmers_dict = {kmer: i for i, kmer in enumerate(kmers, 1)}
                while True:
                    buffer = pickle.load(pickle_input)
                    for line in buffer:
                        line[kmers[-1]] += 0
                        line_new = {kmers_dict[kmer]: value for kmer, value in line.items()}
                        if file_format == 'csv':
                            line_str = ','.join([f'{i}:{line_new[i]}' if i in line_new else f'{i}:0.0' for i in range(1, len(kmers)+1)])
                            file_out.write(f'?,{line_str}\n')
                        elif file_format == 'svm':
                            line_str = ' '.join([f'{i}:{value}' for i, value in sorted(line_new.items(), key=itemgetter(0))])
                            file_out.write(f'? {line_str}\n')
            except EOFError:
                pass
        try:
            os.remove(output_file + '.tmp')
        except:
            print('Cannot remove the tmp file..')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIDASeq tool is used to generate k-mers of amino acids/nucleotides "
                                                 "sequences (beta version).",
                                     epilog="Alexey Orlov, Arkadii Lin, Alexandre Varnek, Strasbourg 2021",
                                     prog="ISIDASeq", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', type=argparse.FileType('r'), dest='input_file', required=True,
                        help='Input file with sequences.')
    parser.add_argument('-o', type=str, dest='output', help='Path to the output files with no extension.',
                        required=True)
    parser.add_argument('-f', type=str, dest='format', default='svm', choices=['svm', 'csv'],
                        help='Format of the output file: svm or csv.')
    parser.add_argument('-hdr', type=argparse.FileType('r'), dest='hdr',
                        help='Path to an existing header (.hdr) file.')
    parser.add_argument('-t', type=int, dest='type', choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], action='append', required=True,
                        help='descriptor type:\n'
                             '  1 - position-independent k-mers DNA/RNA/Protein;\n'
                             '  2 - position-specific k-mers DNA/RNA/Protein;\n'
                             '  3 - normalized position-independent k-mers DNA/RNA/Protein;\n'
                             '  4 - normalized position-specific k-mers DNA/RNA/Protein;\n'
                             '  5 - binarized position-independent k-mers DNA/RNA/Protein;\n'
                             '  6 - binarized position-specific k-mers DNA/RNA/Protein;\n'
                             '  7 - one-hot encoding DNA/RNA/Protein sequence\n'
                             '  8 - BLOSUM62 encoding Protein sequence\n'
                             '  9 - HIVb encoding Protein sequence\n')

    parser.add_argument('-l', type=int, dest='min_length', action='append', help='Minimal length of a k-mer.')
    parser.add_argument('-u', type=int, dest='max_length', action='append', help='Maximal length of a k-mer.')
    parser.add_argument('-s', type=int, dest='shift', action='append',
                        help='Shift of a k-mer position in a sequence (1 by default).')
    parser.add_argument('-st', type=int, dest='step', action='append',
                        help='k-mer length step for position specific k-mers (1 for each descriptor type by default).')
    parser.add_argument('-lv', type=int, dest='level', action='append',
                        help='Position specific k-mers level. 2 levels are used by default.')
    parser.add_argument('-r', type=str, dest='revcomp', action="append", help='Apply reverse compliment (write On) or '
                                                                              'not (write Off).')
    parser.add_argument('-p', type=int, dest='flavor', action="append", choices=[0, 1, 2],
                        help='Apply Reduced/Expanded alphabets:\n  0 - do not use (by default);\n  1 - purine-pyrimidine reduced alphabet;'
                             '\n  2 - amino-keto reduced alphabet;\n  3 - Hbond reduced alphabet;\n  4 - reduced alphabet as in conjoint triad method;'
                             '\n  5 - reduced alphabet polar/neutral/hydrophobic.')
    parser.add_argument('-bs', '--buffer_size', type=int, dest='buffer_size', default=1000,
                        help='Buffer size (1000 sequences by default).')
    parser.add_argument('--accept_non_alpha', dest='acc_non_alpha', action="store_true",
                        help='Accept non alphabetic symbols in sequences.')

    args = parser.parse_args()
    if not args.step:
        args.step = [1 for _ in range(args.type.count(2) + args.type.count(4) + args.type.count(6))]

    if not args.shift:
        args.shift = [1 for _ in range(len(args.type))]

    if not args.level:
        args.level = [2 for _ in range(args.type.count(2) + args.type.count(4) + args.type.count(6))]

    if not args.revcomp:
        args.revcomp = [False for _ in range(len(args.type))]
    else:
        args.revcomp = [True if item.lower() == 'on' else False for item in args.revcomp]

    if not args.flavor:
        args.flavor = [0 for _ in range(len(args.type))]

    greeting = f'{"*" * 80}\n{"ISIDA-Seq v.0.2 version": ^80}\n{"This is a terminal interface for": ^80}\n' \
               f'{"generating descriptors of biological sequences": ^80}\n\n' \
               f'{"Alexey Orlov, Arkadii Lin, Alexandre Varnek": ^80}\n{"*" * 80}\n{"Strasbourg, 2021": ^80}\n' \
               f'{"*" * 80}'
    print(greeting)

    check_inputs(args)

    print_arguments(args)

    main(args.input_file, args.output, args.hdr, args.type, args.min_length, args.max_length, args.shift, args.step,
         args.level, args.revcomp, args.flavor, args.format, args.buffer_size, args.acc_non_alpha)