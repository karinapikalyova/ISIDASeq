# ISIDASeq v.0.1

## Description
Open-source tool for numerical encoding of nucleic acid and amino acid sequences. The ISIDASeq framework integrates generation of k-mers of different length, grouped by various physico-chemical properties, with diverse normalization types, and inclusion of reverse complements (to avoid overcounting of k-mers that are mirror representations of each other).

# Examples of command inputs
## For DNA sequenes:

```
python /home/user/src/ISIDASeq.py -i dna_sequences.seq -o DIO1010S1F -t 1 -l 10 -u 10 -s 1 -p 1 -bs 300

```

## For protein sequences:
```
python /home/user/src/ISIDASeq.py -i protein_sequences.seq -o PIIIOF -t 7 -p 2

```
 <br />
optional arguments: <br />

  -h, --help            show this help message and exit <br />
  -i INPUT_FILE         Input file with sequences. <br />
  -o OUTPUT             Path to the output files with no extension. <br />
  -f {svm,csv}          Format of the output file: svm or csv. <br />
  -hdr HDR              Path to an existing header (.hdr) file. <br />
  -t {1,2,3,4,5,6,7,8,9}: <br />
                         descriptor type: <br />
                          1 - position-independent k-mers DNA/RNA/Protein; <br />
                          2 - position-specific k-mers DNA/RNA/Protein; <br />
                          3 - normalized position-independent k-mers DNA/RNA/Protein; <br />
                          4 - normalized position-specific k-mers DNA/RNA/Protein; <br />
                          5 - binarized position-independent k-mers DNA/RNA/Protein; <br />
                          6 - binarized position-specific k-mers DNA/RNA/Protein; <br />
                          7 - one-hot encoding DNA/RNA/Protein sequence <br />
                          8 - BLOSUM62 encoding Protein sequence <br />
                          9 - HIVb encoding Protein sequence <br />
  -l MIN_LENGTH         Minimal length of a k-mer. <br />
  -u MAX_LENGTH         Maximal length of a k-mer. <br />
  -s SHIFT              Shift of a k-mer position in a sequence (1 by default). <br />
  -st STEP              k-mer length step for position specific k-mers (1 for each descriptor type by default). <br />
  -lv LEVEL             Position specific k-mers level. 2 levels are used by default. <br />
  -r REVCOMP            Apply reverse compliment (write On) or not (write Off). <br />
  -p {0,1,2}            Apply Reduced/Expanded alphabets: <br />
                          0 - do not use (by default); <br />
                          1 - purine-pyrimidine reduced alphabet; <br />
                          2 - amino-keto reduced alphabet; <br />
                          3 - Hbond reduced alphabet; <br />
                          4 - reduced alphabet as in conjoint triad method; <br />
                          5 - reduced alphabet polar/neutral/hydrophobic. <br />
  -bs BUFFER_SIZE, --buffer_size BUFFER_SIZE <br />
                        Buffer size (1000 sequences by default). <br />
  --accept_non_alpha    Accept non alphabetic symbols in sequences. <br />

Licensed under the [BSD 3-Clause License](LICENSE).
