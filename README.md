# ISIDASeq v.0.1

## Description
Open-source tool for numerical encoding of nucleic acid and amino acid sequences. The ISIDASeq framework integrates generation of k-mers of different length, grouped by various physico-chemical properties, with diverse normalization types, and inclusion of reverse complements (to avoid overcounting of k-mers that are mirror representations of each other).

# Examples of comman runs  
## For protein sequenes:
```bash
python /home/user/src/ISIDASeq.py -i final_seq.seq -o DIO1010S1F -t 1 -l 10 -u 10 -s 1 -p 1 -bs 300

```

## For DNA sequences:
