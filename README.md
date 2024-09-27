# MAMMOTH for Sign Language Translation
This project adapts the 🦣 MAMMOTH toolkit, built on top of OpenNMT-py, for sign language translation, leveraging the Phoenix2014T dataset. This work contributes to the advancement of neural machine translation in the domain of sign language using modular, research-friendly tools from Helsinki-NLP.


## Installation (install.sh)

1. Installation is the same as provided by mammoth library. 
   In addition, sentencepiece and sacrebleu are also installed. 


## Project Structure
   - configs dir: the configuration json file
   - data dir: includes the Phoenix2014T sign language translation dataset
   - vocabs dir: includes the tokenizer vocabulary

## Usage
   1. Download the Phoenix2014T dataset files:
   ```
   $ ./download.sh
   ```

   2. Run:

   ```
   sbatch job.sh
   ```
