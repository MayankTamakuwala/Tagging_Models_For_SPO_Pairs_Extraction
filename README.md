# SPO Pair Extraction for Knowledge Graph Construction

This project explores two tagging-based approaches to extract structured Subject–Predicate–Object (SPO) triplets from unstructured text data scraped from the web. These triplets are foundational for building knowledge graphs that power information retrieval, semantic search, and question answering systems.

## 📌 Objective

To extract SPO pairs from domain-specific news articles using:
- Viterbi Algorithm (HMM-based)
- Fine-tuned BERT with BIO Tagging

The project compares traditional sequence models with transformer-based deep learning for token-level classification and relation extraction.

## 📂 Dataset

We scraped press releases from GlobeNewswire across three sectors:
- Finance
- Healthcare
- Technology

Each article was paired with manually annotated SPO triplets to serve as ground truth. Text was tokenized using spaCy and aligned with the BIO tag format.

## 🧠 Models

1) 🔁 Viterbi Algorithm (Hidden Markov Model)
- Implemented using NLTK’s HiddenMarkovModelTrainer
- Input: Sequences of (word, tag) pairs
- Output: Most probable tag sequence using the Viterbi decoding
- Evaluation: Token-level accuracy, F1, and triplet-level semantic scores (ROUGE, BLEU)

2) 🧠 BERT + BIO Tagging
- Base model: bert-base-cased
- Token-level classification for 7 BIO tags: O, B-SUB, I-SUB, B-PRED, I-PRED, B-OBJ, I-OBJ
- Post-processing: Tokens are grouped into triplets using sentence segmentation (spaCy
- Evaluation: Precision, Recall, F1, BLEU, ROUGE

## ⚙️ Training Details (BERT)

- **Optimizer**: AdamW
- **Learning Rate**: 5e-5
- **Batch Size**: 8
- **Epochs**: 10
- **Dropout**: 0.1
- **Train/Val Split**: 90/10

Pretrained weights are auto-downloaded from HuggingFace for reproducibility.


## 📈 Results Summary

| Metric           | BERT        | Viterbi     |
|------------------|-------------|-------------|
| Accuracy         | 93.79%      | 24.11%      |
| Macro F1         | 37.61%      | 36.81% (SUBJ) |
| BLEU Score       | 3.64%       | 19.71%      |
| ROUGE-L (F1)     | 28.64%      | 58.08%      |

- BERT excels in fine-grained token labeling
- Viterbi performs better in semantic structure matching (BLEU/ROUGE)

## 📊 Visualizations

- Confusion matrices
- Error pattern plots
- Tag distribution graphs
- Sequence length histograms

These insights help understand per-tag performance and sequence modeling challenges.

## 📁 Project Structure

```
Knowledge-Graphs/
├── models/
│   ├── BERT + BIO_Tagging.py     # BERT training and inference
│   └── Viterbi/Viterbi.py        # HMM training and evaluation
├── data/                         # Web-scraped articles & ground truth
├── requirements.txt              # All dependencies
└── README.md                     # You're here
```

## 🚀 How to Run

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the models: **Option 1: Python Scripts**
   ```bash
   # Viterbi
   python models/Viterbi/Viterbi.py

   # BERT + BIO Tagging
   python models/BERT\ +\ BIO_Tagging.py
   ```

**Option 2: Jupyter Notebooks**
```bash
# Start Jupyter
jupyter notebook
```

Then navigate to and run:
- models/Viterbi/Viterbi.ipynb
- models/BERT + BIO_Tagging.ipynb

The notebooks contain the same code as the Python scripts but divided into cells for easier experimentation and visualization of intermediate results.

## 🔗 Resources

- Dataset: [GlobeNewswire](https://globenewswire.com)
- Report: See `Final Project Report.pdf` in this repo
- Authors: [Mayank Tamakuwala](https://mayank-tamakuwala.me) [(email)](mailto:tamakuwala.m@northeastern.edu), [Nithin Bhat](mailto:bhat.nith@@northeastern.edu), [Agasti Mhatre](mailto:mhatre.ag@northeastern.edu)

## 🧠 Acknowledgments

We thank the open-source developers behind:
- [spaCy](https://spacy.io)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [NLTK](https://www.nltk.org/)
- [SeqEval](https://github.com/chakki-works/seqeval)
- [Scikit-learn](https://scikit-learn.org)

Special thanks to **Prof. Uzair Ahmad** for project guidance.
