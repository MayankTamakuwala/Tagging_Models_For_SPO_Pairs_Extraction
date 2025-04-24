# %%
# I'm loading all the necessary libraries before proceeding to data loading and preprocessing.
# ! python -m venv venv
# ! source venv/bin/activate
# %pip install pandas transformers torch scikit-learn hf_xet spacy rouge nltk requests seqeval
# ! python -m spacy download en_core_web_sm
# OR
# %pip install -r ../requirements.txt

# %%
import csv
import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import requests
import os
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
import random

nlp = spacy.load("en_core_web_sm")

# %%
# I'm making sure the pretrained model weights are downloaded if not already present.
if not os.path.exists("./BERT_BIO_Tagging_model.pth"):
    # URL of my HuggingFace account where I've uploaded the trained model weights.
    url = "https://huggingface.co/MayankTamakuwala/BERT_BIO_Tagger/resolve/main/BERT_BIO_Tagging_model.pth"
    output_path = "BERT_BIO_Tagging_model.pth"

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Model weights downloaded to {output_path}")
else:
    print(f"Weights already exists")

# %%
# I am combining all sector-specific articles into a single DataFrame for processing.
articles_df = pd.DataFrame()
sectors = ["finance", "healthcare", "tech"]
triplets_list = []
for sector in sectors:
    articles = pd.read_csv(f"../Webscraped Dataset/globenewswire_articles_{sector}.csv")
    articles_df = pd.concat([articles_df, articles], ignore_index=True)

    with open(f"../Ground Truth/{sector}_articles_triplets.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        header = next(reader)
        
        for row in reader:
            if not row:
                continue
            
            url = row[0]
            triplet_fields = []
            
            for field in row[1:]:
                if field.strip():
                    str_tuple = field.strip()
                    if str_tuple.startswith('(') and str_tuple.endswith(')'):
                        inner_str = str_tuple[1:-1]
                        elements = [elem.strip() for elem in inner_str.split(',')]
                        triplet_fields.append(tuple(elements))
                    else:
                        triplet_fields.append((str_tuple,))

            triplets_list.append({"url": url, "triplets": triplet_fields})

triplets_df = pd.DataFrame(triplets_list)

# %%
def tokenize_text(text):
    return [token.text for token in nlp(text)]

# %%
def get_bio_tags(text, spo_list):
    # I'm trying to align SPO triplets with token spans to generate BIO tags.
    tokens = tokenize_text(text)
    tags = ['O'] * len(tokens)

    for spo in spo_list:
        try:
            subject, predicate, obj = spo
            spans = {
                'SUB': subject.split(),
                'PRED': predicate.split(),
                'OBJ': obj.split()
            }

            for label, span_tokens in spans.items():
                for i in range(len(tokens) - len(span_tokens) + 1):
                    if tokens[i:i+len(span_tokens)] == span_tokens:
                        tags[i] = f'B-{label}'
                        for j in range(1, len(span_tokens)):
                            tags[i + j] = f'I-{label}'
                        break
        except Exception as e:
            continue

    return tokens, tags

# %%
# I'm now creating the final dataset by aligning tokens and their corresponding BIO tags.
dataset = []

for idx, row in articles_df.iterrows():
    url = row['url']
    content = row['content']

    matching_triplets_row = triplets_df[triplets_df['url'] == url]
    if matching_triplets_row.empty:
        continue

    triplets = triplets_df.iloc[idx, 1]
    tokens, tags = get_bio_tags(content, triplets)
    dataset.append((tokens, triplets, tags))

# %%
# Here, I'm initializing the tokenizer and defining tag mappings for the BIO tagging scheme.
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

# # Unique BIO tags
tag_values = ['O', 'B-SUB', 'I-SUB', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
tag2id = {tag: i for i, tag in enumerate(tag_values)}
id2tag = {i: tag for tag, i in tag2id.items()}

# %% [markdown]
# # Training

# %%
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print ("MPS device found.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print ("CUDA device found.")
else:
    device = torch.device("cpu")
    print ("Using CPU.")

class BERT_SPO_BIO_Tagger(nn.Module):
    # I am defining my custom BERT-based model for sequence tagging using BIO labels.
    def __init__(self, tag2id, id2tag, tokenizer, lr=5e-5, epochs = 10):
        super(BERT_SPO_BIO_Tagger, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(tag2id))
        self.__tag2id = tag2id
        self.__train_loader = None
        self.__id2tag = id2tag
        self.__val_loader = None
        self.__lr = lr
        self.__epochs = epochs
        self.__tokenizer = tokenizer
        self.__nlp = spacy.load("en_core_web_sm")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        return logits

    def fit(self, dataset):
        class SPOBioDataset(Dataset):
            def __init__(self, data, tokenizer, tag2id, max_len=512):
                self.data = data
                self.tokenizer = tokenizer
                self.tag2id = tag2id
                self.max_len = max_len

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                tokens, _, tags  = self.data[idx]

                tokenized_input = self.tokenizer(tokens,
                                                is_split_into_words=True,
                                                padding='max_length',
                                                truncation=True,
                                                max_length=self.max_len,
                                                return_tensors="pt")

                word_ids = tokenized_input.word_ids(batch_index=0)
                label_ids = []

                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    else:
                        label_ids.append(self.tag2id.get(tags[word_idx], self.tag2id['O']))

                return {
                    'input_ids': tokenized_input['input_ids'].squeeze(),
                    'attention_mask': tokenized_input['attention_mask'].squeeze(),
                    'labels': torch.tensor(label_ids)
                }

            def get_raw_item(self, idx):
                return self.data[idx] 
            
        train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

        train_dataset = SPOBioDataset(train_data, self.__tokenizer, self.__tag2id)
        val_dataset = SPOBioDataset(val_data, self.__tokenizer, self.__tag2id)

        self.__train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        self.__val_loader = DataLoader(val_dataset, batch_size=8)
        
        weights = self.__compute_class_weights(train_data)

        optimizer = optim.AdamW(self.parameters(), lr=self.__lr)
        loss_func = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)

        # Finally, I'm starting the training loop to fine-tune the model on my custom dataset.
        for epoch in range(self.__epochs):
            self.train()
            total_loss = 0
            
            for batch in self.__train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                logits = self(input_ids, attention_mask)

                loss = loss_func(logits.view(-1, len(self.__tag2id)), labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.__train_loader)
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        return self

    def __compute_class_weights(self, dataset):
        tag_counts = Counter(tag for _, _, tags in dataset for tag in tags)
        total = sum(tag_counts.values())
        weights = [1.0 - (tag_counts[tag] / total) for tag in self.__tag2id.keys()]
        # weights[0] += 0.14
        weights[0] += 0.1
        return torch.tensor(weights).to(device)

    @torch.no_grad()
    def evaluate_on_validation_data(self):
        # I am now evaluating the model using F1, Precision, Recall, ROUGE, and BLEU metrics.
        self.eval()
        seqeval_true = []
        seqeval_pred = []

        val_dataset = self.__val_loader.dataset

        for batch in self.__val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            logits = self(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=2)

            for i in range(len(labels)):
                true_tags = []
                pred_tags = []
                for j in range(len(labels[i])):
                    if labels[i][j] != -100:
                        true_tag = self.__id2tag[labels[i][j].item()]
                        pred_tag = self.__id2tag[predictions[i][j].item()]
                        true_tags.append(true_tag)
                        pred_tags.append(pred_tag)
                seqeval_true.append(true_tags)
                seqeval_pred.append(pred_tags)

        seqeval_metrics = {
            "classification_report": classification_report(
                [tag for seq in seqeval_true for tag in seq],
                [tag for seq in seqeval_pred for tag in seq],
                digits=4),
            "f1": f1_score(seqeval_true, seqeval_pred),
            "precision": precision_score(seqeval_true, seqeval_pred),
            "recall": recall_score(seqeval_true, seqeval_pred)
        }

        predicted_triplets_all = []
        reference_triplets_all = []

        val_dataset = self.__val_loader.dataset

        for i in range(len(val_dataset)):
            tokens, true_triplets, _ = val_dataset.get_raw_item(i)
            text = " ".join(tokens)
            pred_tags = self.__predict_bio_tags(tokens)
            pred_triplets = self.__extract_and_form_triplets(text, pred_tags)

            predicted_triplets_all.append(" ".join([" ".join(triplet) for triplet in pred_triplets]))
            reference_triplets_all.append(" ".join([" ".join(triplet) for triplet in true_triplets]))

        rouge = Rouge()
        rouge_scores = []
        smoothie = SmoothingFunction().method4
        bleu_scores = []

        for ref, pred in zip(reference_triplets_all, predicted_triplets_all):
            try:
                score = rouge.get_scores(pred, ref)[0]
                rouge_scores.append(score)
            except:
                continue
            bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
            bleu_scores.append(bleu)

        avg_rouge = {}
        if rouge_scores:
            keys = rouge_scores[0].keys()
            for k in keys:
                avg_rouge[k] = {
                    "f": sum(d[k]["f"] for d in rouge_scores) / len(rouge_scores),
                    "p": sum(d[k]["p"] for d in rouge_scores) / len(rouge_scores),
                    "r": sum(d[k]["r"] for d in rouge_scores) / len(rouge_scores),
                }

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

        return {
            "seqeval": seqeval_metrics,
            "ROUGE": avg_rouge,
            "BLEU": avg_bleu
        }

    def load_model_weights(self, torch_load_weights):
        self.load_state_dict(torch_load_weights)

    @torch.no_grad()
    def __predict_bio_tags(self, tokens):
        self.eval()

        tokenized_input = self.__tokenizer(tokens,
                                    is_split_into_words=True,
                                    return_tensors="pt",
                                    truncation=True,
                                    padding="max_length",
                                    max_length=512)

        input_ids = tokenized_input["input_ids"].to(device)
        attention_mask = tokenized_input["attention_mask"].to(device)

        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=2)

        word_ids = tokenized_input.word_ids(batch_index=0)
        predicted_tags = []

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            tag_id = predictions[0][idx].item()
            tag = self.__id2tag[tag_id]
            if idx == 0 or word_ids[idx] != word_ids[idx - 1]:
                predicted_tags.append((tokens[word_idx], tag))

        return predicted_tags

    def __extract_and_form_triplets(self, text, tagged_tokens):
        # I'm reconstructing SPO triplets by checking which entities co-occur in the same sentence.
        spans = {'SUB': [], 'PRED': [], 'OBJ': []}
        current_span = []
        current_label = None

        for token, tag in tagged_tokens:
            if tag == 'O':
                if current_span and current_label:
                    spans[current_label].append(" ".join(current_span))
                current_span = []
                current_label = None
            elif tag.startswith('B-'):
                if current_span and current_label:
                    spans[current_label].append(" ".join(current_span))
                current_label = tag[2:]
                current_span = [token]
            elif tag.startswith('I-') and current_label == tag[2:]:
                current_span.append(token)
            else:
                if current_span and current_label:
                    spans[current_label].append(" ".join(current_span))
                current_span = []
                current_label = None

        if current_span and current_label:
            spans[current_label].append(" ".join(current_span))

        filtered_spans = spans

        doc = self.__nlp(text)
        triplets = []

        for sent in doc.sents:
            sent_text = sent.text
            subjs = [s for s in filtered_spans["SUB"] if s in sent_text]
            preds = [p for p in filtered_spans["PRED"] if p in sent_text]
            objs = [o for o in filtered_spans["OBJ"] if o in sent_text]

            for s in subjs:
                for p in preds:
                    for o in objs:
                        triplets.append((s, p, o))

        return list(set(triplets))

# %%
model = BERT_SPO_BIO_Tagger(tag2id, id2tag, tokenizer, epochs = 0).to(device)

# to actually train the model, increase the nuber of epochs 
# comment the line below to not load the pretrained weights
model.load_model_weights(torch.load("BERT_BIO_Tagging_model.pth"))

model = model.fit(dataset)

# %% [markdown]
# # Evaluate Validation 

# %%
scores = model.evaluate_on_validation_data()

# %%
print(scores["seqeval"]["classification_report"])

# %%
scores["seqeval"]["f1"]

# %%
scores["seqeval"]["precision"]

# %%
scores["seqeval"]["recall"]

# %%
scores["ROUGE"]

# %%
scores["BLEU"]

# %% [markdown]
# # Inference

# %%
@torch.no_grad()
def predict_bio_tags(text, model, tokenizer, id2tag, device):
    model.eval()
    tokens = tokenize_text(text)

    tokenized_input = tokenizer(tokens,
                                is_split_into_words=True,
                                return_tensors="pt",
                                truncation=True,
                                padding="max_length",
                                max_length=512)

    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(logits, dim=2)

    word_ids = tokenized_input.word_ids(batch_index=0)
    predicted_tags = []

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        tag_id = predictions[0][idx].item()
        tag = id2tag[tag_id]
        if idx == 0 or word_ids[idx] != word_ids[idx - 1]:
            predicted_tags.append((tokens[word_idx], tag))

    return predicted_tags

def extract_and_form_triplets(text, tagged_tokens):

    # Step 1: Extract spans from BIO-tagged tokens
    spans = {'SUB': [], 'PRED': [], 'OBJ': []}
    current_span = []
    current_label = None

    for token, tag in tagged_tokens:
        if tag == 'O':
            if current_span and current_label:
                spans[current_label].append(" ".join(current_span))
            current_span = []
            current_label = None
        elif tag.startswith('B-'):
            if current_span and current_label:
                spans[current_label].append(" ".join(current_span))
            current_label = tag[2:]
            current_span = [token]
        elif tag.startswith('I-') and current_label == tag[2:]:
            current_span.append(token)
        else:
            if current_span and current_label:
                spans[current_label].append(" ".join(current_span))
            current_span = []
            current_label = None

    if current_span and current_label:
        spans[current_label].append(" ".join(current_span))

    # Step 2: Filter out short or lowercase-only spans
    # def filter_spans(spans):
    #     def is_valid(span):
    #         return len(span.split()) > 1 or span[0].isupper()

    #     return {
    #         k: [s for s in v if is_valid(s)] for k, v in spans.items()
    #     }

    # filtered_spans = filter_spans(spans)

    filtered_spans = spans

    # Step 3: Match spans within same sentence
    doc = nlp(text)
    triplets = []

    for sent in doc.sents:
        sent_text = sent.text
        subjs = [s for s in filtered_spans["SUB"] if s in sent_text]
        preds = [p for p in filtered_spans["PRED"] if p in sent_text]
        objs = [o for o in filtered_spans["OBJ"] if o in sent_text]

        for s in subjs:
            for p in preds:
                for o in objs:
                    triplets.append((s, p, o))

    return list(set(triplets))

# %%
# Finally, I'm running inference on a random sample to see the predicted SPO triplets.
sample_text = articles_df.iloc[random.randint(0, articles_df.shape[0] - 1)]["content"]

tagged = predict_bio_tags(sample_text, model, tokenizer, id2tag, device)
triplets = extract_and_form_triplets(sample_text, tagged)

print("Predicted Triplets:")
for t in triplets:
    print(t)


