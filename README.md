# 📘 Named Entity Recognition with BERT and CoNLL-2003 Dataset

This notebook demonstrates how to fine-tune a BERT model for Named Entity Recognition (NER) using the Hugging Face `transformers` library and the CoNLL-2003 dataset. The model is evaluated using the `seqeval` metric from the `evaluate` library.

## 📦 Requirements

To run this notebook, make sure you have the following Python packages installed:

pip install transformers datasets evaluate seqeval

less
Copy
Edit

## 📁 Dataset

We use the [CoNLL-2003](https://www.aclweb.org/anthology/W03-0419.pdf) dataset, a well-known benchmark for NER. It includes annotations for the following entity types:

- `PER` (person)
- `ORG` (organization)
- `LOC` (location)
- `MISC` (miscellaneous)

This dataset is easily accessible via the `datasets` library.

## ⚙️ Key Libraries

We use the following libraries from the Hugging Face ecosystem:

```python
from transformers import Trainer, DataCollatorForTokenClassification
import evaluate

metric = evaluate.load("seqeval")
Trainer: A high-level API for model training and evaluation.

DataCollatorForTokenClassification: Automatically pads token classification batches during training.

evaluate.load("seqeval"): Loads the seqeval metric for NER evaluation, supporting precision, recall, and F1-score at the entity level.
 ```

## 🧠 Model
We fine-tune the bert-base-cased model for token classification. The model head is adapted for the number of unique NER tags in the dataset.

Example:

python
```bash
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))
```

## ✅ Training
The Hugging Face Trainer API is used to simplify training and evaluation:

Pass the model, tokenized datasets, and training arguments

Use DataCollatorForTokenClassification to handle padding

Use the seqeval metric in a compute_metrics function

## 🧪 Evaluation
The model is evaluated using seqeval, which computes:

Precision

Recall

F1 Score

These scores reflect entity-level correctness, which is more meaningful than token-level accuracy in NER tasks.

## 📊 Results
After training, the model is evaluated on the validation or test set. The seqeval metric reports F1 scores per entity type and overall performance.

## 🔄 Reuse & Extension
This notebook can be easily adapted to:

Use other transformer models (e.g., roberta-base, distilbert-base-cased)

Train on custom NER datasets

Integrate domain-specific entities for medical, legal, or financial text
