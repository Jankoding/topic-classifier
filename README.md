# Topic Classifier

A hybrid topic classification tool using zero-shot learning and semantic similarity. Built with HuggingFace and SentenceTransformers.

## Installation

```bash
pip install git+https://github.com/jankoding/topic-classifier.git

## Example usage:

from topic_classifier.classifier import run_classification

run_classification(
    data_folder='/content/drive/MyDrive/mytexts',
    output_folder='/content/drive/MyDrive/results',
    labels_file='/content/drive/MyDrive/labels.txt',
    topics_file='/content/drive/MyDrive/topics.txt'
)
