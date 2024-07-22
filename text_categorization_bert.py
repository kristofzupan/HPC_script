import pandas as pd
import numpy as np
import torch
from datasets import DatasetDict, Dataset
import evaluate
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
accuracy = evaluate.load("accuracy")

id2label = {0: 'Social', 1: 'Seminar', 2: 'Procedure', 3: 'Deliberation', 4: 'UX', 5: 'Imaginative Entry'}
label2id = {'Social': 0, 'Seminar': 1,'Procedure': 2,'Deliberation': 3, 'UX': 4, 'Imaginative Entry': 5}

def preprocess_function(examples):
    model_inputs = tokenizer(examples["Message"], max_length=tokenizer.model_max_length, truncation=True)
    #print(model_inputs)

    #model_inputs["labels"] =
    return model_inputs

def map_labels_to_ids(example):
    example['label'] = label2id[example['label']]
    return example

def compute_metrics(eval_pred):
    print(eval_pred)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    cm = confusion_matrix(labels, predictions)
    f1_micro = f1_score(labels, predictions, average='micro')
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels),
        "confusion_matrix": cm.tolist(),
        "f1_micro": f1_micro
    }

def main():
    t = tokenizer.encode("don't be so judgmental", return_tensors='pt')  # tokenizer will return pytorch tensors

    print(t)
    print(tokenizer.decode(t[0]))  # print decoded string with special tokens included
    print(tokenizer.decode(t[0], skip_special_tokens=True))

    df = pd.read_csv('./data/final_data.csv')

    print(len(df))
    print(df['Discussion Type'].unique())
    print(df[df['Discussion Type'].isnull()])
    print(df[df['Discussion Type'] == 'Others'])

    df = df.dropna(subset=['Discussion Type'])
    df = df[df['Discussion Type'] != 'Others']
    print("\n\nFIXED")
    print(len(df))
    print(df['Discussion Type'].unique())
    print(df[df['Discussion Type'].isnull()])
    print(df[df['Discussion Type'] == 'Others'])

    print(df["Discussion Type"])
    df = df.rename(columns={'Discussion Type': 'label'})
    print(df["label"])

    dataset = Dataset.from_pandas(df)
    print(dataset)

    train_dataset = dataset.shuffle(seed=1).select(range(int(len(dataset) * 0.65)))
    valid_dataset = dataset.shuffle(seed=1).select(range(int(len(dataset) * 0.65), int(len(dataset) * 0.85)))
    test_dataset = dataset.shuffle(seed=1).select(range(int(len(dataset) * 0.85), len(dataset)))

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validate': valid_dataset,
        'test': test_dataset
    })

    train_dataset = dataset_dict['train']
    valid_dataset = dataset_dict['validate']
    test_dataset = dataset_dict['test']

    print(dataset_dict['train'][0])

    target_param = "orig"
    tokenized_datasets = dataset_dict.map(preprocess_function, batched=True, batch_size=16)
    
    tokenized_datasets['train'] = tokenized_datasets['train'].map(map_labels_to_ids)
    tokenized_datasets['validate'] = tokenized_datasets['validate'].map(map_labels_to_ids)
    tokenized_datasets['test'] = tokenized_datasets['test'].map(map_labels_to_ids)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = BertForSequenceClassification.from_pretrained("bert-large-cased", num_labels=6, id2label=id2label, label2id=label2id).to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    training_args = TrainingArguments(
        output_dir="./runs",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validate"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.predict(tokenized_datasets["test"])
    print(results.metrics)

    array = results.metrics['test_confusion_matrix']

    df_cm = pd.DataFrame(array, range(6), range(6))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues')

if __name__ == "__main__":
    main()
	print('test')