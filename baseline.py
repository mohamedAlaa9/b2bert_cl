import torch
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback,
)
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
# from preprocessing import final_eliminations
from transformers import TrainingArguments, EarlyStoppingCallback
class BertTrainer:
    def __init__(self, training_dataset_path, labels, exp_num, stage = 0, threshold=0.5, model_name="CAMeL-Lab/bert-base-arabic-camelbert-ca"):
        self.labels = labels
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.model_name = model_name
        self.exp_num = exp_num
        training_dataset = pd.read_csv(training_dataset_path)
        self.training_dataset_processed = pd.DataFrame({
            'text': training_dataset['tweet'],
            'label': training_dataset[self.labels].values.tolist()
        })
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.train_df, self.val_df = train_test_split(self.training_dataset_processed, test_size=0.1, random_state=42)
        self.train_df['text'] = self.train_df['text'].astype(str)
        self.val_df['text'] = self.val_df['text'].astype(str)
        self.train_dataset = self.create_dataset(self.train_df)
        self.val_dataset = self.create_dataset(self.val_df)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(dropout_rate=0.3)  # Adding dropout rate
        self.stage = stage

    def create_dataset(self, df):
        encodings = self.tokenizer(
            df['text'].tolist(), truncation=True, padding=True, max_length=128
        )
        return TweetDataset(encodings, df['label'].values)

    def load_model(self, dropout_rate=0.3):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
            problem_type="multi_label_classification"
        )
        print(f"layers equal {len(model.bert.encoder.layer)}")
        model.config.hidden_dropout_prob = dropout_rate
        model.config.attention_probs_dropout_prob = dropout_rate
        
        for param in model.bert.encoder.layer[:8].parameters():
            param.requires_grad = False

        model.to(self.device)
        return model

    
    def predict(self, texts):
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=128, 
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()
        predictions = (probabilities >= self.threshold).astype(int)
        variation_score = 1 - (np.sum(probabilities)/18)
        return predictions, probabilities, variation_score

    
    def evaluate(self, dev_path, preprocess_flag=True):
        if '.tsv' in dev_path:
            dev = pd.read_csv(dev_path, sep='\t')
        else:
            dev = pd.read_csv(dev_path)

        df_replaced = dev.replace({'y': 1, 'n': 0})
        country_columns = df_replaced.columns.difference(['sentence'])
        df_replaced['label'] = df_replaced[country_columns].values.tolist()
        df_final = df_replaced[['sentence', 'label']]
        
        predictions, probabilities, _ = self.predict(df_final['sentence'].tolist())
        output_dir = f'./exp_{self.exp_num}'
        output_file = os.path.join(self.save_dir, f"{self.model_name.replace('/', '-')}-experiment-{self.exp_num}_predictions.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                pred_str = ','.join(map(str, pred))
                f.write(f'{pred_str}\n')

        
        indexes = [0, 2, 4, 10, 13, 14, 15, 17]
        predictions = [output[indexes] for output in predictions]


        subset_accuracy = accuracy_score(df_final['label'].tolist(), predictions)
        print(f"Subset Accuracy: {subset_accuracy:.4f}")

        hamming = hamming_loss(df_final['label'].tolist(), predictions)
        print(f"Hamming Loss: {hamming:.4f}")

        precision, recall, f1, _ = precision_recall_fscore_support(
            df_final['label'].tolist(), predictions, average='micro'  
        )
        print(f"Micro Precision: {precision:.4f}")
        print(f"Micro Recall: {recall:.4f}")
        print(f"Micro F1-Score: {f1:.4f}")

        precision_per_label, recall_per_label, f1_per_label, _ = precision_recall_fscore_support(
            df_final['label'].tolist(), predictions, average=None 
        )
        print(f"Precision per label: {precision_per_label}")
        print(f"Recall per label: {recall_per_label}")
        print(f"F1-Score per label: {f1_per_label}")
        multilabel_check = [np.sum(np.array(prediction)) for prediction in predictions]
        print(set(multilabel_check))


    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = self.multi_label_metrics(preds, p.label_ids)
        return result

    def multi_label_metrics(self, predictions, labels):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= self.threshold)] = 1
        f1 = f1_score(labels, y_pred, average='micro')
        roc_auc = roc_auc_score(labels, y_pred, average='micro')
        accuracy = accuracy_score(labels, y_pred)
        return {'f1': f1, 'roc_auc': roc_auc, 'accuracy': accuracy}
    
    def train(
        self,
        num_train_epochs=3,  
        metric_for_best_model="eval_f1",  
        greater_is_better=True,  
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        patience=2,
        warmup_steps=500,  
        base_learning_rate=5e-5,  
    ):
        training_args = TrainingArguments(
            output_dir='./exp_' + str(self.exp_num) + '/results',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            learning_rate=base_learning_rate,
            weight_decay=0.01,
            logging_dir='./exp_' + str(self.exp_num) + '/logs',
            logging_steps=500,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            fp16=True,
            report_to=["tensorboard"],
            lr_scheduler_type="linear",  
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=patience
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping_callback]  
        )

        trainer.train()

        
        best_metric_value = trainer.state.best_metric 
        num_epochs = training_args.num_train_epochs
        greater_is_better = training_args.greater_is_better
        metric_name = training_args.metric_for_best_model
        save_dir = f'./exp_{self.exp_num}/stage_{self.stage}'
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir, safe_serialization=False)
        self.tokenizer.save_pretrained(save_dir)

        


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, **kwargs):
        if output_dir is None:
            output_dir = self.args.output_dir
        for param in self.model.parameters():
            param.data = param.data.contiguous()
        super().save_model(output_dir, **kwargs)
dev_path = "./NADI2024/subtask1/dev/NADI2024_subtask1_dev2.tsv"
dataset_path = "./aggregated_final/stage_1_and_gpt.csv"
labels = ['Algeria', 'Bahrain', 'Egypt', 'Iraq', 'Jordan', 'Kuwait',
       'Lebanon', 'Libya', 'Morocco', 'Oman', 'Palestine', 'Qatar',
       'Saudi_Arabia', 'Sudan', 'Syria', 'Tunisia', 'UAE', 'Yemen']
trainer = BertTrainer(
    training_dataset_path=dataset_path,
    labels=labels,
    threshold=0.3,
    exp_num=4,
    model_name="UBC-NLP/MARBERT"
)
trainer.train(
    num_train_epochs=3,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24
)
trainer.evaluate(dev_path=dev_path, preprocess_flag = False)
import os

scorer_script = "./NADI2024/subtask1/NADI2024-ST1-Scorer.py"
gold_file = "./NADI2024/subtask1/sample_submission/NADI2024_subtask1_dev2_gold.txt"
predictions_file = "./exp_4/stage_0/UBC-NLP-MARBERT-experiment-4_predictions.txt"