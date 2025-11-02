"""
BERT Trainer Module for Multi-label Arabic Dialect Classification
Implements curriculum learning with configurable stages
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, 
    roc_auc_score, 
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
)


class TweetDataset(torch.utils.data.Dataset):
    """Dataset class for tweet data with multi-label classification."""
    
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
    """Custom trainer to ensure model parameters are contiguous before saving."""
    
    def save_model(self, output_dir=None, **kwargs):
        if output_dir is None:
            output_dir = self.args.output_dir
        for param in self.model.parameters():
            param.data = param.data.contiguous()
        super().save_model(output_dir, **kwargs)


class BertTrainer:
    """
    Main trainer class for BERT-based multi-label Arabic dialect classification.
    Supports curriculum learning through staged training.
    """
    
    # Class-level constants
    DIALECT_LABELS = [
        'Algeria', 'Bahrain', 'Egypt', 'Iraq', 'Jordan', 'Kuwait',
        'Lebanon', 'Libya', 'Morocco', 'Oman', 'Palestine', 'Qatar',
        'Saudi_Arabia', 'Sudan', 'Syria', 'Tunisia', 'UAE', 'Yemen'
    ]
    
    EVAL_SUBSET_INDEXES = [0, 2, 4, 10, 13, 14, 15, 17]
    
    def __init__(
        self, 
        training_dataset_path, 
        labels, 
        exp_num, 
        stage=0, 
        threshold=0.5, 
        model_name="CAMeL-Lab/bert-base-arabic-camelbert-ca"
    ):
        """
        Initialize the BertTrainer.
        
        Args:
            training_dataset_path: Path to training CSV file
            labels: List of dialect labels
            exp_num: Experiment number for tracking
            stage: Current curriculum learning stage
            threshold: Classification threshold for predictions
            model_name: Pretrained model name or path
        """
        self.labels = labels
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.model_name = model_name
        self.exp_num = exp_num
        self.stage = stage
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = None
        
        # Load and prepare data
        self.training_dataset_processed = self._load_training_data(training_dataset_path)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Split and prepare datasets
        self._prepare_datasets()
        
        # Load model
        self.model = self._load_model(dropout_rate=0.3)
    
    def _load_training_data(self, training_dataset_path):
        """Load and process training dataset."""
        training_dataset = pd.read_csv(training_dataset_path)
        return pd.DataFrame({
            'text': training_dataset['tweet'],
            'label': training_dataset[self.labels].values.tolist()
        })
    
    def _prepare_datasets(self):
        """Split data into train/val and create datasets."""
        self.train_df, self.val_df = train_test_split(
            self.training_dataset_processed, 
            test_size=0.1, 
            random_state=42
        )
        self.train_df['text'] = self.train_df['text'].astype(str)
        self.val_df['text'] = self.val_df['text'].astype(str)
        
        self.train_dataset = self._create_dataset(self.train_df)
        self.val_dataset = self._create_dataset(self.val_df)
    
    def _create_dataset(self, df):
        """Create a TweetDataset from a dataframe."""
        encodings = self.tokenizer(
            df['text'].tolist(), 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        return TweetDataset(encodings, df['label'].values)
    
    def _load_model(self, dropout_rate=0.3):
        """Load and configure the BERT model."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
            problem_type="multi_label_classification"
        )
        
        # Configure dropout
        model.config.hidden_dropout_prob = dropout_rate
        model.config.attention_probs_dropout_prob = dropout_rate
        
        # Freeze lower layers
        for param in model.bert.encoder.layer[:8].parameters():
            param.requires_grad = False

        model.to(self.device)
        return model
    
    def predict(self, texts):
        """
        Generate predictions for input texts.
        
        Args:
            texts: List of text strings to predict
            
        Returns:
            predictions: Binary predictions
            probabilities: Prediction probabilities
            variation_score: Computed variation score
        """
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
        variation_score = 1 - (np.sum(probabilities) / 18)
        
        return predictions, probabilities, variation_score
    
    def evaluate(self, dev_path):
        """
        Evaluate model on development set.
        
        Args:
            dev_path: Path to development dataset
        """
        # Load development data
        if '.tsv' in dev_path:
            dev = pd.read_csv(dev_path, sep='\t')
        else:
            dev = pd.read_csv(dev_path)
        
        from preprocess import final_eliminations
        dev = final_eliminations(dev, column_name="sentence")
        
        # Prepare labels
        df_replaced = dev.replace({'y': 1, 'n': 0})
        country_columns = df_replaced.columns.difference(['sentence'])
        df_replaced['label'] = df_replaced[country_columns].values.tolist()
        df_final = df_replaced[['sentence', 'label']]
        
        # Generate predictions
        predictions, probabilities, _ = self.predict(df_final['sentence'].tolist())
        
        # Save predictions to file
        self._save_predictions(predictions)
        
        # Apply subset indexing for evaluation
        predictions = [output[self.EVAL_SUBSET_INDEXES] for output in predictions]
        
        # Print metrics
        self._print_evaluation_metrics(df_final['label'].tolist(), predictions)
    
    def _save_predictions(self, predictions):
        """Save predictions to output file."""
        output_dir = f'./exp_{self.exp_num}'
        output_file = output_dir + "/output.txt"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for pred in predictions:
                pred_str = ','.join(map(str, pred))
                f.write(f'{pred_str}\n')
    
    def _print_evaluation_metrics(self, true_labels, predictions):
        """Print comprehensive evaluation metrics."""
        # Subset accuracy
        subset_accuracy = accuracy_score(true_labels, predictions)
        print(f"Subset Accuracy: {subset_accuracy:.4f}")
        
        # Hamming loss
        hamming = hamming_loss(true_labels, predictions)
        print(f"Hamming Loss: {hamming:.4f}")
        
        # Micro-averaged metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='micro'
        )
        print(f"Micro Precision: {precision:.4f}")
        print(f"Micro Recall: {recall:.4f}")
        print(f"Micro F1-Score: {f1:.4f}")
        
        # Per-label metrics
        precision_per_label, recall_per_label, f1_per_label, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        print(f"Precision per label: {precision_per_label}")
        print(f"Recall per label: {recall_per_label}")
        print(f"F1-Score per label: {f1_per_label}")
        
        # Multi-label check
        multilabel_check = [np.sum(np.array(prediction)) for prediction in predictions]
        print(set(multilabel_check))
    
    def compute_metrics(self, p: EvalPrediction):
        """Compute metrics for Trainer evaluation."""
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = self._multi_label_metrics(preds, p.label_ids)
        return result
    
    def _multi_label_metrics(self, predictions, labels):
        """Calculate multi-label classification metrics."""
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
        """
        Train the model.
        
        Args:
            num_train_epochs: Number of training epochs
            metric_for_best_model: Metric to use for model selection
            greater_is_better: Whether higher metric values are better
            per_device_train_batch_size: Training batch size
            per_device_eval_batch_size: Evaluation batch size
            patience: Early stopping patience
            warmup_steps: Number of warmup steps
            base_learning_rate: Initial learning rate
        """
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
            eval_strategy="epoch",
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

        # Save the best model
        save_dir = f'./exp_{self.exp_num}/stage_{self.stage}'
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir, safe_serialization=False)
        self.tokenizer.save_pretrained(save_dir)
