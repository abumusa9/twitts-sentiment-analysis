"""
Sentiment Analysis Model using BERT-based transformers
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import logging
import os
from typing import Dict, List, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    A BERT-based sentiment analysis model for social media text classification.
    Supports multiple pre-trained models optimized for social media content.
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model from Hugging Face Hub
                            Default: cardiffnlp/twitter-roberta-base-sentiment-latest
                            Alternatives: 
                            - "vinai/bertweet-base" (BERTweet)
                            - "nlptown/bert-base-multilingual-uncased-sentiment"
                            - "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self._load_model()
        
        # Sentiment labels mapping
        self.label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive'
        }
        
        # Performance metrics tracking
        self.processed_count = 0
        self.accuracy_scores = []
        
    def _load_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Create pipeline for easier inference
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic preprocessing
        text = text.strip()
        
        # Handle mentions and hashtags (keep them as they provide context)
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (BERT has token limits)
        if len(text) > 512:
            text = text[:512]
            
        return text
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Sentiment analysis results with scores and predicted label
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {
                    'text': text,
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33},
                    'error': 'Empty or invalid text'
                }
            
            # Get predictions
            results = self.sentiment_pipeline(processed_text)
            
            # Process results
            scores = {}
            max_score = 0
            predicted_sentiment = 'neutral'
            
            for result in results[0]:  # results[0] contains all scores
                label = result['label']
                score = result['score']
                
                # Map label to sentiment
                if label in self.label_mapping:
                    sentiment = self.label_mapping[label]
                else:
                    # Handle different model label formats
                    sentiment = label.lower()
                
                scores[sentiment] = round(score, 4)
                
                if score > max_score:
                    max_score = score
                    predicted_sentiment = sentiment
            
            # Update processing count
            self.processed_count += 1
            
            return {
                'text': text,
                'sentiment': predicted_sentiment,
                'confidence': round(max_score, 4),
                'scores': scores,
                'processed_text': processed_text,
                'model_used': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'text': text,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33},
                'error': str(e)
            }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            batch_size (int): Batch size for processing
            
        Returns:
            List[Dict]: List of sentiment analysis results
        """
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Preprocess batch
            processed_batch = [self.preprocess_text(text) for text in batch]
            
            try:
                # Get predictions for batch
                batch_results = self.sentiment_pipeline(processed_batch)
                
                # Process results
                for j, (original_text, result_set) in enumerate(zip(batch, batch_results)):
                    scores = {}
                    max_score = 0
                    predicted_sentiment = 'neutral'
                    
                    for result in result_set:
                        label = result['label']
                        score = result['score']
                        
                        # Map label to sentiment
                        if label in self.label_mapping:
                            sentiment = self.label_mapping[label]
                        else:
                            sentiment = label.lower()
                        
                        scores[sentiment] = round(score, 4)
                        
                        if score > max_score:
                            max_score = score
                            predicted_sentiment = sentiment
                    
                    results.append({
                        'text': original_text,
                        'sentiment': predicted_sentiment,
                        'confidence': round(max_score, 4),
                        'scores': scores,
                        'processed_text': processed_batch[j],
                        'model_used': self.model_name
                    })
                
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                # Add error results for failed batch
                for text in batch:
                    results.append({
                        'text': text,
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33},
                        'error': str(e)
                    })
        
        self.processed_count += len(texts)
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model.
        
        Returns:
            Dict: Model information and statistics
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'processed_count': self.processed_count,
            'label_mapping': self.label_mapping,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }
    
    def calculate_accuracy_metrics(self, predictions: List[Dict], ground_truth: List[str]) -> Dict:
        """
        Calculate accuracy metrics for model evaluation.
        
        Args:
            predictions (List[Dict]): Model predictions
            ground_truth (List[str]): True sentiment labels
            
        Returns:
            Dict: Accuracy metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        correct = 0
        total = len(predictions)
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        correct_by_sentiment = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for pred, true_label in zip(predictions, ground_truth):
            predicted_sentiment = pred['sentiment']
            
            if predicted_sentiment == true_label:
                correct += 1
                correct_by_sentiment[true_label] += 1
            
            sentiment_counts[true_label] += 1
        
        # Calculate overall accuracy
        overall_accuracy = correct / total if total > 0 else 0
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for sentiment in sentiment_counts:
            if sentiment_counts[sentiment] > 0:
                per_class_accuracy[sentiment] = correct_by_sentiment[sentiment] / sentiment_counts[sentiment]
            else:
                per_class_accuracy[sentiment] = 0
        
        metrics = {
            'overall_accuracy': round(overall_accuracy, 4),
            'per_class_accuracy': per_class_accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'sentiment_distribution': sentiment_counts
        }
        
        # Store accuracy for tracking
        self.accuracy_scores.append(overall_accuracy)
        
        return metrics


# Factory function for creating different sentiment analyzers
def create_sentiment_analyzer(model_type: str = "twitter-roberta") -> SentimentAnalyzer:
    """
    Factory function to create sentiment analyzers with different models.
    
    Args:
        model_type (str): Type of model to use
                         Options: "twitter-roberta", "bertweet", "multilingual-bert"
    
    Returns:
        SentimentAnalyzer: Configured sentiment analyzer
    """
    model_configs = {
        "twitter-roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "bertweet": "vinai/bertweet-base",
        "multilingual-bert": "nlptown/bert-base-multilingual-uncased-sentiment",
        "twitter-xlm-roberta": "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    }
    
    if model_type not in model_configs:
        logger.warning(f"Unknown model type: {model_type}. Using default twitter-roberta.")
        model_type = "twitter-roberta"
    
    model_name = model_configs[model_type]
    logger.info(f"Creating sentiment analyzer with model: {model_name}")
    
    return SentimentAnalyzer(model_name=model_name)


# Example usage and testing
if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = create_sentiment_analyzer("twitter-roberta")
    
    # Test samples
    test_texts = [
        "I love this new product! It's amazing!",
        "This is terrible, worst experience ever.",
        "It's okay, nothing special but not bad either.",
        "Just had the best day ever! 😊",
        "Feeling really disappointed with the service today.",
        "@user thanks for the great support! #happy",
        "Can't believe how good this movie was! 🎬✨"
    ]
    
    # Single text analysis
    print("=== Single Text Analysis ===")
    for text in test_texts[:3]:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']})")
        print(f"Scores: {result['scores']}")
        print("-" * 50)
    
    # Batch analysis
    print("\n=== Batch Analysis ===")
    batch_results = analyzer.analyze_batch(test_texts)
    for result in batch_results:
        print(f"{result['sentiment'].upper()}: {result['text']} (conf: {result['confidence']})")
    
    # Model info
    print("\n=== Model Information ===")
    info = analyzer.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")

