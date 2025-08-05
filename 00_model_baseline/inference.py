#!/usr/bin/env python3
"""
Local Inference Script for Airline Sentiment Analysis

This script allows you to test the trained sentiment analysis model locally.
You can use it in two ways:
1. Command line: python inference.py "Your tweet text here"
2. Interactive: python inference.py (then enter text when prompted)

Author: MLOps Course
Version: 1.0
"""

import sys
import os
import joblib
import pandas as pd
import json


def load_model_artifacts():
    """
    Load the trained model, vectorizer, and metadata from the models directory.
    
    Returns:
        tuple: (model, vectorizer, metadata) or raises error if not found
    """
    # Define paths to model artifacts
    model_path = 'models/sentiment_model_v1.pkl'
    vectorizer_path = 'models/tfidf_vectorizer_v1.pkl'
    metadata_path = 'models/model_metadata_v1.json'
    
    # Check if all files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run train.py first to create the model."
        )
    
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Vectorizer not found at {vectorizer_path}. "
            "Please run train.py first to create the model."
        )
    
    # Load model and vectorizer
    print("Loading model artifacts...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Load metadata if available
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    print(f"Model loaded successfully! (Version: {metadata.get('model_version', 'unknown')})")
    print(f"Model accuracy on test set: {metadata.get('accuracy', 'N/A'):.2%}\n")
    
    return model, vectorizer, metadata


def preprocess_text(text):
    """
    Preprocess tweet text - same function as used in training.
    This ensures consistency between training and inference.
    
    Args:
        text (str): Raw tweet text
        
    Returns:
        str: Cleaned text ready for vectorization
    """
    if pd.isna(text) or not text:
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove @mentions (e.g., @airline)
    text = ' '.join(word for word in text.split() if not word.startswith('@'))
    
    # Remove URLs
    text = ' '.join(word for word in text.split() if not word.startswith('http'))
    
    # Remove special characters but keep spaces
    text = text.replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ')
    text = text.replace('&amp;', ' and ').replace('&gt;', ' ').replace('&lt;', ' ')
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text


def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for a given text.
    
    Args:
        text (str): Tweet text to analyze
        model: Trained scikit-learn model
        vectorizer: Trained TF-IDF vectorizer
        
    Returns:
        tuple: (sentiment, confidence) where sentiment is 'positive' or 'negative'
               and confidence is a percentage
    """
    # Preprocess the text
    clean_text = preprocess_text(text)
    
    if not clean_text:
        return "neutral", 0.0
    
    # Transform text to TF-IDF features
    text_features = vectorizer.transform([clean_text])
    
    # Get prediction and probability
    prediction = model.predict(text_features)[0]
    probabilities = model.predict_proba(text_features)[0]
    
    # Convert prediction to sentiment label
    sentiment = "positive" if prediction == 1 else "negative"
    
    # Get confidence (probability of predicted class)
    confidence = max(probabilities) * 100
    
    return sentiment, confidence


def main():
    """
    Main function to handle command line arguments and run inference.
    """
    try:
        # Load model artifacts once at startup
        model, vectorizer, metadata = load_model_artifacts()
        
        # Check if text was provided via command line
        if len(sys.argv) > 1:
            # Join all arguments in case the text wasn't quoted
            tweet_text = ' '.join(sys.argv[1:])
        else:
            # Interactive mode - ask for input
            print("No text provided. Entering interactive mode.")
            print("Type 'quit' or 'exit' to stop.\n")
            
            while True:
                tweet_text = input("Enter tweet text: ").strip()
                
                if tweet_text.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if not tweet_text:
                    print("Please enter some text.\n")
                    continue
                
                # Predict sentiment
                sentiment, confidence = predict_sentiment(tweet_text, model, vectorizer)
                
                # Display results
                print(f"\nOriginal text: {tweet_text}")
                print(f"Preprocessed: {preprocess_text(tweet_text)}")
                print(f"Sentiment: {sentiment.upper()}")
                print(f"Confidence: {confidence:.1f}%")
                print("-" * 50 + "\n")
            
            return
        
        # Single prediction mode
        print(f"Analyzing: {tweet_text}\n")
        
        # Predict sentiment
        sentiment, confidence = predict_sentiment(tweet_text, model, vectorizer)
        
        # Display results
        print(f"Original text: {tweet_text}")
        print(f"Preprocessed: {preprocess_text(tweet_text)}")
        print(f"Sentiment: {sentiment.upper()}")
        print(f"Confidence: {confidence:.1f}%")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()