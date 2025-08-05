import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load and prepare data
def load_airline_data(filepath='airline_sentiment.csv'):
    """Load airline sentiment dataset"""
    if os.path.exists(filepath):
        print(f"Loading airline sentiment data from {filepath}...")
        
        # Try multiple encodings to ensure cross-platform compatibility
        encodings = ['utf-8', 'windows-1252', 'iso-8859-1', 'cp1252', 'latin1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"Successfully loaded file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            # If all encodings fail, try with error handling
            print("Warning: Standard encodings failed, loading with error replacement")
            df = pd.read_csv(filepath, encoding='utf-8', errors='replace')
            print("Warning: Some characters may have been replaced due to encoding issues")
        
        # Check if it has the required columns
        if 'text' in df.columns and 'airline_sentiment' in df.columns:
            # Filter out neutral tweets for binary classification
            # You can modify this to keep all three classes if needed
            df = df[df['airline_sentiment'].isin(['positive', 'negative'])]
            print(f"Loaded {len(df)} tweets (excluding neutral)")
            return df
        else:
            raise ValueError("Dataset missing required columns: 'text' and 'airline_sentiment'")
    else:
        print(f"Error: {filepath} not found!")
        print("Please ensure your airline sentiment CSV file is in the current directory")
        raise FileNotFoundError(filepath)

# Step 2: Preprocess text
def preprocess_text(text):
    """Preprocess tweet text"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove @mentions
    text = ' '.join(word for word in text.split() if not word.startswith('@'))
    
    # Remove URLs
    text = ' '.join(word for word in text.split() if not word.startswith('http'))
    
    # Remove special characters but keep spaces
    text = text.replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ')
    text = text.replace('&amp;', ' and ').replace('&gt;', ' ').replace('&lt;', ' ')
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

# Step 3: Train the model
def train_sentiment_model():
    # Load data
    print("Loading data...")
    df = load_airline_data()
    
    # Basic data exploration
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSentiment distribution:")
    print(df['airline_sentiment'].value_counts())
    
    # Show some example tweets
    print("\nExample tweets:")
    for sentiment in ['positive', 'negative']:
        print(f"\n{sentiment.upper()} examples:")
        examples = df[df['airline_sentiment'] == sentiment]['text'].head(2)
        for i, text in enumerate(examples):
            print(f"{i+1}. {text}")
    
    # Preprocess text
    print("\nPreprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts after preprocessing
    df = df[df['clean_text'].str.len() > 0]
    print(f"Samples after removing empty texts: {len(df)}")
    
    # Prepare features and labels
    X = df['clean_text']
    y = df['airline_sentiment'].map({'positive': 1, 'negative': 0})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create TF-IDF vectorizer
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Use top 5000 words
        ngram_range=(1, 2),  # Use unigrams and bigrams
        stop_words='english',
        max_df=0.95,  # Ignore terms that appear in >95% of documents
        min_df=2  # Ignore terms that appear in <2 documents
    )
    
    # Transform text to features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train logistic regression model
    print("\nTraining model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        verbose=1
    )
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.3f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['negative', 'positive']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model artifacts
    print("\nSaving model artifacts...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, 'models/sentiment_model_v1.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer_v1.pkl')
    
    # Save model metadata
    metadata = {
        'model_version': 'v1',
        'accuracy': float(accuracy),
        'features': int(X_train_tfidf.shape[1]),
        'training_samples': int(len(X_train)),
        'model_type': 'LogisticRegression',
        'vectorizer_type': 'TfidfVectorizer'
    }
    
    import json
    with open('models/model_metadata_v1.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nModel training complete!")
    print(f"Model saved to: models/sentiment_model_v1.pkl")
    print(f"Vectorizer saved to: models/tfidf_vectorizer_v1.pkl")
    print(f"Metadata saved to: models/model_metadata_v1.json")
    
    return model, vectorizer, accuracy

# Step 4: Test the model with examples
def test_model(model, vectorizer):
    print("\n" + "="*50)
    print("Testing model with sample tweets:")
    print("="*50)
    
    test_tweets = [
        "Just had an amazing flight! The crew was fantastic and so helpful.",
        "Worst airline experience ever. Lost my luggage and terrible customer service.",
        "The flight was okay, nothing special but got me there on time.",
        "Love flying with you guys! Always a great experience.",
        "Delayed for 3 hours with no explanation. Never flying with them again.",
        "Thank you for the smooth flight and excellent service!",
        "Cancelled my flight last minute. Absolutely unacceptable.",
        "Best airline food I've ever had! Pleasant surprise."
    ]
    
    for tweet in test_tweets:
        # Preprocess
        clean_tweet = preprocess_text(tweet)
        # Transform
        tweet_tfidf = vectorizer.transform([clean_tweet])
        # Predict
        prediction = model.predict(tweet_tfidf)[0]
        probability = model.predict_proba(tweet_tfidf)[0]
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        confidence = max(probability) * 100
        
        print(f"\nTweet: {tweet}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.1f}%)")

# Step 5: Feature importance analysis
def analyze_features(model, vectorizer, top_n=20):
    print("\n" + "="*50)
    print(f"Top {top_n} Features for Each Class:")
    print("="*50)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Sort features by importance
    feature_importance = sorted(zip(coefficients, feature_names), reverse=True)
    
    # Top positive features (indicating positive sentiment)
    print(f"\nTop {top_n} Positive Indicators (words associated with positive tweets):")
    for coef, feature in feature_importance[:top_n]:
        print(f"  {feature}: {coef:.3f}")
    
    # Top negative features (indicating negative sentiment)
    print(f"\nTop {top_n} Negative Indicators (words associated with negative tweets):")
    for coef, feature in feature_importance[-top_n:]:
        print(f"  {feature}: {coef:.3f}")
    
    # Show some bigrams if they exist
    bigrams = [f for f in feature_names if ' ' in f]
    if bigrams:
        print(f"\nInteresting bigrams found:")
        bigram_importance = [(coef, feat) for coef, feat in feature_importance if ' ' in feat]
        # Show top positive bigrams
        print("Positive bigrams:")
        for coef, feature in bigram_importance[:5]:
            if coef > 0:
                print(f"  {feature}: {coef:.3f}")
        # Show top negative bigrams  
        print("Negative bigrams:")
        for coef, feature in bigram_importance[-5:]:
            if coef < 0:
                print(f"  {feature}: {coef:.3f}")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("Airline Sentiment Analysis Model Training")
    print("="*60)
    print("\nThis script expects a CSV file named 'airline_sentiment.csv'")
    print("with columns: 'text' (tweet text) and 'airline_sentiment' (positive/negative/neutral)")
    print("="*60)
    
    try:
        # Train the model
        model, vectorizer, accuracy = train_sentiment_model()
        
        # Test with examples
        test_model(model, vectorizer)
        
        # Analyze important features
        analyze_features(model, vectorizer)
        
        # Final summary
        print("\n" + "="*50)
        print("Model Training Summary:")
        print("="*50)
        print(f"✓ Model trained with {accuracy:.1%} accuracy")
        print(f"✓ Artifacts saved in 'models/' directory")
        print(f"✓ Ready for deployment!")
        print("\nNote: This model is trained on airline tweets.")
        print("For best results, use it with similar customer service tweets.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure 'airline_sentiment.csv' is in the current directory.")
        print("Your CSV should have columns: 'text' and 'airline_sentiment'")