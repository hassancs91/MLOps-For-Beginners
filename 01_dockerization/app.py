#!/usr/bin/env python3
"""
FastAPI Server for Airline Sentiment Analysis

This is a production-ready API server for the sentiment analysis model.
It provides REST endpoints for predictions and monitoring.

To run locally:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Then visit: http://localhost:8000/docs for interactive API documentation

Author: MLOps Course
Version: 1.0
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store model artifacts
model = None
vectorizer = None
metadata = {}


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events.
    Code before yield runs on startup.
    Code after yield runs on shutdown.
    """
    # Startup
    logger.info("Starting up API server...")
    load_model_artifacts()
    logger.info("API server ready!")
    yield
    # Shutdown
    logger.info("Shutting down API server...")


# Create FastAPI app instance with lifespan
app = FastAPI(
    title="Airline Sentiment Analysis API",
    description="API for analyzing sentiment in airline-related tweets",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """
    Request model for sentiment prediction.
    
    Attributes:
        text: The tweet text to analyze
    """
    text: str = Field(
        ..., 
        description="Tweet text to analyze for sentiment",
        example="Just had an amazing flight! The crew was fantastic!"
    )


class PredictionResponse(BaseModel):
    """
    Response model for sentiment prediction.
    
    Attributes:
        text: Original input text
        sentiment: Predicted sentiment (positive/negative)
        confidence: Confidence score as percentage
        model_version: Version of the model used
    """
    text: str
    sentiment: str
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    timestamp: str
    model_loaded: bool
    model_version: Optional[str]


class ModelInfoResponse(BaseModel):
    """Response model for model information endpoint"""
    model_version: str
    model_type: str
    accuracy: float
    training_samples: int
    features: int
    vectorizer_type: str


def load_model_artifacts():
    """
    Load model artifacts at startup.
    This function is called once when the server starts.
    """
    global model, vectorizer, metadata
    
    model_path = 'models/sentiment_model_v1.pkl'
    vectorizer_path = 'models/tfidf_vectorizer_v1.pkl'
    metadata_path = 'models/model_metadata_v1.json'
    
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        
        # Load model and vectorizer
        logger.info("Loading model artifacts...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Model loaded successfully! Version: {metadata.get('model_version', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise


def preprocess_text(text):
    """
    Preprocess tweet text - must match the preprocessing in training.
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned text ready for vectorization
    """
    if pd.isna(text) or not text:
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


# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        Status of the API and model loading state
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None,
        model_version=metadata.get("model_version") if model else None
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model metadata including version, accuracy, and configuration
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_version=metadata.get("model_version", "unknown"),
        model_type=metadata.get("model_type", "unknown"),
        accuracy=metadata.get("accuracy", 0.0),
        training_samples=metadata.get("training_samples", 0),
        features=metadata.get("features", 0),
        vectorizer_type=metadata.get("vectorizer_type", "unknown")
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for a given tweet text.
    
    Args:
        request: PredictionRequest containing the text to analyze
        
    Returns:
        PredictionResponse with sentiment and confidence
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Check if model is loaded
    if not model or not vectorizer:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Preprocess the text
        clean_text = preprocess_text(request.text)
        
        # Handle empty text after preprocessing
        if not clean_text:
            return PredictionResponse(
                text=request.text,
                sentiment="neutral",
                confidence=0.0,
                model_version=metadata.get("model_version", "unknown")
            )
        
        # Transform text to features
        text_features = vectorizer.transform([clean_text])
        
        # Get prediction and probabilities
        prediction = model.predict(text_features)[0]
        probabilities = model.predict_proba(text_features)[0]
        
        # Convert to sentiment label
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = max(probabilities) * 100
        
        # Log prediction for monitoring
        logger.info(f"Prediction: {sentiment} ({confidence:.1f}%) for text: {request.text[:50]}...")
        
        return PredictionResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence,
            model_version=metadata.get("model_version", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    # This allows running the script directly for development
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)