import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import joblib
from typing import Optional
from pathlib import Path

# Initialize FastAPI
app = FastAPI(title="Salary Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'ds_salaries.csv'
MODEL_DIR = BASE_DIR / 'model'
MODEL_DIR.mkdir(exist_ok=True)

# Define the request body format
class PredictionFeatures(BaseModel):
    experience_level: str  # EN, MI, SE, EX
    company_size: str     # S, M, L
    employment_type: str  # FT, PT
    job_title: str       # Data Engineer, Data Manager, Data Scientist, Machine Learning Engineer

# Define response model
class PredictionResponse(BaseModel):
    salary_usd: float
    confidence: float

# Global variables for encoders and model
model = None
exp_encoder = None
size_encoder = None

def clean_dataset(df):
    """Clean and preprocess the dataset"""
    # Select only the columns we need
    columns_to_use = ['experience_level', 'employment_type', 'job_title', 'salary_in_usd', 'company_size']
    df = df[columns_to_use]
    
    # Remove any rows with missing values
    df = df.dropna()
    
    # Ensure salary is numeric
    df['salary_in_usd'] = pd.to_numeric(df['salary_in_usd'], errors='coerce')
    df = df.dropna()  # Remove any rows where salary couldn't be converted to numeric
    
    print(f"Dataset shape after cleaning: {df.shape}")
    print("Column types:", df.dtypes)
    
    return df

def load_or_train_model():
    """Load existing model or train a new one if none exists"""
    global model, exp_encoder, size_encoder
    
    try:
        model = joblib.load(MODEL_DIR / 'lin_regress.sav')
        exp_encoder = joblib.load(MODEL_DIR / 'exp_encoder.sav')
        size_encoder = joblib.load(MODEL_DIR / 'size_encoder.sav')
        print("Loaded existing model and encoders")
    except:
        print("Training new model...")
        try:
            # Load and clean the dataset
            if not DATA_PATH.exists():
                raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
            
            salary_data = pd.read_csv(DATA_PATH)
            print(f"Loaded raw dataset with shape: {salary_data.shape}")
            
            # Clean the dataset
            salary_data = clean_dataset(salary_data)
            
            # Initialize encoders
            exp_encoder = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
            size_encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])
            
            # Transform the data
            salary_data['experience_level_encoded'] = exp_encoder.fit_transform(salary_data[['experience_level']])
            salary_data['company_size_encoded'] = size_encoder.fit_transform(salary_data[['company_size']])
            
            # Create dummy variables
            salary_data = pd.get_dummies(
                salary_data,
                columns=['employment_type', 'job_title'],
                drop_first=True
            )
            
            # Prepare features for training
            feature_columns = [col for col in salary_data.columns 
                             if col not in ['salary_in_usd', 'experience_level', 'company_size']]
            
            print("Feature columns:", feature_columns)
            
            X = salary_data[feature_columns]
            y = salary_data['salary_in_usd']
            
            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = linear_model.LinearRegression()
            model.fit(X_train, y_train)
            
            # Calculate R-squared score
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"Model R-squared score (train): {train_score:.3f}")
            print(f"Model R-squared score (test): {test_score:.3f}")
            
            # Save model and encoders
            joblib.dump(model, MODEL_DIR / 'lin_regress.sav')
            joblib.dump(exp_encoder, MODEL_DIR / 'exp_encoder.sav')
            joblib.dump(size_encoder, MODEL_DIR / 'size_encoder.sav')
            
            print("Model trained and saved")
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise

# Initialize model when starting
load_or_train_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Salary Prediction API",
        "version": "1.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_path_exists": DATA_PATH.exists()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PredictionFeatures):
    """Make salary prediction based on input features"""
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([{
            'experience_level': features.experience_level,
            'company_size': features.company_size,
            'employment_type': features.employment_type,
            'job_title': features.job_title
        }])
        
        # Transform features
        input_df['experience_level_encoded'] = exp_encoder.transform(
            input_df[['experience_level']]
        )
        input_df['company_size_encoded'] = size_encoder.transform(
            input_df[['company_size']]
        )
        
        # Create dummy variables
        input_df = pd.get_dummies(
            input_df,
            columns=['employment_type', 'job_title'],
            drop_first=True
        )
        
        # Ensure all necessary columns exist
        feature_columns = model.feature_names_in_
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Reorder columns to match training data
        input_df = input_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Calculate confidence (simplified for demo)
        confidence = 0.85  # In a real model, this would be calculated based on model metrics
        
        return PredictionResponse(
            salary_usd=float(prediction),
            confidence=confidence
        )
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
