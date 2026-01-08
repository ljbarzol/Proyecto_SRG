"""
Módulo de modelos de Machine Learning y Deep Learning.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class MLModels:
    """Clase que encapsula todos los modelos de ML/DL."""
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = MinMaxScaler()
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> None:
        """
        Divide datos en train/test.
        
        Args:
            X: Features
            y: Target
            test_size: Proporción de test
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
    
    def train_xgboost(self, params: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Entrena modelo XGBoost.
        
        Args:
            params: Hiperparámetros personalizados
            
        Returns:
            Diccionario con métricas de evaluación
        """
        default_params = {
            'n_estimators': 150,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'binary:logistic' if len(np.unique(self.y_train)) == 2 else 'multi:softmax',
            'num_class': len(np.unique(self.y_train)) if len(np.unique(self.y_train)) > 2 else None
        }
        
        if params:
            default_params.update(params)
        
        if default_params['num_class'] is None:
            del default_params['num_class']
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(self.X_train, self.y_train, verbose=False)
        
        y_pred = model.predict(self.X_test)
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        self.models['xgboost'] = model
        self.metrics['xgboost'] = metrics
        
        return metrics
    
    def train_random_forest(self, params: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Entrena modelo Random Forest.
        
        Args:
            params: Hiperparámetros personalizados
            
        Returns:
            Diccionario con métricas de evaluación
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        model = RandomForestClassifier(**default_params)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = metrics
        
        return metrics
    
    def train_lstm(self, lookback: int = 10, epochs: int = 50, batch_size: int = 32) -> Dict[str, float]:
        """
        Entrena modelo LSTM para series temporales.
        
        Args:
            lookback: Ventana de observación
            epochs: Número de épocas
            batch_size: Tamaño del batch
            
        Returns:
            Diccionario con métricas de evaluación
        """
        try:
            # Preparar datos para LSTM
            X_scaled = self.scaler.fit_transform(self.X_train)
            
            # Crear secuencias
            X_seq_train = self._create_sequences(X_scaled, lookback)
            y_train_seq = self.y_train[lookback:]
            
            if len(X_seq_train) == 0:
                raise ValueError("No hay suficientes datos para crear secuencias LSTM")
            
            # Construir modelo LSTM
            model = Sequential([
                LSTM(64, activation='relu', input_shape=(lookback, X_scaled.shape[1]), return_sequences=True),
                Dropout(0.2),
                LSTM(32, activation='relu', return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(len(np.unique(self.y_train)), activation='softmax')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Entrenar
            model.fit(X_seq_train, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Evaluar
            X_scaled_test = self.scaler.transform(self.X_test)
            X_seq_test = self._create_sequences(X_scaled_test, lookback)
            y_test_seq = self.y_test[lookback:]
            
            if len(X_seq_test) > 0:
                y_pred = np.argmax(model.predict(X_seq_test), axis=1)
                metrics = self._calculate_metrics(y_test_seq, y_pred)
            else:
                metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            
            self.models['lstm'] = model
            self.metrics['lstm'] = metrics
            
            return metrics
        except Exception as e:
            return {'error': str(e), 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    def train_gradient_boosting(self, params: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Entrena modelo con Gradient Boosting (segundo modelo XGBoost con diferentes parámetros).
        
        Args:
            params: Hiperparámetros personalizados
            
        Returns:
            Diccionario con métricas de evaluación
        """
        default_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': 42,
            'objective': 'binary:logistic' if len(np.unique(self.y_train)) == 2 else 'multi:softmax',
        }
        
        if params:
            default_params.update(params)
        
        if 'num_class' in default_params and default_params['num_class'] is None:
            del default_params['num_class']
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(self.X_train, self.y_train, verbose=False)
        
        y_pred = model.predict(self.X_test)
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        self.models['gradient_boosting'] = model
        self.metrics['gradient_boosting'] = metrics
        
        return metrics
    
    def _create_sequences(self, data: np.ndarray, lookback: int) -> np.ndarray:
        """Crea secuencias para LSTM."""
        X_seq = []
        for i in range(len(data) - lookback):
            X_seq.append(data[i:i + lookback])
        return np.array(X_seq)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de evaluación."""
        try:
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            }
        except Exception as e:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'error': str(e)}
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Retorna métricas de todos los modelos entrenados."""
        return self.metrics
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con un modelo específico.
        
        Args:
            model_name: Nombre del modelo
            X: Features
            
        Returns:
            Predicciones
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no ha sido entrenado")
        
        model = self.models[model_name]
        if model_name == 'lstm':
            X_scaled = self.scaler.transform(X)
            X_seq = self._create_sequences(X_scaled, 10)
            if len(X_seq) > 0:
                return np.argmax(model.predict(X_seq), axis=1)
            return np.array([])
        else:
            return model.predict(X)
    
    def save_model(self, model_name: str, filepath: str) -> bool:
        """Guarda modelo a disco."""
        try:
            if model_name not in self.models:
                return False
            joblib.dump(self.models[model_name], filepath)
            return True
        except Exception as e:
            print(f"Error al guardar modelo: {e}")
            return False
    
    def load_model(self, model_name: str, filepath: str) -> bool:
        """Carga modelo desde disco."""
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            return True
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            return False
