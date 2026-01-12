"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      GESTOR DE MODELOS DE ML/DL                           ║
║                                                                            ║
║  Módulo responsable de:                                                   ║
║  • Entrenar múltiples modelos (XGBoost, Random Forest)                    ║
║  • Evaluar desempeño con métricas estándar                                ║
║  • Generar predicciones sobre nuevos datos                                ║
║  • Calcular matrices de confusión y visualizaciones                       ║
║                                                                            ║
║  Modelos disponibles:                                                     ║
║  - XGBoost: Gradient boosting extremadamente eficiente                    ║
║  - Random Forest: Ensamble robusto con interpretabilidad                  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import joblib
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class MLModels:
    """
    Gestor centralizado de modelos de Machine Learning.
    
    Funcionalidades:
    - Preparación y división de datos (train/test)
    - Entrenamiento de múltiples algoritmos
    - Evaluación con métricas estándar
    - Predicción en datos nuevos
    - Cálculo de matrices de confusión
    """
    
    def __init__(self):
        """
        Inicializa el gestor de modelos con estructuras vacías.
        
        Atributos:
            models (dict): Almacena modelos entrenados por nombre
            metrics (dict): Almacena métricas de evaluación de cada modelo
            X_train, X_test: Datos de entrenamiento y prueba
            y_train, y_test: Etiquetas de entrenamiento y prueba
            last_preds, last_true: Predicciones y valores reales para matriz de confusión
        """

        self.models = {}
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.last_preds = None
        self.last_true = None
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> None:
        """
        Prepara los datos dividiendo en conjunto de entrenamiento y prueba.
        
        Estrategia:
        - Usa stratified split para mantener distribución de clases
        - 80% datos de entrenamiento (entrenar modelos)
        - 20% datos de prueba (evaluar desempeño)
        - Random state = 42 para reproducibilidad
        
        Args:
            X: Array de características (features) [n_samples, n_features]
            y: Array de etiquetas (target) [n_samples,]
            test_size: Proporción de datos para test (default: 0.2 = 20%)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
    
    def train_xgboost(self, params: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Entrena modelo XGBoost (Extreme Gradient Boosting).
        
        Características:
        - Algoritmo de boosting extremadamente eficiente
        - Manejo automático de multiclase (>2 clases)
        - Regularización incorporada (L1/L2)
        - Rápido y escalable
        
        Parámetros ajustables:
        - n_estimators: Número de árboles (default: 150)
        - max_depth: Profundidad máxima de árbol (default: 8)
        - learning_rate: Tasa de aprendizaje (default: 0.1)
        - subsample: Fracción de muestras por árbol (default: 0.8)
        - colsample_bytree: Fracción de features por árbol (default: 0.8)
        
        Args:
            params: Diccionario con hiperparámetros personalizados
            
        Returns:
            Diccionario con métricas de evaluación:
            - accuracy: Precisión general
            - precision: Precisión ponderada
            - recall: Sensibilidad ponderada
            - f1: F1-score ponderado
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

        self.last_preds = y_pred
        self.last_true = self.y_test
        
        self.models['xgboost'] = model
        self.metrics['xgboost'] = metrics
        
        return metrics

    def train_random_forest(self, params: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Entrena modelo Random Forest (Bosque Aleatorio).
        
        Características:
        - Ensamble de árboles de decisión independientes
        - Muy robusto ante overfitting
        - Proporciona importancia de características
        - Fácil de interpretar y paralelizable
        
        Parámetros ajustables:
        - n_estimators: Número de árboles (default: 100)
        - max_depth: Profundidad máxima (default: 10)
        - min_samples_split: Mínimo de muestras para dividir (default: 5)
        - min_samples_leaf: Mínimo de muestras en hoja (default: 2)
        - n_jobs: Procesadores en paralelo (default: -1 = todos)
        
        Args:
            params: Diccionario con hiperparámetros personalizados
            
        Returns:
            Diccionario con métricas de evaluación:
            - accuracy: Precisión general
            - precision: Precisión ponderada
            - recall: Sensibilidad ponderada
            - f1: F1-score ponderado
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

        self.last_preds = y_pred
        self.last_true = self.y_test
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = metrics
        
        return metrics
    
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

        self.last_preds = y_pred
        self.last_true = self.y_test
        
        self.models['gradient_boosting'] = model
        self.metrics['gradient_boosting'] = metrics
        
        return metrics

    def get_confusion_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve la matriz de confusión y las etiquetas usadas."""
        if self.last_true is None or self.last_preds is None:
            return None, None
        labels = np.unique(np.concatenate([self.last_true, self.last_preds]))
        cm = confusion_matrix(self.last_true, self.last_preds, labels=labels)
        return cm, labels
    
    def _create_sequences(self, data: np.ndarray, lookback: int) -> np.ndarray:
        """Crea secuencias para LSTM."""
        X_seq = []
        for i in range(len(data) - lookback):
            X_seq.append(data[i:i + lookback])
        return np.array(X_seq)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de evaluación estándar para clasificación.
        
        Métricas calculadas:
        - Accuracy: Porcentaje de predicciones correctas
        - Precision: De los predichos positivos, cuántos eran correctos
        - Recall: De los positivos reales, cuántos se detectaron
        - F1-score: Media armónica de precision y recall
        
        Se usa promediado ponderado para manejar clases desbalanceadas.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Etiquetas predichas
            
        Returns:
            Diccionario con las 4 métricas principales
        """
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
        Realiza predicciones usando un modelo específico.
        
        El modelo debe haber sido entrenado previamente con train_xgboost() o
        train_random_forest().
        
        Args:
            model_name: Nombre del modelo ('xgboost' o 'random_forest')
            X: Array de características [n_samples, n_features]
            
        Returns:
            Array de predicciones (etiquetas de clases) [n_samples,]
            
        Raises:
            KeyError: Si el modelo especificado no existe o no fue entrenado
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no ha sido entrenado")
        
        model = self.models[model_name]
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
