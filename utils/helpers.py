"""
Utilidades generales del sistema.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Any


def generate_sample_dataset(n_records: int = 500) -> pd.DataFrame:
    """
    Genera un dataset de muestra para pruebas.
    
    Args:
        n_records: Número de registros a generar
        
    Returns:
        DataFrame con datos de muestra
    """
    np.random.seed(42)
    
    provincias = ['Pichincha', 'Guayas', 'Azuay', 'Tungurahua', 'Imbabura', 
                  'Carchi', 'Cotopaxi', 'Manabí', 'Los Ríos', 'Loja']
    tipos_evento = ['Inundación', 'Deslizamiento', 'Incendio', 'Erupción Volcánica', 'Terremoto']
    
    data = {
        'Fecha': pd.date_range('2015-01-01', periods=n_records, freq='D'),
        'Provincia': np.random.choice(provincias, n_records),
        'Cantón': [f"Cantón_{i%10}" for i in range(n_records)],
        'Parroquia': [f"Parroquia_{i%20}" for i in range(n_records)],
        'Tipo_Evento': np.random.choice(tipos_evento, n_records),
        'Personas_Afectadas': np.random.randint(10, 1000, n_records),
        'Viviendas_Dañadas': np.random.randint(5, 500, n_records),
        'Infraestructura_Dañada': np.random.randint(0, 100, n_records),
        'Latitude': np.random.uniform(-5, 2, n_records),
        'Longitude': np.random.uniform(-81, -75, n_records),
        'Severidad': np.random.choice(['Baja', 'Media', 'Alta'], n_records, p=[0.3, 0.5, 0.2]),
    }
    
    return pd.DataFrame(data)


def export_metrics_to_json(metrics: Dict[str, Dict], filepath: str) -> bool:
    """
    Exporta métricas a archivo JSON.
    
    Args:
        metrics: Diccionario con métricas
        filepath: Ruta del archivo
        
    Returns:
        True si se exportó exitosamente
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        return True
    except Exception as e:
        print(f"Error al exportar métricas: {e}")
        return False


def calculate_vulnerability_score(affected_people: int, damaged_houses: int, 
                                 event_frequency: int, max_values: Dict) -> float:
    """
    Calcula un score de vulnerabilidad normalizado (0-1).
    
    Args:
        affected_people: Número de personas afectadas
        damaged_houses: Número de viviendas dañadas
        event_frequency: Frecuencia de eventos
        max_values: Diccionario con valores máximos para normalización
        
    Returns:
        Score normalizado entre 0 y 1
    """
    normalized_people = affected_people / max(max_values.get('max_people', 1), 1)
    normalized_houses = damaged_houses / max(max_values.get('max_houses', 1), 1)
    normalized_frequency = event_frequency / max(max_values.get('max_frequency', 1), 1)
    
    # Promedio ponderado
    score = (normalized_people * 0.4 + normalized_houses * 0.4 + normalized_frequency * 0.2)
    
    return min(score, 1.0)


def get_risk_level(score: float) -> str:
    """
    Obtiene nivel de riesgo textual basado en score.
    
    Args:
        score: Score normalizado (0-1)
        
    Returns:
        Texto del nivel de riesgo
    """
    if score >= 0.7:
        return "Alto"
    elif score >= 0.4:
        return "Medio"
    else:
        return "Bajo"


def format_number(number: int, prefix: str = "") -> str:
    """
    Formatea número con separadores de miles.
    
    Args:
        number: Número a formatear
        prefix: Prefijo (ej: $, €)
        
    Returns:
        Número formateado
    """
    return f"{prefix}{number:,}"


def get_color_for_risk(risk_level: str) -> str:
    """
    Retorna color HTML para nivel de riesgo.
    
    Args:
        risk_level: Nivel de riesgo (Alto, Medio, Bajo)
        
    Returns:
        Código color HTML
    """
    colors = {
        'Alto': '#FF6B6B',
        'Medio': '#FFA500',
        'Bajo': '#51CF66'
    }
    return colors.get(risk_level, '#888888')


class PerformanceMetrics:
    """Clase para calcular y almacenar métricas de desempeño."""
    
    def __init__(self):
        self.metrics = {}
        self.training_time = 0
        self.prediction_time = 0
    
    def add_metric(self, name: str, value: Any) -> None:
        """Agrega una métrica."""
        self.metrics[name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen de métricas."""
        return {
            'total_metrics': len(self.metrics),
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }


class DataValidator:
    """Validador de datos."""
    
    @staticmethod
    def validate_csv_structure(df: pd.DataFrame, required_cols: List[str] = None) -> Dict[str, Any]:
        """
        Valida estructura de un CSV.
        
        Args:
            df: DataFrame a validar
            required_cols: Columnas requeridas
            
        Returns:
            Diccionario con resultados de validación
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validar columnas requeridas
        if required_cols:
            missing = set(required_cols) - set(df.columns)
            if missing:
                results['is_valid'] = False
                results['errors'].append(f"Columnas faltantes: {missing}")
        
        # Validar valores nulos
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            results['warnings'].append(f"Detectados {null_counts.sum()} valores nulos")
        
        # Validar duplicados
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            results['warnings'].append(f"Detectados {duplicates} registros duplicados")
        
        return results
