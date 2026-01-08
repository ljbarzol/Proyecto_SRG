"""
Módulo para procesamiento y carga de datos de eventos naturales.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
from typing import Tuple, Dict, Any


class DataProcessor:
    """Procesa y limpia datos de eventos naturales."""
    
    def __init__(self):
        self.df_original = None
        self.df_processed = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def load_data(self, uploaded_file=None, file_path: str = None) -> pd.DataFrame:
        """
        Carga datos desde un archivo subido o ruta local.
        
        Args:
            uploaded_file: Archivo subido por Streamlit
            file_path: Ruta a archivo local
            
        Returns:
            DataFrame con los datos cargados
        """
        try:
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    self.df_original = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    self.df_original = pd.read_excel(uploaded_file)
            elif file_path:
                if file_path.endswith('.csv'):
                    self.df_original = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.df_original = pd.read_excel(file_path)
            
            return self.df_original.copy() if self.df_original is not None else None
        except Exception as e:
            raise Exception(f"Error al cargar datos: {str(e)}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera un resumen del dataset.
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con estadísticas del dataset
        """
        return {
            "total_registros": len(df),
            "columnas": len(df.columns),
            "lista_columnas": list(df.columns),
            "valores_nulos": df.isnull().sum().to_dict(),
            "tipos_datos": df.dtypes.to_dict(),
            "registros_duplicados": len(df[df.duplicated()])
        }
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Limpia el dataset: elimina duplicados y valores nulos.
        
        Args:
            df: DataFrame sin procesar
            
        Returns:
            Tupla con (DataFrame limpio, diccionario de estadísticas)
        """
        stats = {
            "registros_iniciales": len(df),
            "duplicados_eliminados": len(df[df.duplicated()]),
            "registros_con_nulos": df.isnull().any(axis=1).sum()
        }
        
        # Eliminar duplicados
        df_clean = df.drop_duplicates()
        
        # Llenar valores nulos estratégicamente
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Llenar numéricos con mediana
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Llenar categóricos con moda
        for col in categorical_cols:
            df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Desconocido', inplace=True)
        
        stats["registros_finales"] = len(df_clean)
        stats["registros_descartados"] = stats["registros_iniciales"] - stats["registros_finales"]
        
        self.df_processed = df_clean
        return df_clean, stats
    
    def preprocess_for_modeling(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Preprocesa datos para modelado: encoding y escalado.
        
        Args:
            df: DataFrame limpio
            target_col: Columna objetivo
            
        Returns:
            DataFrame procesado
        """
        df_prep = df.copy()
        
        # Identificar columnas categóricas
        categorical_cols = df_prep.select_dtypes(include=['object']).columns.tolist()
        
        # Remover columna objetivo de las categóricas si existe
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # Encoding de variables categóricas
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_prep[col] = self.label_encoders[col].fit_transform(df_prep[col].astype(str))
            else:
                df_prep[col] = self.label_encoders[col].transform(df_prep[col].astype(str))
        
        # Escalado de variables numéricas
        numeric_cols = df_prep.select_dtypes(include=[np.number]).columns
        if target_col and target_col in numeric_cols and target_col is not None:
            numeric_cols = numeric_cols.drop(target_col)
        
        df_prep[numeric_cols] = self.scaler.fit_transform(df_prep[numeric_cols])
        
        return df_prep
    
    def get_event_types(self, df: pd.DataFrame) -> list:
        """Obtiene tipos únicos de eventos."""
        # Buscar columna de tipo de evento (nombres comunes)
        event_cols = [col for col in df.columns if 'evento' in col.lower() or 'tipo' in col.lower() or 'peligro' in col.lower()]
        
        if event_cols:
            return df[event_cols[0]].unique().tolist()
        return []
    
    def get_regions(self, df: pd.DataFrame) -> Tuple[list, Dict]:
        """Obtiene provincias y cantones únicos."""
        region_cols = [col for col in df.columns if any(x in col.lower() for x in ['provincia', 'canton', 'región'])]
        
        regions = {}
        if len(region_cols) > 0:
            provinces = df[region_cols[0]].unique().tolist() if region_cols else []
            regions['provincias'] = sorted([str(p) for p in provinces if pd.notna(p)])
        
        return regions.get('provincias', []), regions
    
    def filter_data(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """
        Aplica filtros al dataframe.
        
        Args:
            df: DataFrame
            filters: Diccionario con filtros a aplicar
                    {
                        'provincia': ['Pichincha'],
                        'year_range': (2015, 2022),
                        'event_types': ['Inundación', 'Deslizamiento']
                    }
        
        Returns:
            DataFrame filtrado
        """
        df_filtered = df.copy()
        
        # Detectar columnas automáticamente
        provincia_col = None
        year_col = None
        event_col = None
        
        for col in df.columns:
            if 'provincia' in col.lower():
                provincia_col = col
            if 'año' in col.lower() or 'year' in col.lower() or 'fecha' in col.lower():
                year_col = col
            if 'evento' in col.lower() or 'tipo' in col.lower() or 'peligro' in col.lower():
                event_col = col
        
        # Aplicar filtros
        if filters.get('provincia') and provincia_col:
            # Normalizar nombres para comparación insensible a mayúsculas/minúsculas
            df_filtered[provincia_col] = df_filtered[provincia_col].astype(str).str.strip()
            provincias_normalized = [p.strip().upper() for p in filters['provincia']]
            df_filtered = df_filtered[df_filtered[provincia_col].str.upper().isin(provincias_normalized)]
        
        if filters.get('year_range') and year_col:
            year_min, year_max = filters['year_range']
            year_series = df_filtered[year_col]

            # Manejar columnas numéricas (años) y de fechas sin provocar comparaciones inválidas
            if np.issubdtype(year_series.dtype, np.number):
                df_filtered = df_filtered[(year_series >= year_min) & (year_series <= year_max)]
            else:
                dates = pd.to_datetime(year_series, errors='coerce')
                df_filtered = df_filtered[
                    (dates.dt.year >= year_min) &
                    (dates.dt.year <= year_max)
                ]
        
        if filters.get('event_types') and event_col:
            df_filtered = df_filtered[df_filtered[event_col].isin(filters['event_types'])]
        
        return df_filtered
