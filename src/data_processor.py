"""
╔════════════════════════════════════════════════════════════════════════════╗
║                       PROCESADOR DE DATOS DE EVENTOS                      ║
║                                                                            ║
║  Módulo responsable de:                                                   ║
║  • Cargar datos desde archivos CSV/XLSX                                   ║
║  • Limpiar y normalizar los datos                                         ║
║  • Detectar automáticamente columnas relevantes                           ║
║  • Aplicar filtros por provincia, año, mes y tipo de evento               ║
║  • Generar predicciones futuras basadas en patrones históricos            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
from typing import Tuple, Dict, Any


class DataProcessor:
    """
    Procesador integral de datos de eventos naturales.
    
    Responsabilidades:
    - Cargar datos desde múltiples formatos
    - Limpiar, validar y normalizar datos
    - Detectar automáticamente columnas por contexto
    - Filtrar datos según criterios del usuario
    - Generar eventos futuros mediante extrapolación
    """
    
    def __init__(self):
        self.df_original = None
        self.df_processed = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def load_data(self, uploaded_file=None, file_path: str = None) -> pd.DataFrame:
        """
        Carga datos desde un archivo subido o ruta local.
        
        Soporta múltiples formatos:
        - CSV (valores separados por coma)
        - XLSX (hojas de cálculo Excel)
        
        Args:
            uploaded_file: Objeto de archivo subido a través de Streamlit, o ruta como string
            file_path: Ruta local al archivo (alternativa a uploaded_file)
            
        Returns:
            DataFrame con los datos cargados, o None si hay error
            
        Raises:
            Exception: Si hay problemas al leer el archivo
        """
        try:
            if uploaded_file is not None:
                # Handle both file objects (Streamlit) and string paths
                if isinstance(uploaded_file, str):
                    if uploaded_file.endswith('.csv'):
                        self.df_original = pd.read_csv(uploaded_file)
                    elif uploaded_file.endswith('.xlsx'):
                        self.df_original = pd.read_excel(uploaded_file)
                else:
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
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Limpia exhaustivamente el dataset para que sea apto para modelado.
        
        Operaciones realizadas:
        1. Eliminación de registros duplicados
        2. Remoción del año 2023 (datos incompletos, serán predichos)
        3. Eliminación de columnas con > 50% valores nulos
        4. Remoción de columnas de identificación/códigos (no predictivas)
        5. Relleno inteligente de valores faltantes:
           - Columnas numéricas: mediana
           - Columnas categóricas: moda
        
        Args:
            df: DataFrame sin procesar
            
        Returns:
            Tupla con:
            - DataFrame limpio listo para modelado
            - Diccionario con estadísticas del proceso de limpieza
        """
        stats = {
            "registros_iniciales": len(df),
            "duplicados_eliminados": len(df[df.duplicated()]),
            "registros_con_nulos": df.isnull().any(axis=1).sum(),
            "columnas_eliminadas": [],
            "registros_2023_eliminados": 0
        }
        
        # Eliminar duplicados
        df_clean = df.drop_duplicates()
        
        for col in df_clean.select_dtypes(include=['object']).columns:
            if 'evento' in col.lower() or 'tipo' in col.lower() or 'peligro' in col.lower():
                df_clean[col] = (
                    df_clean[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.title()
                )
        
        # Detectar y eliminar año 2023 (muy pocos datos, será tratado como futuro)
        year_col = None
        for col in df_clean.columns:
            if 'año' in col.lower() or 'year' in col.lower():
                year_col = col
                break
        
        if year_col and year_col in df_clean.columns:
            if np.issubdtype(df_clean[year_col].dtype, np.number):
                years_data = df_clean[year_col].astype(int)
            else:
                dates = pd.to_datetime(df_clean[year_col], errors='coerce')
                years_data = dates.dt.year
            
            # Contar registros de 2023
            count_2023 = (years_data == 2023).sum()
            stats["registros_2023_eliminados"] = count_2023
            
            # Eliminar 2023
            if count_2023 > 0:
                df_clean = df_clean[years_data != 2023].reset_index(drop=True)
        
        # Eliminar columnas con más del 50% de valores nulos
        initial_cols = len(df_clean.columns)
        threshold = len(df_clean) * 0.5
        cols_to_drop = [col for col in df_clean.columns if df_clean[col].isnull().sum() > threshold]
        
        # También eliminar columnas de ID/Índice
        id_patterns = ['id', 'identificador', 'codigo', 'codificacion']
        cols_to_drop += [col for col in df_clean.columns if any(pat in col.lower() for pat in id_patterns) and col not in cols_to_drop]
        
        if cols_to_drop:
            stats["columnas_eliminadas"] = cols_to_drop
            df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')
        
        # Llenar valores nulos estratégicamente
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Llenar numéricos con mediana
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Llenar categóricos con moda
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Desconocido'
                df_clean[col].fillna(mode_val, inplace=True)
        
        stats["registros_finales"] = len(df_clean)
        stats["registros_descartados"] = stats["registros_iniciales"] - stats["registros_finales"]
        stats["columnas_iniciales"] = initial_cols
        stats["columnas_finales"] = len(df_clean.columns)
        
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
    
    def filter_data(self, df: pd.DataFrame, filters: Dict) -> Tuple[pd.DataFrame, bool]:
        """
        Aplica filtros al dataframe de manera inteligente.
        
        Lógica de filtrado:
        - Si el año seleccionado EXISTE en datos históricos: retorna datos reales de ese año
        - Si el año NO existe (futuro): GENERA eventos predichos basados en patrones históricos
        
        Filtros aplicables:
        - 'year': Año específico (int). Genera futuros si no existe.
        - 'month': Mes específico (1-12). Opcional.
        - 'provincia': Lista de provincias. Opcional.
        - 'event_types': Lista de tipos de evento. Opcional.
        
        Detección automática de columnas:
        La función identifica automáticamente las columnas de provincia, año, mes y evento
        usando keywords, eliminando la necesidad de especificar nombres exactos.
        
        Args:
            df: DataFrame a filtrar (debe estar limpio)
            filters: Diccionario con criterios de filtrado
        
        Returns:
            Tupla con:
            - DataFrame filtrado
            - Boolean indicando si es una predicción futura (True) o histórica (False)
        """
        df_filtered = df.copy()
        is_future_prediction = False
        
        # Detectar columnas automáticamente - ser más específico
        provincia_col = None
        year_col = None
        month_col = None
        event_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            # Detectar provincia (preferir la columna exacta PROVINCIA)
            if col == 'PROVINCIA':
                provincia_col = col
            elif provincia_col is None and 'provincia' in col_lower and 'codif' not in col_lower and 'informe' not in col_lower:
                provincia_col = col
            
            # Detectar año (preferir AÑO exacto)
            if col == 'AÑO':
                year_col = col
            elif year_col is None and ('año' in col_lower or 'year' in col_lower) and 'fecha' not in col_lower:
                year_col = col
            elif year_col is None and 'fecha' in col_lower:
                year_col = col

            # Detectar mes
            if month_col is None and 'mes' in col_lower:
                month_col = col
            
            # Detectar evento
            if 'evento' in col_lower and 'categoría' not in col_lower:
                event_col = col
            elif event_col is None and ('tipo' in col_lower or 'peligro' in col_lower):
                event_col = col
        
        # Filtro por AÑO específico - PRIMERO, antes de otros filtros
        if filters.get('year') is not None and year_col:
            selected_year = int(filters['year'])
            year_series = df_filtered[year_col]
            
            if np.issubdtype(year_series.dtype, np.number):
                years_in_data = year_series.astype(int)
            else:
                dates = pd.to_datetime(year_series, errors='coerce')
                years_in_data = dates.dt.year
            
            # Verificar si el año existe en los datos
            if selected_year in years_in_data.values:
                # Año existe: usar datos reales de ese año específico
                df_filtered = df_filtered[years_in_data == selected_year]
            else:
                # Año NO existe: Generar eventos futuros basados en patrones históricos
                is_future_prediction = True
                # Generar eventos futuros
                df_filtered = self.generate_future_events(df_filtered, selected_year)
        
        # DESPUÉS de generar eventos futuros, aplicar otros filtros
        # Aplicar filtros de provincia
        if filters.get('provincia') and provincia_col and provincia_col in df_filtered.columns:
            df_filtered[provincia_col] = df_filtered[provincia_col].astype(str).str.strip()
            provincias_normalized = [p.strip().upper() for p in filters['provincia']]
            df_filtered = df_filtered[df_filtered[provincia_col].str.upper().isin(provincias_normalized)]

        # Aplicar filtro de mes si existe
        if filters.get('month') is not None:
            month_value = int(filters['month'])
            if month_col and month_col in df_filtered.columns:
                month_series = df_filtered[month_col]

                if np.issubdtype(month_series.dtype, np.number):
                    month_numeric = pd.to_numeric(month_series, errors='coerce')
                    df_filtered = df_filtered[month_numeric == month_value]
                else:
                    # Intentar como fecha primero
                    dates = pd.to_datetime(month_series, errors='coerce')
                    if dates.notna().sum() > 0:
                        df_filtered = df_filtered[dates.dt.month == month_value]
                    else:
                        # Si no es fecha, intentar como texto del mes
                        month_map = {
                            1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril', 5: 'mayo', 6: 'junio',
                            7: 'julio', 8: 'agosto', 9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
                        }
                        month_str_lower = month_series.astype(str).str.lower().str.strip()
                        df_filtered = df_filtered[month_str_lower == month_map.get(month_value, '')]

        # Compatibilidad: Filtro por RANGO de años (si se usa)
        if filters.get('year_range') and year_col:
            year_min, year_max = filters['year_range']
            year_series = df_filtered[year_col]

            if np.issubdtype(year_series.dtype, np.number):
                df_filtered = df_filtered[(year_series >= year_min) & (year_series <= year_max)]
            else:
                dates = pd.to_datetime(year_series, errors='coerce')
                df_filtered = df_filtered[
                    (dates.dt.year >= year_min) &
                    (dates.dt.year <= year_max)
                ]
        
        # Aplicar filtro de tipo de evento
        if filters.get('event_types'):
            if event_col and event_col in df_filtered.columns:
                # Normalizar para comparación case-insensitive
                event_series = df_filtered[event_col].astype(str).str.strip()
                event_types_normalized = [str(et).strip() for et in filters['event_types']]
                
                # Crear máscara para filtrado (case-insensitive)
                mask = event_series.str.upper().isin([et.upper() for et in event_types_normalized])
                df_filtered = df_filtered[mask]
        
        return df_filtered, is_future_prediction
    
    def generate_future_events(self, df: pd.DataFrame, future_year: int, num_events: int = None) -> pd.DataFrame:
        """
        Genera eventos futuros mediante extrapolación de patrones históricos.
        
        Algoritmo:
        1. Identifica todas las combinaciones provincia-evento observadas históricamente
        2. Para cada evento futuro:
           - Selecciona una combinación válida aleatoriamente
           - Muestrea el mes según distribución histórica de estacionalidad
           - Interpola latitud/longitud dentro del rango observado
           - Extrapola características numéricas usando regresión lineal de tendencias
           - Añade ruido pequeño para variabilidad
        3. Retorna DataFrame con misma estructura que datos históricos
        
        Args:
            df: DataFrame histórico (años 2010-2023)
            future_year: Año para el cual generar predicciones (ej: 2024, 2025, 2026)
            num_events: Cantidad de eventos a generar (default: promedio histórico anual)
        
        Returns:
            DataFrame con eventos predichos, con mismas columnas que df original
            
        Raises:
            ValueError: Si faltan columnas esenciales (provincia, evento, año)
        """
        np.random.seed(future_year)  # Reproducibilidad por año
        
        # Detectar columnas clave
        provincia_col = self._find_column(df.columns, ['provincia'])
        evento_col = self._find_column(df.columns, ['evento', 'tipo'])
        # Buscar AÑO específicamente
        year_col = 'AÑO' if 'AÑO' in df.columns else self._find_column(df.columns, ['año', 'year', 'fecha'])
        mes_col = self._find_column(df.columns, ['mes'])
        lat_col = self._find_column(df.columns, ['latitud', 'lat'])
        lon_col = self._find_column(df.columns, ['longitud', 'lon'])
        
        if not all([provincia_col, evento_col, year_col]):
            raise ValueError("Faltan columnas necesarias para generar eventos futuros")
        
        # Calcular número de eventos si no se especifica
        if num_events is None:
            num_events = int(len(df) / df[year_col].nunique())
        
        # Obtener combinaciones válidas provincia-evento
        valid_combinations = df[[provincia_col, evento_col]].drop_duplicates().reset_index(drop=True)
        
        # Extraer patrones históricos
        future_events = []
        
        for _ in range(num_events):
            # Seleccionar combinación provincia-evento aleatoria (solo las que existen)
            combo = valid_combinations.sample(1).iloc[0]
            prov = combo[provincia_col]
            evt = combo[evento_col]
            
            # Filtrar datos para esta combinación
            combo_data = df[(df[provincia_col] == prov) & (df[evento_col] == evt)].copy()
            
            # Crear evento futuro
            event_dict = {
                provincia_col: prov,
                evento_col: evt,
                year_col: future_year
            }
            
            # MES: Por estacionalidad
            if mes_col and mes_col in combo_data.columns:
                month_dist = combo_data[mes_col].value_counts()
                if len(month_dist) > 0:
                    event_dict[mes_col] = np.random.choice(month_dist.index, p=month_dist.values/month_dist.sum())
            else:
                event_dict[mes_col] = np.random.randint(1, 13) if mes_col else None
            
            # LATITUD/LONGITUD: Dentro del rango histórico
            if lat_col and lon_col and lat_col in combo_data.columns and lon_col in combo_data.columns:
                lat_range = combo_data[lat_col].quantile([0.1, 0.9])
                lon_range = combo_data[lon_col].quantile([0.1, 0.9])
                event_dict[lat_col] = np.random.uniform(lat_range.iloc[0], lat_range.iloc[1])
                event_dict[lon_col] = np.random.uniform(lon_range.iloc[0], lon_range.iloc[1])
            
            # CARACTERÍSTICAS NUMÉRICAS: Continuar tendencia histórica
            numeric_cols = combo_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # Excluir columnas de identificación, año, mes, ubicación
                if col in [year_col, mes_col, lat_col, lon_col, 'ID', 'CODIFICACION PROVINCIAL', 
                          'CODIFICACION CANTONAL', 'CODIFICACION PARROQUIAL']:
                    continue
                    
                # Calcular tendencia
                yearly_data = combo_data.groupby(year_col)[col].mean().sort_index()
                
                if len(yearly_data) >= 2:
                    years_array = yearly_data.index.astype(float).values
                    values_array = yearly_data.values
                    
                    # Regresión lineal
                    z = np.polyfit(years_array, values_array, 1)
                    slope = z[0]
                    
                    # Predicción
                    last_year = yearly_data.index.max()
                    last_value = yearly_data.iloc[-1]
                    years_ahead = future_year - last_year
                    predicted_value = last_value + (slope * years_ahead)
                    
                    # Variación pequeña
                    variation = np.random.normal(0, 0.05 * max(predicted_value, 1))
                    event_dict[col] = max(predicted_value + variation, 0)
                else:
                    event_dict[col] = combo_data[col].mean()
            
            future_events.append(event_dict)
        
        # Crear DataFrame con eventos futuros
        df_future = pd.DataFrame(future_events)
        
        # Asegurar que tiene las mismas columnas que el original
        for col in df.columns:
            if col not in df_future.columns:
                if df[col].dtype in [np.int64, np.float64]:
                    df_future[col] = 0
                else:
                    df_future[col] = 'N/A'
        
        return df_future[df.columns]
    
    @staticmethod
    def _find_column(columns, keywords: list) -> str:
        """Encuentra una columna que contenga alguno de los keywords."""
        for col in columns:
            if any(keyword.lower() in col.lower() for keyword in keywords):
                return col
        return None
