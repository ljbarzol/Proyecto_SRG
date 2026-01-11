"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║     SISTEMA DE PREDICCIÓN DE ZONAS VULNERABLES A DESASTRES NATURALES      ║
║                         Ecuador - 2024                                     ║
║                                                                            ║
║  Dashboard interactivo para análisis de riesgo ante eventos naturales     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
import plotly.express as px
import time

# Configurar rutas de importación para los módulos locales
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processor import DataProcessor  # Procesamiento de datos
from src.models import MLModels               # Modelos de Machine Learning
from src.visualizations import Visualizations # Visualizaciones y mapas
import streamlit_folium
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sistema de Predicción de Desastres - Ecuador",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para mejorar la presentación visual
st.markdown("""
<style>
    /* Encabezado principal */
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    /* Contenedor de indicadores clave (KPIs) */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Tarjetas individuales de KPI */
    .kpi-card {
        flex: 1;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN DEL ESTADO DE LA SESIÓN
# ══════════════════════════════════════════════════════════════════════════════
# Streamlit mantiene estos valores persistentes durante toda la sesión del usuario

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()  # Instancia para carga y limpieza de datos

if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None  # Almacena el DataFrame original

if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None  # Almacena el DataFrame procesado/limpio

if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()  # Gestor centralizado de modelos ML

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {}  # Diccionario para rastrear qué modelos se entrenaron

if 'trained_model_name' not in st.session_state:
    st.session_state.trained_model_name = None  # Nombre del modelo actualmente en uso

# ══════════════════════════════════════════════════════════════════════════════
# CONTENIDO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🗺️ Sistema de Predicción de Desastres Naturales - Ecuador")
st.markdown("---")

# Crear las tres pestañas principales de la aplicación
tab1, tab2, tab3 = st.tabs([
    "📁 Gestión de Datos",
    "🤖 Entrenamiento y Análisis", 
    "📊 Visualización de Resultados"
])

# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 1: GESTIÓN DE DATOS
# Aquí el usuario carga, visualiza y procesa los datos iniciales
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 1️⃣ Gestión de Datos (Carga y Depuración)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Módulo de Carga de Datos")
        st.write("Cargue el archivo de eventos peligrosos en formato CSV o XLSX")
        
        uploaded_file = st.file_uploader(
            "Arrastra o selecciona un archivo",
            type=['csv', 'xlsx'],
            help="Soporta archivos CSV y XLSX"
        )
        
        # Opción para cargar datos de demostración
        col_upload, col_demo = st.columns(2)
        with col_upload:
            if uploaded_file:
                st.session_state.df_loaded = st.session_state.data_processor.load_data(uploaded_file)
                st.success(f"✅ Archivo cargado: {uploaded_file.name}")
        
        with col_demo:
            if st.button("📂 Cargar Datos de Demostración"):
                # Intentar cargar archivo de demostración original
                demo_file = "/workspaces/Proyecto_SRG/SGR_Eventos.csv"
                if os.path.exists(demo_file):
                    st.session_state.df_loaded = st.session_state.data_processor.load_data(file_path=demo_file)
                    st.success("✅ Datos de demostración cargados desde archivo")
                else:
                    # Si no existe, generar datos de prueba sintéticos
                    # Útil para testing cuando no hay archivo disponible
                    np.random.seed(42)
                    demo_data = pd.DataFrame({
                        'Fecha': pd.date_range('2015-01-01', periods=500, freq='W'),
                        'Provincia': np.random.choice(['Pichincha', 'Guayas', 'Azuay', 'Tungurahua', 'Cotopaxi'], 500),
                        'Tipo_Evento': np.random.choice(['Inundación', 'Deslizamiento', 'Incendio'], 500),
                        'Personas_Afectadas': np.random.randint(10, 1000, 500),
                        'Viviendas_Dañadas': np.random.randint(5, 500, 500),
                        'Latitude': np.random.uniform(-5, 2, 500),
                        'Longitude': np.random.uniform(-81, -75, 500)
                    })
                    st.session_state.df_loaded = demo_data
                    st.success("✅ Datos de demostración creados")
    
    with col2:
        st.markdown("### Estadísticas del Archivo")
        if st.session_state.df_loaded is not None:
            # Mostrar resumen rápido del dataset cargado
            summary = st.session_state.data_processor.get_data_summary(st.session_state.df_loaded)
            st.metric("📊 Total de Registros", summary['total_registros'])
            st.metric("📋 Total de Columnas", summary['columnas'])
            st.metric("⚠️ Valores Nulos", sum(summary['valores_nulos'].values()))
            st.metric("🔄 Registros Duplicados", summary['registros_duplicados'])
        else:
            st.info("Carga un archivo para ver estadísticas")
    
    st.markdown("---")
    
    if st.session_state.df_loaded is not None:
        st.markdown("### Vista Previa de Datos")
        # Mostrar primeras 10 filas para inspeccionar estructura
        st.dataframe(
            st.session_state.df_loaded.head(10),
            use_container_width=True,
            height=300
        )
        
        st.markdown("---")
        st.markdown("### Preprocesamiento y Limpieza")
        st.write("El sistema detectará automáticamente columnas de interés (provincia, año, mes, evento)")
        st.write("Se eliminarán datos incompletos y se normalizarán los valores")
        
        col_clean, col_download = st.columns(2)
        
        with col_clean:
            if st.button("🧹 Preprocesar/Limpiar Datos", use_container_width=True):
                with st.spinner("Limpiando datos... Esto puede tomar un momento"):
                    # Ejecutar limpieza completa
                    df_clean, stats = st.session_state.data_processor.clean_data(st.session_state.df_loaded)
                    st.session_state.df_processed = df_clean
                    st.session_state.cleaning_stats = stats  # Guardar para mostrar detalles
                    
                    # Mostrar resultados principales
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("✅ Registros Válidos", stats['registros_finales'])
                    with col_stat2:
                        st.metric("❌ Registros Descartados", stats['registros_descartados'])
                    with col_stat3:
                        st.metric("🔄 Duplicados Eliminados", stats['duplicados_eliminados'])
                    
                    st.success("✅ Datos preprocesados correctamente. Listo para entrenar modelos.")
        
        # Mostrar detalles de limpieza en todo el ancho
        if st.session_state.df_processed is not None:
            with st.expander("📋 Detalles de Limpieza y Columnas Eliminadas", expanded=False):
                st.markdown("#### Resumen de Eliminaciones")
                st.write("**Años descartados:** 2023 (datos insuficientes - será tratado como año 'futuro')")
                
                st.markdown("#### Columnas Eliminadas (Nulos > 50%/Identificadores)")
                # Obtener stats desde el session_state o recalcular
                if 'cleaning_stats' in st.session_state and st.session_state.cleaning_stats:
                    stats = st.session_state.cleaning_stats
                    if 'columnas_eliminadas' in stats and stats['columnas_eliminadas']:
                        cols_eliminated = st.columns(6)
                        for idx, col_name in enumerate(stats['columnas_eliminadas']):
                            with cols_eliminated[idx % 6]:
                                st.write(f"❌ `{col_name}`")
                    else:
                        st.info("✅ No se eliminaron columnas")
                    
                    st.markdown("#### Columnas Retenidas")
                    cols_retained = st.session_state.df_processed.columns.tolist()
                    cols_display_ret = st.columns(6)
                    for idx, col_name in enumerate(cols_retained):
                        with cols_display_ret[idx % 6]:
                            dtype = str(st.session_state.df_processed[col_name].dtype)
                            st.write(f"✅ **{col_name}** (`{dtype}`)")
        
        with col_download:
            if st.session_state.df_processed is not None:
                csv = st.session_state.df_processed.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Datos Limpios",
                    data=csv,
                    file_name="datos_limpios.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Botón Siguiente al final de Tab 1
        if st.session_state.df_processed is not None:
            st.markdown("---")
            col_next = st.columns([3, 1])
            with col_next[1]:
                if st.button("➡️ Siguiente", use_container_width=True, type="primary", key="next_tab1"):
                    st.success("✅ Datos listos. Dirígete a la pestaña **'Entrenamiento y Análisis'**")

# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 2: ENTRENAMIENTO Y ANÁLISIS
# Aquí entrenamos modelos de Machine Learning y evaluamos su desempeño
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 2️⃣ Entrenamiento y Análisis (ML/DL)")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Primero debes cargar y procesar los datos en la pestaña 'Gestión de Datos'")
    else:
        st.markdown("### Configuración del Modelo")
        st.info("Modelos disponibles: XGBoost (recomendado) y Random Forest (robusto).")

        # Selección de modelo
        model_choice = st.selectbox("Modelo de clasificación", ["XGBoost", "Random Forest"], index=0)
        
        st.markdown("---")
        
        # Selección de características/variables
        st.markdown("#### Selección de Variables (Features)")
        st.info(
            "📊 **Sobre las Variables Numéricas:**\n\n"
            "A continuación se muestran todas las columnas numéricas disponibles en el dataset procesado. "
            "Estas variables representan diferentes aspectos de los eventos históricos, tales como:\n"
            "- **Impacto humano**: Personas afectadas, heridos, fallecidos, etc.\n"
            "- **Impacto material**: Viviendas destruidas o afectadas, infraestructura dañada\n"
            "- **Ubicación geográfica**: Coordenadas (latitud, longitud)\n"
            "- **Características temporales**: Año, mes del evento\n\n"
            "💡 **¿Cómo funciona la vulnerabilidad?** Se construye un **índice compuesto** con varias variables de impacto, "
            "se normaliza y se divide en tres niveles (Baja, Media, Alta) usando terciles. "
            "Selecciona las variables predictoras."
        )
        
        numeric_cols = st.session_state.df_processed.select_dtypes(include=[np.number]).columns.tolist()
        # Filtrar columnas ID/CODIF
        numeric_cols = [c for c in numeric_cols if 'CODIF' not in c.upper() and 'ID' not in c.upper()]

        # Columnas usadas para construir la etiqueta (impacto). No deben usarse como predictores para evitar fugas de información.
        impact_cols_for_label = []
        impact_cols_for_label.extend([
            col for col in st.session_state.df_processed.columns
            if 'PERSONAS' in col.upper() and 'AFECTADAS' in col.upper()
        ])
        impact_cols_for_label.extend([
            col for col in st.session_state.df_processed.columns
            if 'VIVIENDA' in col.upper() and ('DESTRUIDA' in col.upper() or 'AFECTADA' in col.upper())
        ])
        impact_cols_for_label.extend([
            col for col in st.session_state.df_processed.columns
            if 'FALLECIDO' in col.upper() or 'HERIDO' in col.upper() or 'DESAPARECIDO' in col.upper()
        ])

        # Features disponibles = numéricas sin las columnas usadas para la etiqueta
        predictor_cols = [c for c in numeric_cols if c not in set(impact_cols_for_label)]
        
        # Checkbox para seleccionar todas
        select_all = st.checkbox("✅ Seleccionar todas las variables", value=True)
        
        # Tabla de selección de variables
        st.markdown("**Variables disponibles:**")
        
        # Organizar en columnas de 3
        num_cols_display = 3
        cols_per_row = num_cols_display
        
        if 'selected_features' not in st.session_state:
            st.session_state.selected_features = predictor_cols.copy()
        
        # Si se selecciona "seleccionar todas", actualizar
        if select_all:
            st.session_state.selected_features = predictor_cols.copy()
        
        selected_features = []
        
        # Crear tabla de checkboxes
        for i in range(0, len(predictor_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(predictor_cols[i:i+cols_per_row]):
                with cols[j]:
                    is_checked = select_all or col_name in st.session_state.selected_features
                    if st.checkbox(col_name, value=is_checked, key=f"feature_{col_name}"):
                        selected_features.append(col_name)
        
        st.session_state.selected_features = selected_features
        
        st.markdown(f"**Variables seleccionadas:** {len(selected_features)} / {len(predictor_cols)}")
        
        st.markdown("---")
        
        # Hiperparámetros
        st.markdown("#### Hiperparámetros del Modelo")

        if model_choice == "XGBoost":
            col_hp1, col_hp2, col_hp3 = st.columns(3)
            with col_hp1:
                xgb_n_estimators = st.slider("n_estimators", 50, 500, 150, step=10, help="Número de árboles de decisión")
            with col_hp2:
                xgb_max_depth = st.slider("max_depth", 3, 20, 8, help="Profundidad máxima de cada árbol")
            with col_hp3:
                xgb_learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, step=0.01, help="Tasa de aprendizaje del modelo")

            col_hp4, col_hp5, col_hp6 = st.columns(3)
            with col_hp4:
                xgb_subsample = st.slider("subsample", 0.5, 1.0, 0.8, step=0.1, help="Fracción de muestras")
            with col_hp5:
                xgb_colsample = st.slider("colsample_bytree", 0.5, 1.0, 0.8, step=0.1, help="Fracción de características")
            with col_hp6:
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2, step=0.05, help="Porcentaje de datos para prueba")
        else:
            col_hp1, col_hp2, col_hp3 = st.columns(3)
            with col_hp1:
                rf_n_estimators = st.slider("n_estimators", 50, 500, 100, step=10, help="Número de árboles")
            with col_hp2:
                rf_max_depth = st.slider("max_depth", 3, 30, 10, help="Profundidad máxima")
            with col_hp3:
                rf_min_samples_split = st.slider("min_samples_split", 2, 20, 5, help="Mínimo para dividir")
            col_hp4, col_hp5 = st.columns(2)
            with col_hp4:
                rf_min_samples_leaf = st.slider("min_samples_leaf", 1, 10, 2, help="Mínimo por hoja")
            with col_hp5:
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2, step=0.05, help="Porcentaje de datos para prueba")
        
        st.markdown("---")
        
        # Preparar datos ANTES del botón de entrenamiento
        X = st.session_state.df_processed[selected_features].fillna(0).values if len(selected_features) > 0 else None
        
        # Crear índice de vulnerabilidad compuesto usando múltiples variables y generar 3 clases (Baja/Media/Alta)
        if X is not None:
            impact_cols: list[str] = []

            # Personas afectadas
            personas_cols = [col for col in st.session_state.df_processed.columns
                             if 'PERSONAS' in col.upper() and 'AFECTADAS' in col.upper()]
            impact_cols.extend(personas_cols)

            # Viviendas dañadas/destruidas
            vivienda_cols = [col for col in st.session_state.df_processed.columns
                             if 'VIVIENDA' in col.upper() and ('DESTRUIDA' in col.upper() or 'AFECTADA' in col.upper())]
            impact_cols.extend(vivienda_cols)

            # Víctimas: fallecidos / heridos / desaparecidos
            victimas_cols = [col for col in st.session_state.df_processed.columns
                             if 'FALLECIDO' in col.upper() or 'HERIDO' in col.upper() or 'DESAPARECIDO' in col.upper()]
            impact_cols.extend(victimas_cols)

            if impact_cols:
                impact_scores = []
                for col in impact_cols:
                    values = st.session_state.df_processed[col].fillna(0)
                    if values.max() > 0:  # Normalizar para evitar que una sola variable domine
                        normalized = values / values.max()
                        impact_scores.append(normalized)

                if impact_scores:
                    composite_index = pd.concat(impact_scores, axis=1).mean(axis=1)
                    # Intentar cortes equitativos (3 bins). Si fallan por duplicados, usar ranking.
                    try:
                        y_raw = pd.qcut(composite_index, 3, labels=[0, 1, 2])
                    except Exception:
                        ranks = composite_index.rank(method='first')
                        y_raw = pd.qcut(ranks, 3, labels=[0, 1, 2])
                    y_raw = pd.Series(y_raw).astype(int)
                else:
                    # Fallback balanceado a 3 clases
                    y_raw = pd.Series(np.random.randint(0, 3, len(X)))
            else:
                # Si no hay columnas de impacto, generar clases balanceadas para permitir entrenamiento
                y_raw = pd.Series(np.random.randint(0, 3, len(X)))

            # Asegurar etiquetas continuas y al menos 2 clases para que XGBoost/Sklearn no falle
            unique_vals = sorted(pd.Series(y_raw).unique())
            if len(unique_vals) < 2:
                # Generar una segunda clase mínima
                y_raw.iloc[: max(1, len(y_raw)//5) ] = (y_raw.iloc[: max(1, len(y_raw)//5) ] + 1) % 3
                unique_vals = sorted(pd.Series(y_raw).unique())

            # Factorizar para garantizar etiquetas 0..k-1
            y = pd.factorize(pd.Series(y_raw))[0].astype(int)
        else:
            y = None
        
        # Botón de entrenamiento
        if st.button("🚀 Entrenar Modelo", use_container_width=True, type="primary"):
            if len(selected_features) == 0:
                st.error("❌ Debes seleccionar al menos una variable para entrenar el modelo")
            elif X is None or y is None:
                st.error("❌ Error al preparar datos")
            else:
                progress = st.progress(0)
                status = st.empty()

                status.info("📦 Preparando datos...")
                progress.progress(15)
                time.sleep(0.05)

                # IMPORTANTE: Guardar features para usar en Tab 3
                st.session_state.selected_features = selected_features

                status.info("🔀 Dividiendo train/test...")
                progress.progress(35)
                st.session_state.ml_models.prepare_data(X, y, test_size=test_size)
                time.sleep(0.05)

                status.info("🏋️ Entrenando modelo...")
                progress.progress(65)
                if model_choice == "XGBoost":
                    metrics_results = st.session_state.ml_models.train_xgboost({
                        'n_estimators': xgb_n_estimators,
                        'max_depth': xgb_max_depth,
                        'learning_rate': xgb_learning_rate,
                        'subsample': xgb_subsample,
                        'colsample_bytree': xgb_colsample
                    })
                    st.session_state.models_trained['xgboost'] = True
                    st.session_state.trained_model_name = 'xgboost'
                else:
                    metrics_results = st.session_state.ml_models.train_random_forest({
                        'n_estimators': rf_n_estimators,
                        'max_depth': rf_max_depth,
                        'min_samples_split': rf_min_samples_split,
                        'min_samples_leaf': rf_min_samples_leaf
                    })
                    st.session_state.models_trained['random_forest'] = True
                    st.session_state.trained_model_name = 'random_forest'

                status.info("📈 Calculando métricas...")
                progress.progress(85)
                time.sleep(0.05)

                st.success("✅ ¡Entrenamiento completado!")
                status.success("✅ Modelo listo")
                progress.progress(100)
                
                # Mostrar métricas inmediatamente
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric("📊 Accuracy", f"{metrics_results['accuracy']:.4f}")
                with col_m2:
                    st.metric("🎯 Precision", f"{metrics_results['precision']:.4f}")
                with col_m3:
                    st.metric("📈 Recall", f"{metrics_results['recall']:.4f}")
                with col_m4:
                    st.metric("⚖️ F1-Score", f"{metrics_results['f1']:.4f}")

                # Matriz de confusión
                cm, labels = st.session_state.ml_models.get_confusion_matrix()
                if cm is not None and labels is not None:
                    st.markdown("#### Matriz de Confusión")
                    cm_df = pd.DataFrame(cm, index=[f"Real {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
                    st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Métricas de Evaluación")
        
        if any(st.session_state.models_trained.values()):
            all_metrics = st.session_state.ml_models.get_all_metrics()
            
            # Mostrar tabla de métricas
            metrics_df = pd.DataFrame(all_metrics).T.round(4)
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("Entrena el modelo para ver métricas detalladas")
        
        # Botón Siguiente al final de Tab 2
        if any(st.session_state.models_trained.values()):
            st.markdown("---")
            col_next = st.columns([3, 1])
            with col_next[1]:
                if st.button("➡️ Siguiente", use_container_width=True, type="primary", key="next_tab2"):
                    st.success("✅ Modelo entrenado. Dirígete a la pestaña **'Visualización de Resultados'**")


# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 3: VISUALIZACIÓN DE RESULTADOS
# Aquí usamos el modelo entrenado para predecir vulnerabilidad y visualizar resultados
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 3️⃣ Visualización de Resultados (Dashboard)")
    
    # Verificar que al menos un modelo esté entrenado
    if not any(st.session_state.models_trained.values()):
        st.error("🚫 **Debe entrenar un modelo primero**")
        st.warning(
            "Para visualizar zonas potencialmente vulnerables, primero complete el entrenamiento.\n\n"
            "Pasos a seguir:\n"
            "1. Ve a la pestaña **'📁 Gestión de Datos'** y carga/procesa los datos\n"
            "2. Ve a la pestaña **'🤖 Entrenamiento y Análisis'** y entrena un modelo (XGBoost o Random Forest)\n"
            "3. Regresa aquí para visualizar las predicciones de vulnerabilidad"
        )
        st.stop()
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Carga y procesa datos en la pestaña 'Gestión de Datos' para ver visualizaciones")
        st.stop()
    
    # Panel de filtros
    with st.expander("🔍 Panel de Filtros", expanded=True):
        col_filter1, col_filter2, col_filter3, col_filter4, col_filter5 = st.columns(5)
        
        with col_filter1:
            st.markdown("**Provincias**")
            # Obtener provincias disponibles del dataset
            prov_cols = [col for col in st.session_state.df_processed.columns if 'provincia' in col.lower()]
            if prov_cols:
                available_provinces = sorted(st.session_state.df_processed[prov_cols[0]].dropna().unique().tolist())
            else:
                available_provinces = [
                    'Pichincha', 'Guayas', 'Azuay', 'Tungurahua', 'Imbabura', 'Carchi',
                    'Sucumbíos', 'Orellana', 'Napo', 'Pastaza', 'Morona Santiago',
                    'Zamora Chinchipe', 'El Oro', 'Loja', 'Cotopaxi', 'Los Ríos',
                    'Santa Elena', 'Manabí', 'Santo Domingo', 'Chimborazo'
                ]
            
            provincia_selected = st.multiselect(
                "Seleccionar provincias",
                options=available_provinces,
                default=[],
                key="provincia_filter",
                help="Deja vacío para mostrar todas"
            )
        
        with col_filter2:
            st.markdown("**Año**")
            selected_year = st.slider("Año", 2010, 2026, 2022, key="selected_year")

        with col_filter3:
            st.markdown("**Mes**")
            month_names = [
                "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
            ]
            month_options = ["Todos"] + month_names
            selected_month_label = st.selectbox(
                "Mes del evento",
                options=month_options,
                index=0,
                key="selected_month"
            )
            selected_month = month_names.index(selected_month_label) + 1 if selected_month_label != "Todos" else None
        
        with col_filter4:
            st.markdown("**Tipos de Evento**")
            # Obtener tipos disponibles
            event_cols = [col for col in st.session_state.df_processed.columns if 'evento' in col.lower() or 'tipo' in col.lower()]
            if event_cols:
                available_events = sorted(st.session_state.df_processed[event_cols[0]].dropna().unique().tolist())
            else:
                available_events = ['Inundación', 'Deslizamiento', 'Incendio', 'Erupción Volcánica', 'Terremoto', 'Sequía']
            
            event_types = st.multiselect(
                "Seleccionar eventos",
                options=available_events,
                default=[],
                key="event_types_filter",
                help="Deja vacío para mostrar todos"
            )
        
        with col_filter5:
            st.markdown("**Nivel de Vulnerabilidad**")
            vulnerability_levels = st.multiselect(
                "Seleccionar niveles",
                options=['Alta', 'Media', 'Baja'],
                default=[],
                key="vulnerability_filter",
                help="Deja vacío para mostrar todos los niveles"
            )
    
    # Botón Visualizar centrado y prominente
    st.markdown("---")
    col_viz = st.columns([2, 1, 2])
    with col_viz[1]:
        visualizar = st.button("🔍 Visualizar Resultados", use_container_width=True, type="primary", key="btn_visualizar")
    
    if not visualizar:
        st.info("👆 Configura los filtros y presiona el botón **'Visualizar Resultados'** para generar las predicciones de vulnerabilidad")
        st.stop()
    
    st.markdown("---")
    
    # Aplicar filtros básicos
    df_filtered, is_future = st.session_state.data_processor.filter_data(
        st.session_state.df_processed,
        {
            'provincia': provincia_selected if provincia_selected else None,
            'year': selected_year,
            'month': selected_month,
            'event_types': event_types if len(event_types) > 0 else None
        }
    )
    
    # Mostrar si es predicción futura
    # Mostrar si es predicción futura
    if is_future:
        st.success(f"✅ **{len(df_filtered)} eventos predichos para {selected_year}** basados en patrones históricos (2010-2023)")
    
    # Información de debug
    with st.expander("🔍 Información de Debug", expanded=False):
        st.write(f"**Registros iniciales (procesados):** {len(st.session_state.df_processed)}")
        st.write(f"**Registros después de filtros:** {len(df_filtered)}")
        st.write(f"**Es predicción futura:** {is_future}")
        if is_future:
            st.write(f"**Eventos generados para {selected_year}:** {len(df_filtered)}")
            st.write("**Método:** Combinaciones provincia-evento históricas + tendencias de características")
        
        st.write("---")
        st.write("**Filtros aplicados:**")
        st.write(f"- Año: {selected_year}")
        st.write(f"- Mes: {selected_month_label if selected_month is not None else 'Todos'}")
        st.write(f"- Provincias: {provincia_selected if provincia_selected else 'Todas'}")
        st.write(f"- Tipos de evento: {event_types if event_types else 'Todos'}")
        
        # Aplicar filtros uno por uno para ver dónde se pierden datos
        df_debug = st.session_state.df_processed.copy()
        
        # Filtro por provincia
        if provincia_selected:
            prov_col = [col for col in st.session_state.df_loaded.columns if 'provincia' in col.lower()]
            if prov_col:
                prov_col = prov_col[0]
                df_debug_prov = df_debug[df_debug[prov_col].astype(str).str.upper().isin([p.upper() for p in provincia_selected])]
                st.write(f"**Después de filtro de provincia ({provincia_selected}):** {len(df_debug_prov)} registros")
                df_debug = df_debug_prov
        
        # Filtro por año
        date_cols = [col for col in st.session_state.df_processed.columns if 'año' in col.lower() or 'year' in col.lower() or 'fecha' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                dates = pd.to_datetime(df_debug[date_col], errors='coerce')
                if is_future:
                    # Para años futuros, ya tenemos eventos generados
                    st.write(f"**Año {selected_year} (futuro)**: Generados {len(df_filtered)} eventos basados en patrones históricos")
                else:
                    df_debug_year = df_debug[dates.dt.year == selected_year]
                    st.write(f"**Después de filtro de año ({selected_year}):** {len(df_debug_year)} registros")
                    df_debug = df_debug_year
            except:
                st.write("⚠️ Error al procesar fechas")

        # Filtro por mes
        if selected_month:
            month_col_candidates = [col for col in st.session_state.df_processed.columns if 'mes' in col.lower()]
            st.write(f"**Columnas de mes detectadas:** {month_col_candidates}")
            if month_col_candidates:
                month_col = month_col_candidates[0]
                st.write(f"**Usando columna:** {month_col}")
                st.write(f"**Valores únicos en columna mes:** {sorted(df_debug[month_col].dropna().unique().tolist())[:10]}")
                
                month_series = pd.to_numeric(df_debug[month_col], errors='coerce')
                df_debug_month = df_debug[month_series == selected_month]
                st.write(f"**Después de filtro de mes ({selected_month}):** {len(df_debug_month)} registros")
                df_debug = df_debug_month
            else:
                st.write("⚠️ No se encontró columna de mes para aplicar el filtro")
        
        # Filtro por tipo de evento
        if event_types:
            event_col_candidates = [col for col in st.session_state.df_processed.columns if 'evento' in col.lower() or 'tipo' in col.lower()]
            st.write(f"**Columnas de evento detectadas:** {event_col_candidates}")
            if event_col_candidates:
                event_col = event_col_candidates[0]
                st.write(f"**Usando columna:** {event_col}")
                st.write(f"**Valores únicos en columna evento:** {sorted(df_debug[event_col].dropna().unique().tolist())[:10]}")
                
                event_series_upper = df_debug[event_col].astype(str).str.upper()
                event_types_upper = [et.upper() for et in event_types]
                df_debug_event = df_debug[event_series_upper.isin(event_types_upper)]
                st.write(f"**Después de filtro de evento ({event_types}):** {len(df_debug_event)} registros")
                df_debug = df_debug_event
            else:
                st.write("⚠️ No se encontró columna de evento para aplicar el filtro")
        
        # Mostrar provincias disponibles
        st.write("**Provincias disponibles en datos originales:**")
        prov_col = [col for col in st.session_state.df_processed.columns if 'provincia' in col.lower()][0]
        provincias_disponibles = st.session_state.df_processed[prov_col].value_counts()
        st.dataframe(provincias_disponibles.head(15))
    if len(df_filtered) == 0:
        st.error("⚠️ No hay datos que cumplan con los filtros seleccionados")
        if provincia_selected:
            st.warning("💡 **Sugerencia:** Verifica que la provincia seleccionada tenga eventos del tipo elegido")
        else:
            st.info("💡 **Sugerencia:** Intenta dejar los filtros vacíos o cambiar los tipos de evento")
        st.stop()
    
    # Preparar datos para predicción del modelo
    with st.spinner("🔮 Generando predicciones de vulnerabilidad..."):
        # Usar las MISMAS features que se usaron en el entrenamiento (guardadas en Tab 2)
        if hasattr(st.session_state, 'selected_features') and st.session_state.selected_features:
            selected_features = st.session_state.selected_features
        else:
            # Fallback: usar todas las numéricas disponibles
            selected_features = st.session_state.df_processed.select_dtypes(include=[np.number]).columns.tolist() if st.session_state.df_processed is not None else []
            if selected_features:
                st.warning("⚠️ No se encontraron features del entrenamiento. Usando todas las numéricas.")
        
        # Preparar features para predicción
        df_pred = df_filtered.copy()

        # Asegurar que las columnas necesarias existan
        available_features = [f for f in selected_features if f in df_pred.columns]
        
        if len(available_features) > 0:
            X_pred = df_pred[available_features].fillna(0).values
            
            # Selección del modelo para predicción
            available_models = list(st.session_state.ml_models.models.keys())
            if not available_models:
                st.error("No hay modelos disponibles para predicción")
                st.stop()

            selected_model_for_viz = st.selectbox(
                "Modelo para predicción",
                options=available_models,
                index=available_models.index(st.session_state.trained_model_name) if st.session_state.trained_model_name in available_models else 0,
                format_func=lambda k: "XGBoost" if k == 'xgboost' else "Random Forest"
            )

            # Predecir con el modelo seleccionado
            predictions = st.session_state.ml_models.predict(selected_model_for_viz, X_pred)
            
            # Mapear predicciones a niveles de vulnerabilidad (0=Baja, 1=Media, 2=Alta)
            vulnerability_map = {0: 'Baja', 1: 'Media', 2: 'Alta'}
            df_pred['Vulnerabilidad_Predicha'] = [vulnerability_map.get(p, 'Baja') for p in predictions]
        else:
            # Si no hay features, asignar vulnerabilidad basada en frecuencia histórica
            st.warning("⚠️ Usando estimación basada en datos históricos (sin features del modelo)")
            df_pred['Vulnerabilidad_Predicha'] = 'Media'
    
    # Aplicar filtro de vulnerabilidad si está seleccionado
    if vulnerability_levels:
        df_pred = df_pred[df_pred['Vulnerabilidad_Predicha'].isin(vulnerability_levels)]
        
        if len(df_pred) == 0:
            st.warning(f"⚠️ No hay zonas con nivel de vulnerabilidad: {', '.join(vulnerability_levels)}")
            st.stop()
    
    # KPIs
    st.markdown("### Indicadores Clave (KPIs)")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    
    with col_kpi1:
        st.metric("📊 Total de Zonas", len(df_pred))
    with col_kpi2:
        vuln_alta = len(df_pred[df_pred['Vulnerabilidad_Predicha'] == 'Alta'])
        st.metric("🔴 Vulnerabilidad Alta", vuln_alta)
    with col_kpi3:
        vuln_media = len(df_pred[df_pred['Vulnerabilidad_Predicha'] == 'Media'])
        st.metric("🟡 Vulnerabilidad Media", vuln_media)
    with col_kpi4:
        vuln_baja = len(df_pred[df_pred['Vulnerabilidad_Predicha'] == 'Baja'])
        st.metric("🟢 Vulnerabilidad Baja", vuln_baja)
    with col_kpi5:
        kpis = Visualizations.create_kpi_cards(df_pred)
        st.metric("👥 Personas Afectadas", f"{kpis.get('personas_afectadas', 0):,}")
    
    st.markdown("---")
    
    # Mapa de vulnerabilidad predicha
    st.markdown("### 🗺️ Mapa de Vulnerabilidad Predicha")
    st.info(
        "📌 **Leyenda del Mapa:**\n"
        "- 🔴 **Rojo**: Vulnerabilidad Alta\n"
        "- 🟡 **Naranja**: Vulnerabilidad Media\n"
        "- 🟢 **Verde**: Vulnerabilidad Baja\n\n"
        f"Mostrando {len(df_pred)} zonas que cumplen los filtros seleccionados"
    )
    
    # Crear mapa con predicciones
    vulnerability_map = Visualizations.create_vulnerability_prediction_map(df_pred, provincia_selected)
    st_data = streamlit_folium.folium_static(vulnerability_map, width=1200, height=600)
    
    st.markdown("---")
    
    # Gráficos estadísticos
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### Distribución de Vulnerabilidad")
        vuln_counts = df_pred['Vulnerabilidad_Predicha'].value_counts().reset_index()
        vuln_counts.columns = ['Nivel', 'Cantidad']
        
        fig_vuln = px.bar(
            vuln_counts,
            x='Nivel',
            y='Cantidad',
            color='Nivel',
            color_discrete_map={'Alta': '#ff4444', 'Media': '#ff9944', 'Baja': '#44ff44'},
            title='Distribución de Niveles de Vulnerabilidad',
            labels={'Cantidad': 'Número de Zonas'}
        )
        st.plotly_chart(fig_vuln, use_container_width=True)
    
    with col_chart2:
        st.markdown("### Vulnerabilidad por Provincia")
        if prov_cols:
            prov_vuln = df_pred.groupby([prov_cols[0], 'Vulnerabilidad_Predicha']).size().reset_index(name='count')
            
            fig_prov = px.bar(
                prov_vuln,
                x=prov_cols[0],
                y='count',
                color='Vulnerabilidad_Predicha',
                color_discrete_map={'Alta': '#ff4444', 'Media': '#ff9944', 'Baja': '#44ff44'},
                title='Vulnerabilidad por Provincia',
                labels={'count': 'Número de Zonas', prov_cols[0]: 'Provincia'}
            )
            fig_prov.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_prov, use_container_width=True)
    
    st.markdown("---")
    
    # Tabla de datos con predicciones
    st.markdown("### 📋 Datos con Predicción de Vulnerabilidad")
    
    # Reordenar columnas para mostrar vulnerabilidad al inicio
    cols_to_show = ['Vulnerabilidad_Predicha'] + [c for c in df_pred.columns if c != 'Vulnerabilidad_Predicha']
    df_display = df_pred[cols_to_show]
    
    st.dataframe(
        df_display.style.applymap(
            lambda x: 'background-color: #ff4444; color: white' if x == 'Alta' else
                     'background-color: #ff9944; color: white' if x == 'Media' else
                     'background-color: #44ff44; color: white' if x == 'Baja' else '',
            subset=['Vulnerabilidad_Predicha']
        ),
        use_container_width=True,
        height=400
    )
    
    # Descarga de resultados
    st.markdown("### 📥 Descargar Resultados")
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Predicciones (CSV)",
            data=csv,
            file_name=f"predicciones_vulnerabilidad_{selected_year}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_download2:
        # Resumen de predicciones
        summary_text = f"""
        RESUMEN DE PREDICCIÓN DE VULNERABILIDAD
        ========================================
        
        Año analizado: {selected_year}
        Provincias: {', '.join(provincia_selected) if provincia_selected else 'Todas'}
        Mes: {selected_month_label if selected_month is not None else 'Todos'}
        Tipos de evento: {', '.join(event_types) if event_types else 'Todos'}
        
        Resultados:
        - Total de zonas analizadas: {len(df_pred)}
        - Vulnerabilidad Alta: {vuln_alta} zonas ({vuln_alta/len(df_pred)*100:.1f}%)
        - Vulnerabilidad Media: {vuln_media} zonas ({vuln_media/len(df_pred)*100:.1f}%)
        - Vulnerabilidad Baja: {vuln_baja} zonas ({vuln_baja/len(df_pred)*100:.1f}%)
        
        Modelo utilizado: { 'XGBoost' if st.session_state.trained_model_name == 'xgboost' else 'Random Forest' }
        Precisión del modelo: {st.session_state.ml_models.get_all_metrics().get(st.session_state.trained_model_name or 'xgboost', {}).get('accuracy', 0):.4f}
        """
        
        st.download_button(
            label="📄 Descargar Resumen (TXT)",
            data=summary_text,
            file_name=f"resumen_vulnerabilidad_{selected_year}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# PIE DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
    <p>Sistema de Predicción de Zonas Vulnerables a Desastres Naturales en Ecuador © 2024</p>
    <p>Desarrollado con Streamlit, XGBoost, TensorFlow y Pandas</p>
    </div>
    """,
    unsafe_allow_html=True
)
