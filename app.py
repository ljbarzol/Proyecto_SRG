"""
Aplicación Streamlit: Dashboard de Predicción de Zonas Vulnerables a Desastres Naturales en Ecuador
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
import plotly.express as px

# Agregar la ruta del src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processor import DataProcessor
from src.models import MLModels
from src.visualizations import Visualizations
import streamlit_folium
import warnings
warnings.filterwarnings('ignore')


# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción de Desastres - Ecuador",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
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

# Inicializar session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {}

# ============================================================================
# CONTENIDO PRINCIPAL
# ============================================================================
st.markdown("# 🗺️ Sistema de Predicción de Desastres Naturales - Ecuador")
st.markdown("---")

# Crear las pestañas (tabs)
tab1, tab2, tab3 = st.tabs(["📁 Gestión de Datos", "🤖 Entrenamiento y Análisis", "📊 Visualización de Resultados"])

# ============================================================================
# TAB 1: GESTIÓN DE DATOS
# ============================================================================
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
                demo_file = "/workspaces/Proyecto_SRG/SGR_Eventos.csv"
                if os.path.exists(demo_file):
                    st.session_state.df_loaded = st.session_state.data_processor.load_data(file_path=demo_file)
                    st.success("✅ Datos de demostración cargados")
                else:
                    # Crear datos de demostración
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
        
        col_preview, col_info = st.columns([2, 1])
        
        with col_preview:
            st.dataframe(
                st.session_state.df_loaded.head(10),
                use_container_width=True,
                height=300
            )
        
        with col_info:
            st.markdown("### Información de Columnas")
            for col in st.session_state.df_loaded.columns:
                dtype = str(st.session_state.df_loaded[col].dtype)
                st.text(f"• **{col}**: {dtype}")
        
        st.markdown("---")
        st.markdown("### Preprocesamiento y Limpieza")
        
        col_clean, col_download = st.columns(2)
        
        with col_clean:
            if st.button("🧹 Preprocesar/Limpiar Datos", use_container_width=True):
                with st.spinner("Limpiando datos..."):
                    df_clean, stats = st.session_state.data_processor.clean_data(st.session_state.df_loaded)
                    st.session_state.df_processed = df_clean
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("✅ Registros Válidos", stats['registros_finales'])
                    with col_stat2:
                        st.metric("❌ Registros Descartados", stats['registros_descartados'])
                    with col_stat3:
                        st.metric("🔄 Duplicados Eliminados", stats['duplicados_eliminados'])
                    
                    st.success("✅ Datos preprocesados correctamente")
        
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

# ============================================================================
# TAB 2: ENTRENAMIENTO Y ANÁLISIS
# ============================================================================
with tab2:
    st.markdown("## 2️⃣ Entrenamiento y Análisis (ML/DL)")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Primero debes cargar y procesar los datos en la pestaña 'Gestión de Datos'")
    else:
        st.markdown("### Configuración del Modelo XGBoost")
        st.info("📌 El sistema utiliza **XGBoost** como modelo de predicción principal")
        
        # Selección de variable objetivo
        st.markdown("#### Variable Objetivo")
        target_options = [col for col in st.session_state.df_processed.columns if st.session_state.df_processed[col].dtype in ['object', 'int64', 'float64']]
        target_col = st.selectbox("Seleccionar Variable Objetivo", target_options if target_options else ["No disponible"])
        
        st.markdown("---")
        
        # Selección de características/variables
        st.markdown("#### Selección de Variables (Features)")
        
        numeric_cols = st.session_state.df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remover la variable objetivo de las opciones si es numérica
        if target_col != "No disponible" and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Checkbox para seleccionar todas
        select_all = st.checkbox("✅ Seleccionar todas las variables", value=True)
        
        # Tabla de selección de variables
        st.markdown("**Variables disponibles:**")
        
        # Organizar en columnas de 3
        num_cols_display = 3
        cols_per_row = num_cols_display
        
        if 'selected_features' not in st.session_state:
            st.session_state.selected_features = numeric_cols.copy()
        
        # Si se selecciona "seleccionar todas", actualizar
        if select_all:
            st.session_state.selected_features = numeric_cols.copy()
        
        selected_features = []
        
        # Crear tabla de checkboxes
        for i in range(0, len(numeric_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
                with cols[j]:
                    is_checked = select_all or col_name in st.session_state.selected_features
                    if st.checkbox(col_name, value=is_checked, key=f"feature_{col_name}"):
                        selected_features.append(col_name)
        
        st.session_state.selected_features = selected_features
        
        st.markdown(f"**Variables seleccionadas:** {len(selected_features)} / {len(numeric_cols)}")
        
        st.markdown("---")
        
        # Hiperparámetros de XGBoost
        st.markdown("#### Hiperparámetros del Modelo")
        
        col_hp1, col_hp2, col_hp3 = st.columns(3)
        
        with col_hp1:
            xgb_n_estimators = st.slider("n_estimators", 50, 500, 150, step=10, 
                                         help="Número de árboles de decisión")
        
        with col_hp2:
            xgb_max_depth = st.slider("max_depth", 3, 20, 8, 
                                      help="Profundidad máxima de cada árbol")
        
        with col_hp3:
            xgb_learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, step=0.01,
                                         help="Tasa de aprendizaje del modelo")
        
        col_hp4, col_hp5, col_hp6 = st.columns(3)
        
        with col_hp4:
            xgb_subsample = st.slider("subsample", 0.5, 1.0, 0.8, step=0.1,
                                     help="Fracción de muestras para entrenar cada árbol")
        
        with col_hp5:
            xgb_colsample = st.slider("colsample_bytree", 0.5, 1.0, 0.8, step=0.1,
                                     help="Fracción de características para cada árbol")
        
        with col_hp6:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, step=0.05,
                                 help="Porcentaje de datos para prueba")
        
        st.markdown("---")
        
        # Botón de entrenamiento
        if st.button("🚀 Entrenar Modelo XGBoost", use_container_width=True, type="primary"):
            if len(selected_features) == 0:
                st.error("❌ Debes seleccionar al menos una variable para entrenar el modelo")
            else:
                with st.spinner("Entrenando modelo XGBoost..."):
                    # Preparar datos con las features seleccionadas
                    X = st.session_state.df_processed[selected_features].fillna(0).values
                    
                    # Crear target variable
                    if target_col != "No disponible":
                        if st.session_state.df_processed[target_col].dtype == 'object':
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            y = le.fit_transform(st.session_state.df_processed[target_col].astype(str))
                        else:
                            y = (st.session_state.df_processed[target_col] > st.session_state.df_processed[target_col].median()).astype(int)
                    else:
                        y = np.random.randint(0, 2, len(X))
                    
                    # Preparar datos en los modelos
                    st.session_state.ml_models.prepare_data(X, y, test_size=test_size)
                    
                    # Entrenar XGBoost con los hiperparámetros configurados
                    metrics_results = st.session_state.ml_models.train_xgboost({
                        'n_estimators': xgb_n_estimators,
                        'max_depth': xgb_max_depth,
                        'learning_rate': xgb_learning_rate,
                        'subsample': xgb_subsample,
                        'colsample_bytree': xgb_colsample
                    })
                    st.session_state.models_trained['xgboost'] = True
                    
                    st.success("✅ ¡Entrenamiento completado!")
                    
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
        
        st.markdown("---")
        st.markdown("### Métricas de Evaluación")
        
        if st.session_state.models_trained.get('xgboost'):
            all_metrics = st.session_state.ml_models.get_all_metrics()
            
            # Mostrar tabla de métricas
            metrics_df = pd.DataFrame(all_metrics).T.round(4)
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("Entrena el modelo para ver métricas detalladas")


# ============================================================================
# TAB 3: VISUALIZACIÓN DE RESULTADOS
# ============================================================================
with tab3:
    st.markdown("## 3️⃣ Visualización de Resultados (Dashboard)")
    
    # Verificar que el modelo esté entrenado (CU-03 completado)
    if not st.session_state.models_trained.get('xgboost'):
        st.error("🚫 **Debe entrenar el modelo XGBoost primero**")
        st.warning(
            "⚠️ Según el caso de uso CU-04, para visualizar zonas potencialmente vulnerables "
            "es necesario completar primero el CU-03 'Analizar patrones de vulnerabilidad'.\n\n"
            "**Pasos a seguir:**\n"
            "1. Ve a la pestaña **'📁 Gestión de Datos'** y carga/procesa los datos\n"
            "2. Ve a la pestaña **'🤖 Entrenamiento y Análisis'** y entrena el modelo XGBoost\n"
            "3. Regresa aquí para visualizar las predicciones de vulnerabilidad"
        )
        st.stop()
    
    if st.session_state.df_loaded is None:
        st.warning("⚠️ Carga datos en la pestaña 'Gestión de Datos' para ver visualizaciones")
        st.stop()
    
    # Panel de filtros
    with st.expander("🔍 Panel de Filtros", expanded=True):
        col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
        
        with col_filter1:
            st.markdown("**Provincias**")
            # Obtener provincias disponibles del dataset
            prov_cols = [col for col in st.session_state.df_loaded.columns if 'provincia' in col.lower()]
            if prov_cols:
                available_provinces = sorted(st.session_state.df_loaded[prov_cols[0]].dropna().unique().tolist())
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
            st.markdown("**Rango Temporal**")
            year_start = st.slider("Año Inicial", 2010, 2022, 2015, key="year_start")
            year_end = st.slider("Año Final", 2010, 2022, 2022, key="year_end")
        
        with col_filter3:
            st.markdown("**Tipos de Evento**")
            # Obtener tipos disponibles
            event_cols = [col for col in st.session_state.df_loaded.columns if 'evento' in col.lower() or 'tipo' in col.lower()]
            if event_cols:
                available_events = sorted(st.session_state.df_loaded[event_cols[0]].dropna().unique().tolist())
            else:
                available_events = ['Inundación', 'Deslizamiento', 'Incendio', 'Erupción Volcánica', 'Terremoto', 'Sequía']
            
            event_types = st.multiselect(
                "Seleccionar eventos",
                options=available_events,
                default=[],
                key="event_types_filter",
                help="Deja vacío para mostrar todos"
            )
        
        with col_filter4:
            st.markdown("**Nivel de Vulnerabilidad**")
            vulnerability_levels = st.multiselect(
                "Seleccionar niveles",
                options=['Alta', 'Media', 'Baja'],
                default=[],
                key="vulnerability_filter",
                help="Deja vacío para mostrar todos los niveles"
            )
    
    st.markdown("---")
    
    # Aplicar filtros básicos
    df_filtered = st.session_state.data_processor.filter_data(
        st.session_state.df_loaded,
        {
            'provincia': provincia_selected if provincia_selected else None,
            'year_range': (year_start, year_end),
            'event_types': event_types if len(event_types) > 0 else None  # Solo aplicar si hay eventos seleccionados
        }
    )
    
    # Información de debug
    with st.expander("🔍 Información de Debug", expanded=False):
        st.write(f"**Registros iniciales:** {len(st.session_state.df_loaded)}")
        
        # Aplicar filtros uno por uno para ver dónde se pierden datos
        df_debug = st.session_state.df_loaded.copy()
        
        # Filtro por provincia
        if provincia_selected:
            prov_col = [col for col in st.session_state.df_loaded.columns if 'provincia' in col.lower()]
            if prov_col:
                prov_col = prov_col[0]
                df_debug_prov = df_debug[df_debug[prov_col].astype(str).str.upper().isin([p.upper() for p in provincia_selected])]
                st.write(f"**Después de filtro de provincia ({provincia_selected}):** {len(df_debug_prov)} registros")
                df_debug = df_debug_prov
        
        # Filtro por años
        date_cols = [col for col in st.session_state.df_loaded.columns if 'año' in col.lower() or 'year' in col.lower() or 'fecha' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                dates = pd.to_datetime(df_debug[date_col], errors='coerce')
                df_debug_year = df_debug[(dates.dt.year >= year_start) & (dates.dt.year <= year_end)]
                st.write(f"**Después de filtro de años ({year_start}-{year_end}):** {len(df_debug_year)} registros")
                df_debug = df_debug_year
            except:
                st.write("⚠️ Error al procesar fechas")
        
        # Filtro por evento
        if event_types:
            event_cols = [col for col in st.session_state.df_loaded.columns if 'evento' in col.lower() or 'tipo' in col.lower()]
            if event_cols:
                event_col = event_cols[0]
                df_debug_event = df_debug[df_debug[event_col].isin(event_types)]
                st.write(f"**Después de filtro de eventos ({event_types}):** {len(df_debug_event)} registros")
                df_debug = df_debug_event
        
        # Mostrar provincias disponibles
        st.write("**Provincias disponibles en datos originales:**")
        prov_col = [col for col in st.session_state.df_loaded.columns if 'provincia' in col.lower()][0]
        provincias_disponibles = st.session_state.df_loaded[prov_col].value_counts()
        st.dataframe(provincias_disponibles.head(15))
    
    if len(df_filtered) == 0:
        st.error("⚠️ No hay datos que cumplan con los filtros seleccionados")
        st.info("💡 **Sugerencia:** Intenta ampliar el rango de años o dejar los filtros vacíos")
        st.stop()
    
    # Preparar datos para predicción del modelo
    with st.spinner("🔮 Generando predicciones de vulnerabilidad..."):
        # Usar las mismas features que se usaron en el entrenamiento
        if 'selected_features' in st.session_state and len(st.session_state.selected_features) > 0:
            selected_features = st.session_state.selected_features
        else:
            # Fallback: usar todas las numéricas disponibles
            selected_features = st.session_state.df_processed.select_dtypes(include=[np.number]).columns.tolist() if st.session_state.df_processed is not None else []
        
        # Preparar features para predicción
        df_pred = df_filtered.copy()
        
        # Asegurar que las columnas necesarias existan
        available_features = [f for f in selected_features if f in df_pred.columns]
        
        if len(available_features) > 0:
            X_pred = df_pred[available_features].fillna(0).values
            
            # Predecir con el modelo entrenado
            predictions = st.session_state.ml_models.predict('xgboost', X_pred)
            
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
            file_name=f"predicciones_vulnerabilidad_{year_start}_{year_end}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_download2:
        # Resumen de predicciones
        summary_text = f"""
        RESUMEN DE PREDICCIÓN DE VULNERABILIDAD
        ========================================
        
        Período analizado: {year_start} - {year_end}
        Provincias: {', '.join(provincia_selected) if provincia_selected else 'Todas'}
        Tipos de evento: {', '.join(event_types) if event_types else 'Todos'}
        
        Resultados:
        - Total de zonas analizadas: {len(df_pred)}
        - Vulnerabilidad Alta: {vuln_alta} zonas ({vuln_alta/len(df_pred)*100:.1f}%)
        - Vulnerabilidad Media: {vuln_media} zonas ({vuln_media/len(df_pred)*100:.1f}%)
        - Vulnerabilidad Baja: {vuln_baja} zonas ({vuln_baja/len(df_pred)*100:.1f}%)
        
        Modelo utilizado: XGBoost
        Precisión del modelo: {st.session_state.ml_models.get_all_metrics().get('xgboost', {}).get('accuracy', 0):.4f}
        """
        
        st.download_button(
            label="📄 Descargar Resumen (TXT)",
            data=summary_text,
            file_name=f"resumen_vulnerabilidad_{year_start}_{year_end}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ============================================================================
# PIE DE PÁGINA
# ============================================================================
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
