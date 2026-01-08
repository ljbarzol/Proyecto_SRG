# Configuración del proyecto

## Variables de Proyecto
PROJECT_NAME = "Sistema de Predicción de Desastres Naturales"
VERSION = "1.0.0"
AUTHOR = "Equipo de Desarrollo - SRG"

## Configuración de Datos
DATA_DIR = "data/"
MODELS_DIR = "models/"
UTILS_DIR = "utils/"

## Configuración de Modelos
XGB_DEFAULT_PARAMS = {
    "n_estimators": 150,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

RF_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42
}

LSTM_DEFAULT_PARAMS = {
    "lookback": 10,
    "epochs": 50,
    "batch_size": 32,
    "validation_split": 0.2
}

## Configuración de Streamlit
STREAMLIT_CONFIG = {
    "page_title": "Sistema de Predicción de Desastres - Ecuador",
    "page_icon": "🗺️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

## Colores y Estilos
COLORS = {
    "risk_high": "#FF6B6B",
    "risk_medium": "#FFA500",
    "risk_low": "#51CF66",
    "primary": "#1f77b4",
    "secondary": "#764ba2"
}

## Provincias de Ecuador
PROVINCIAS_ECUADOR = [
    'Azuay', 'Bolívar', 'Cañar', 'Carchi', 'Chimborazo', 'Cotopaxi',
    'El Oro', 'Esmeraldas', 'Galápagos', 'Guayas', 'Imbabura', 'Loja',
    'Los Ríos', 'Manabí', 'Morona Santiago', 'Napo', 'Orellana',
    'Pastaza', 'Pichincha', 'Santa Elena', 'Santo Domingo',
    'Sucumbíos', 'Tungurahua', 'Zamora Chinchipe'
]

## Tipos de Eventos
TIPOS_EVENTOS = [
    'Inundación',
    'Deslizamiento',
    'Incendio',
    'Erupción Volcánica',
    'Terremoto',
    'Sequía',
    'Granizo',
    'Vendaval'
]

## Niveles de Severidad
NIVELES_SEVERIDAD = ['Baja', 'Media', 'Alta', 'Crítica']

## Configuración de Mapas
MAP_CONFIG = {
    "center": [-1.8312, -78.1834],  # Centro de Ecuador
    "zoom": 6,
    "tiles": "OpenStreetMap"
}

## Parámetros de Evaluación
EVAL_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

## Parámetros de Visualización
PLOT_HEIGHT = 400
PLOT_WIDTH = 700
HEATMAP_RADIUS = 20
HEATMAP_BLUR = 15
