# 🗺️ Sistema de Predicción de Zonas Vulnerables a Desastres Naturales en Ecuador

Sistema interactivo basado en **Streamlit** para predecir y visualizar zonas vulnerables a desastres naturales en Ecuador, utilizando múltiples modelos de Machine Learning y Deep Learning.

## 📋 Descripción General

Este proyecto implementa un **Dashboard Interactivo** que permite:

1. **Carga y Preprocesamiento de Datos** - Gestión de archivos CSV/XLSX con limpieza automática
2. **Entrenamiento de Modelos** - Selección y configuración de 4+ modelos ML/DL
3. **Visualización de Resultados** - Mapas interactivos, gráficos y KPIs
4. **Análisis Temporal** - Series temporales y tendencias de desastres
5. **Predicción Futura** - Generación de eventos predichos basados en patrones históricos

## 🛠️ Stack Tecnológico

### Lenguaje y Framework
- **Python 3.10+** - Lenguaje principal
- **Streamlit** - Framework para interfaz interactiva

### Librerías de Datos y ML
- **Pandas** - Manipulación de datos
- **NumPy** - Cálculos numéricos
- **Scikit-Learn** - Preprocesamiento y Random Forest
- **XGBoost** - Gradient Boosting (2 modelos diferentes)
- **TensorFlow/Keras** - Redes LSTM

### Visualización
- **Plotly** - Gráficos interactivos
- **Folium** - Mapas geoespaciales
- **Matplotlib & Seaborn** - Gráficos estáticos
- **GeoPandas** - Análisis geoespacial

---

## 📦 Archivos Creados

### 📂 Estructura de Directorios
```
Proyecto_SRG/
├── .streamlit/
│   └── config.toml              # Configuración de Streamlit
├── src/
│   ├── __init__.py
│   ├── data_processor.py         # Módulo de procesamiento de datos
│   ├── models.py                 # Modelos ML/DL
│   └── visualizations.py         # Visualizaciones e gráficos
├── utils/
│   └── helpers.py                # Funciones utilitarias
├── data/                         # Carpeta para datos
├── models/                       # Carpeta para guardar modelos
├── app.py                        # 🎯 Aplicación Streamlit principal
├── config.py                     # Configuración centralizada
├── examples.py                   # Ejemplos de uso
├── install.py                    # Script de instalación interactivo
├── verify.py                     # Verificación de instalación
├── run.sh                        # Script para ejecutar en Linux/Mac
├── requirements.txt              # Dependencias Python
└── README.md                     # 📚 Este archivo
```

### 🔧 Módulos Principales

#### 1. **data_processor.py** (Procesamiento de Datos)
```python
DataProcessor:
  - load_data()                # Carga CSV/XLSX
  - clean_data()               # Limpia duplicados y nulos
  - preprocess_for_modeling()  # Encoding y escalado
  - get_event_types()          # Obtiene tipos de eventos
  - get_regions()              # Obtiene provincias
  - filter_data()              # Filtra por provincia/año/mes/evento
  - generate_future_events()   # Genera predicciones futuras
```

#### 2. **models.py** (Machine Learning)
```python
MLModels:
  - prepare_data()             # Division train/test
  - train_xgboost()            # Modelo XGBoost
  - train_random_forest()      # Modelo Random Forest
  - train_gradient_boosting()  # Gradient Boosting
  - predict()                  # Realiza predicciones
  - get_all_metrics()          # Retorna métricas de evaluación
  - get_confusion_matrix()     # Matriz de confusión
```

#### 3. **visualizations.py** (Gráficos y Mapas)
```python
Visualizations:
  - create_kpi_cards()              # Indicadores clave
  - create_vulnerability_prediction_map()  # Mapa de predicciones
  - create_event_distribution()     # Distribución de eventos
  - create_risk_analysis()          # Análisis de riesgo
```

---

## 🚀 Instalación y Ejecución

### 1. Requisitos Previos
- Python 3.10 o superior
- pip (gestor de paquetes Python)
- Git (opcional, para clonar el repositorio)

### 2. Instalación

#### Opción A: Script Automático (Recomendado)
```bash
cd /path/to/Proyecto_SRG
python install.py
```

#### Opción B: Manual
```bash
# 1. Navegar al directorio del proyecto
cd /path/to/Proyecto_SRG

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación
streamlit run app.py
```

#### Opción C: Script en Linux/macOS
```bash
cd /path/to/Proyecto_SRG
chmod +x run.sh
./run.sh
```

### 3. Acceso a la Aplicación
La aplicación se abrirá automáticamente en:
```
http://localhost:8501
```

---

## 📊 Componentes de la Interfaz

### Estructura Principal
```
┌─────────────────────────────────────────────────────────────┐
│ 🗺️ Sistema de Predicción de Desastres - Ecuador            │
├─────────────────────────────────────────────────────────────┤
│ [📁 Datos] [🤖 Modelos] [📊 Visualización]                  │
├─────────────────────────────────────────────────────────────┤
│                   CONTENIDO DINÁMICO                         │
└─────────────────────────────────────────────────────────────┘
```

### 📑 Pestaña 1: Gestión de Datos
**Funcionalidades:**
- ✅ Carga de archivos (Drag & Drop)
- 📊 Vista previa de datos
- 📈 Estadísticas de calidad (registros, columnas, nulos, duplicados)
- 🧹 Preprocesamiento automático
- 📥 Descarga de datos limpios

**Operaciones:**
- Carga automática de demostración
- Detección automática de columnas (provincia, año, mes, evento)
- Limpieza inteligente de datos
- Resumen de cambios realizados

### 🤖 Pestaña 2: Entrenamiento y Análisis
**Modelos disponibles:**

1. **XGBoost** (Modelo principal)
   - Características: Rápido, preciso, manejo de no linealidades
   - Parámetros ajustables: n_estimators (50-500), max_depth (3-20), learning_rate (0.01-0.3)
   - Métricas: Accuracy, Precision, Recall, F1-Score

2. **Random Forest**
   - Características: Robusto, interpretable, maneja interacciones
   - Parámetros ajustables: n_estimators (50-500), max_depth (3-30), min_samples_split (2-20)
   - Métricas: Accuracy, Precision, Recall, F1-Score

3. **Gradient Boosting**
   - Características: Mejora iterativa, flexible, preciso
   - Parámetros ajustables: n_estimators (50-500), max_depth (3-20), learning_rate
   - Métricas: Accuracy, Precision, Recall, F1-Score

**Proceso de Entrenamiento:**
- Selección de variables predictoras
- Ajuste de hiperparámetros
- Barra de progreso en vivo
- Matriz de confusión
- Comparativa de métricas

### 📊 Pestaña 3: Visualización de Resultados
**Componentes:**
- 📊 **KPIs:**
  - Total de zonas analizadas
  - Vulnerabilidad Alta (🔴)
  - Vulnerabilidad Media (🟡)
  - Vulnerabilidad Baja (🟢)
  - Personas afectadas

- 🗺️ **Mapas Interactivos:**
  - Mapa de vulnerabilidad predicha con leyenda de colores
  - Zoom y paneo interactivo
  - Marcadores con información de zonas

- 📈 **Gráficos:**
  - Distribución de vulnerabilidad (pie chart)
  - Vulnerabilidad por provincia (bar chart)
  - Análisis temporal de eventos

- 📋 **Tabla de Datos:**
  - Datos filtrados con predicciones
  - Colores según nivel de vulnerabilidad
  - Exportable a CSV

- 📥 **Descargas:**
  - Predicciones en CSV
  - Resumen en TXT

**Filtros:**
- 📍 **Provincias:** Multiselección de provincias ecuatorianas
- 📅 **Año:** Slider para seleccionar año (con generación automática de eventos futuros)
- 🗓️ **Mes:** Selector de mes específico
- 🌍 **Tipos de Evento:** Multiselección (Inundación, Deslizamiento, Incendio, etc.)
- 🎯 **Nivel de Vulnerabilidad:** Multiselección (Alta, Media, Baja)

---

## 💾 Formato de Datos Esperados

El archivo CSV/XLSX debe contener columnas similares a:

```
Fecha         | Provincia    | Mes | Evento        | Personas_Afectadas | Viviendas_Dañadas | Latitud | Longitud
2015-01-15    | Pichincha    | 1   | Inundación    | 150                | 45                | -0.35   | -78.52
2015-02-20    | Guayas       | 2   | Deslizamiento | 250                | 120               | -2.20   | -79.89
2015-03-10    | Tungurahua   | 3   | Incendio      | 80                 | 30                | -1.20   | -78.60
```

### Columnas Requeridas Mínimas:
- Una columna de fecha/año
- Una columna de provincia/región
- Una columna de tipo de evento

### Columnas Opcionales:
- Mes del evento
- Personas afectadas
- Viviendas dañadas/destruidas
- Fallecidos, heridos, desaparecidos
- Latitud/Longitud (para mapas precisos)

---

## 🔍 Predicción Futura

El sistema puede predecir eventos para años no presentes en los datos:

**Algoritmo:**
1. Detecta si el año seleccionado existe en datos históricos
2. Si NO existe: genera eventos predichos basados en:
   - Combinaciones provincia-evento observadas históricamente
   - Distribución de estacionalidad (mes)
   - Rango de ubicación geográfica
   - Tendencias de características numéricas (regresión lineal)

**Ejemplo:**
- Si seleccionas 2025 (sin datos) → Genera ~X eventos predichos
- Si seleccionas 2020 (con datos) → Usa datos reales de ese año

---

## ⚙️ Configuración Avanzada

### Hiperparámetros Predeterminados
Están en `src/models.py`:
- **XGBoost:** n_estimators=150, max_depth=8, learning_rate=0.1
- **Random Forest:** n_estimators=100, max_depth=10, min_samples_split=5
- **Gradient Boosting:** n_estimators=200, max_depth=5, learning_rate=0.05

### Variables de Entorno
```bash
# Puerto personalizado
export STREAMLIT_SERVER_PORT=8501

# Modo de depuración
export STREAMLIT_LOGGER_LEVEL=debug
```

---

## 🧪 Testing y Validación

### Scripts de Verificación
```bash
# Verificar instalación completa
python verify.py

# Ejecutar ejemplos funcionales
python examples.py basic       # Flujo completo
python examples.py filter      # Filtrado de datos
python examples.py comparison  # Comparación de modelos
python examples.py visual      # Visualizaciones
```

### Instalación Interactiva
```bash
python install.py  # Instalación guiada con feedback
```

---

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit --upgrade
```

### Error: "No hay datos disponibles"
- Asegúrate de cargar un archivo en la pestaña "Gestión de Datos"
- El archivo debe estar en formato CSV o XLSX
- Verifica que tenga al menos una columna de fecha/año

### Error: "TensorFlow issues"
En Windows puede requerir instalación especial:
```bash
pip install tensorflow-cpu  # Versión CPU más ligera
```

### El mapa no carga
- Verifica tu conexión a internet (Folium requiere OpenStreetMap)
- Intenta actualizar: `pip install folium --upgrade`

### Filtros no funcionan
- Asegúrate de haber presionado "Visualizar Resultados"
- Los datos deben estar procesados (Tab 1)
- El modelo debe estar entrenado (Tab 2)

---

## 📊 Ejemplo de Uso Completo

### Paso 1: Iniciar
```bash
streamlit run app.py
```

### Paso 2: Cargar Datos (Pestaña 1)
- Click en "📂 Cargar Datos de Demostración"
- O sube tu archivo CSV/XLSX

### Paso 3: Limpiar Datos
- Click en "🧹 Preprocesar/Limpiar Datos"
- Verifica resumen de limpieza

### Paso 4: Entrenar Modelo (Pestaña 2)
- Selecciona variables predictoras
- Elige modelo (XGBoost recomendado)
- Ajusta hiperparámetros si deseas
- Click en "🚀 Entrenar Modelo"
- Espera a ver métricas y matriz de confusión

### Paso 5: Visualizar Resultados (Pestaña 3)
- Aplica filtros (provincia, año, mes, evento)
- Click en "🔍 Visualizar Resultados"
- Explora mapas, gráficos y tablas
- Descarga resultados en CSV/TXT

---

## 📈 Mejoras Futuras

- [ ] Integración con bases de datos en tiempo real
- [ ] Predicciones en tiempo real (streaming)
- [ ] Exportación de reportes PDF
- [ ] Validación cruzada k-fold automática
- [ ] Tuning automático de hiperparámetros (AutoML)
- [ ] Análisis de importancia de características
- [ ] Ensemble de modelos combinados
- [ ] API REST para integraciones externas
- [ ] Sistema de alertas de riesgo
- [ ] Caché de predicciones
