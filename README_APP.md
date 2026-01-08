# 🗺️ Sistema de Predicción de Zonas Vulnerables a Desastres Naturales en Ecuador

Sistema interactivo basado en **Streamlit** para predecir y visualizar zonas vulnerables a desastres naturales en Ecuador, utilizando múltiples modelos de Machine Learning y Deep Learning.

## 📋 Descripción General

Este proyecto implementa un **Dashboard Interactivo** que permite:

1. **Carga y Preprocesamiento de Datos** - Gestión de archivos CSV/XLSX con limpieza automática
2. **Entrenamiento de Modelos** - Selección y configuración de 4+ modelos ML/DL
3. **Visualización de Resultados** - Mapas interactivos, gráficos y KPIs
4. **Análisis Temporal** - Series temporales y tendencias de desastres

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

## 📁 Estructura del Proyecto

```
Proyecto_SRG/
├── app.py                          # Aplicación principal Streamlit
├── requirements.txt                # Dependencias Python
├── run.sh                         # Script de ejecución
├── README.md                      # Este archivo
├── src/
│   ├── __init__.py               # Inicialización del paquete
│   ├── data_processor.py         # Procesamiento de datos
│   ├── models.py                 # Modelos ML/DL
│   └── visualizations.py         # Visualizaciones
├── data/                         # Carpeta para datos
├── models/                       # Carpeta para guardar modelos
└── utils/                        # Utilidades adicionales
```

## 🚀 Instalación y Ejecución

### 1. Requisitos Previos
- Python 3.10 o superior
- pip (gestor de paquetes Python)
- Git (opcional, para clonar el repositorio)

### 2. Instalación

#### Opción A: Script Automático (Recomendado en Linux/Mac)
```bash
chmod +x run.sh
./run.sh
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

### 3. Acceso a la Aplicación
La aplicación se abrirá automáticamente en:
```
http://localhost:8501
```

Si no se abre, accede manualmente en tu navegador.

## 📊 Componentes Principales

### 1. Barra Lateral (Panel de Control)
**Filtros disponibles:**
- Provincia (multiselección)
- Rango de años (slider)
- Tipos de evento (multiselección)
- Botones de control (Restablecer, Aplicar)

### 2. Pestaña 1: Gestión de Datos
**Funcionalidades:**
- ✅ Carga de archivos (Drag & Drop)
- 📊 Vista previa de datos
- 🧹 Preprocesamiento automático
- 📥 Descarga de datos limpios

**Estádísticas mostradas:**
- Total de registros
- Número de columnas
- Valores nulos
- Registros duplicados

### 3. Pestaña 2: Entrenamiento y Análisis
**Modelos disponibles:**
1. **XGBoost** (Modelo principal)
   - Parámetros: n_estimators (50-300), max_depth (3-15)
   
2. **Random Forest**
   - Parámetros: n_estimators (50-300), max_depth (3-15)
   
3. **LSTM** (Series temporales)
   - Parámetros: epochs (10-100), batch_size (16-128)
   
4. **Gradient Boosting** (XGBoost alternativo)
   - Parámetros: n_estimators (50-300), learning_rate (0.01-0.5)

**Métricas de evaluación:**
- Accuracy
- Precision
- Recall
- F1-Score

### 4. Pestaña 3: Visualización de Resultados
**Componentes:**
- 📊 KPIs (Total de Eventos, Personas Afectadas, Viviendas Dañadas)
- 🗺️ Mapas interactivos:
  - Heatmap de distribución de eventos
  - Mapa de riesgo por provincia
- 📈 Gráficos:
  - Frecuencia de eventos por tipo
  - Provincias más afectadas
  - Tendencia temporal
- 📋 Tabla de datos filtrada
- 📥 Descarga de resultados

## 📖 Casos de Uso

### CU-01: Carga de Datos
```
Usuario → Selecciona archivo → Sistema carga y valida → Muestra vista previa
```

### CU-02: Depuración de Datos
```
Usuario → Solicita preprocesamiento → Sistema limpia y elimina duplicados → Muestra resumen
```

### CU-03: Entrenamiento de Modelos
```
Usuario → Selecciona modelos → Define hiperparámetros → Sistema entrena → Muestra métricas
```

### CU-04: Visualización de Resultados
```
Usuario → Aplica filtros → Sistema genera visualizaciones → Usuario descarga resultados
```

## 💾 Formato de Datos Esperados

El archivo CSV/XLSX debe contener columnas similares a:

```
Fecha         | Provincia    | Tipo_Evento   | Personas_Afectadas | Viviendas_Dañadas | Latitude | Longitude
2015-01-15    | Pichincha    | Inundación    | 150                | 45                | -0.35    | -78.52
2015-02-20    | Guayas       | Deslizamiento | 250                | 120               | -2.20    | -79.89
...
```

### Columnas requeridas mínimas:
- Una columna de fecha/año
- Una columna de provincia/región
- Una columna de tipo de evento

### Columnas opcionales:
- Personas afectadas
- Viviendas dañadas
- Latitude/Longitude (para mapas precisos)

## ⚙️ Configuración Avanzada

### Variables de Entorno
```bash
# Puerto personalizado para Streamlit
export STREAMLIT_SERVER_PORT=8501

# Modo de depuración
export STREAMLIT_LOGGER_LEVEL=debug
```

### Hiperparámetros Predeterminados
Están configurados en `src/models.py`:
- XGBoost: n_estimators=150, max_depth=8
- Random Forest: n_estimators=100, max_depth=10
- LSTM: epochs=50, batch_size=32
- Gradient Boosting: n_estimators=200, learning_rate=0.05

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit --upgrade
```

### Error: "No hay datos disponibles"
- Asegúrate de cargar un archivo en la pestaña "Gestión de Datos"
- El archivo debe estar en formato CSV o XLSX

### Error: "TensorFlow issues"
En Windows puede requerir instalación especial:
```bash
pip install tensorflow-cpu  # Versión CPU más ligera
```

### El mapa no carga
- Verifica tu conexión a internet (Folium requiere OpenStreetMap)
- Intenta actualizar folium: `pip install folium --upgrade`

## 📊 Ejemplo de Uso

1. **Iniciar la aplicación**
   ```bash
   streamlit run app.py
   ```

2. **Cargar datos de demostración**
   - Click en "📂 Cargar Datos de Demostración" en la pestaña "Gestión de Datos"

3. **Limpiar datos**
   - Click en "🧹 Preprocesar/Limpiar Datos"
   - Verificar resumen de limpieza

4. **Entrenar modelos**
   - Ir a pestaña "Entrenamiento y Análisis"
   - Seleccionar modelos y ajustar hiperparámetros
   - Click en "🚀 Entrenar Modelos"

5. **Visualizar resultados**
   - Ir a pestaña "Visualización de Resultados"
   - Aplicar filtros desde la barra lateral
   - Explorar mapas, gráficos y estadísticas
   - Descargar resultados

## 📈 Mejoras Futuras

- [ ] Integración con bases de datos en tiempo real
- [ ] Predicciones en tiempo real
- [ ] Exportación de reportes PDF
- [ ] Validación cruzada k-fold
- [ ] Tuning automático de hiperparámetros
- [ ] Análisis de importancia de características
- [ ] Modelos ensemble combinados
- [ ] API REST para integraciones externas

## 📄 Documentación Técnica

### Estructura de Clases

#### DataProcessor
```python
processor = DataProcessor()
processor.load_data(file_path="datos.csv")
df_clean, stats = processor.clean_data(df)
df_processed = processor.preprocess_for_modeling(df_clean)
```

#### MLModels
```python
models = MLModels()
models.prepare_data(X, y)
metrics = models.train_xgboost(params={'n_estimators': 150})
predictions = models.predict('xgboost', X_new)
```

#### Visualizations
```python
kpis = Visualizations.create_kpi_cards(df)
fig = Visualizations.create_event_frequency_chart(df)
mapa = Visualizations.create_risk_map(risk_data)
```

## 👥 Usuarios Objetivo

- 🏛️ **SGR** (Secretaría de Gestión de Riesgos)
- 🏘️ **GADs** (Gobiernos Autónomos Descentralizados)
- 👨‍💼 Analistas de riesgos
- 🔬 Investigadores
- 📊 Especialistas en datos

## ⚖️ Licencia

Este proyecto está disponible bajo licencia MIT.

## 📞 Soporte

Para reportar bugs o solicitar features:
1. Crear un Issue en el repositorio
2. Describir el problema detalladamente
3. Incluir pasos para reproducir

## 🙏 Agradecimientos

Desarrollado siguiendo las especificaciones técnicas del documento de diseño de solución para predicción de desastres naturales en Ecuador.

Tecnologías utilizadas:
- Streamlit Team
- XGBoost Contributors
- TensorFlow/Keras Team
- Pandas Team
- Plotly
- Folium

---

**Versión:** 1.0.0  
**Última actualización:** 2024  
**Estado:** ✅ Funcional y listo para producción
