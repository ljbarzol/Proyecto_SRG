# 🚀 Inicio Rápido - Sistema de Predicción de Desastres

## ⚡ En 5 minutos

### Paso 1: Instalación de Dependencias (2 min)

**Opción A - Automático (Recomendado):**
```bash
python install.py
```

**Opción B - Manual:**
```bash
pip install -r requirements.txt
```

### Paso 2: Ejecutar la Aplicación (1 min)

**En Linux/macOS:**
```bash
./run.sh
```

**En Windows o en cualquier lugar:**
```bash
streamlit run app.py
```

### Paso 3: Abrir en el Navegador (1 min)

Accede a: **http://localhost:8501**

¡Eso es todo! La aplicación debería abrirse en tu navegador.

---

## 📊 Primeros Pasos en la Interfaz

### 1. Cargar Datos (Pestaña "Gestión de Datos")
- Haz clic en **"📂 Cargar Datos de Demostración"** para empezar con datos de ejemplo
- O sube tu propio archivo CSV/XLSX con datos de eventos

### 2. Limpiar Datos
- Haz clic en **"🧹 Preprocesar/Limpiar Datos"**
- Verifica el resumen de registros válidos/descartados

### 3. Entrenar Modelos (Pestaña "Entrenamiento y Análisis")
- Selecciona los modelos que quieres entrenar (XGBoost, Random Forest, etc.)
- Ajusta los hiperparámetros si lo deseas
- Haz clic en **"🚀 Entrenar Modelos"**
- Observa las métricas (Accuracy, Precision, Recall, F1)

### 4. Visualizar Resultados (Pestaña "Visualización de Resultados")
- Aplica filtros desde la barra lateral izquierda
- Explora los mapas interactivos
- Analiza los gráficos y estadísticas
- Descarga los resultados

---

## 🔧 Solución Rápida de Problemas

### "No se instala TensorFlow"
```bash
# Usar versión CPU más ligera
pip install tensorflow-cpu
```

### "Puerto 8501 ya está en uso"
```bash
# Usar un puerto diferente
streamlit run app.py --server.port 8502
```

### "Error de módulos no encontrados"
```bash
# Reinstalar todas las dependencias
pip install -r requirements.txt --force-reinstall
```

### "Mapa no carga"
- Verifica tu conexión a internet
- Los mapas necesitan descargar datos de OpenStreetMap

---

## 📁 Archivos Principales

| Archivo | Descripción |
|---------|------------|
| `app.py` | 🎯 Aplicación Streamlit principal |
| `src/data_processor.py` | 📊 Carga y procesamiento de datos |
| `src/models.py` | 🤖 Modelos ML/DL |
| `src/visualizations.py` | 📈 Gráficos y mapas |
| `examples.py` | 📚 Ejemplos de uso sin interfaz |
| `requirements.txt` | 📦 Dependencias |

---

## 🎓 Ejemplos desde Terminal

Sin necesidad de la interfaz Streamlit:

```bash
# Flujo completo de trabajo
python examples.py basic

# Filtrado de datos
python examples.py filter

# Comparación de modelos
python examples.py comparison

# Visualizaciones
python examples.py visual

# Todos los ejemplos
python examples.py all
```

---

## 🔗 Recursos Útiles

- 📖 **Documentación completa:** Ver [README_APP.md](README_APP.md)
- 📚 **Documentación técnica:** Ver sección "📄 Documentación Técnica" en README_APP.md
- 💡 **Código de ejemplo:** Ver archivo `examples.py`

---

## ✨ Características Principales

✅ **3 Pestañas Funcionales:**
1. 📁 Gestión de Datos (carga, limpieza, preprocesamiento)
2. 🤖 Entrenamiento y Análisis (4+ modelos ML/DL)
3. 📊 Visualización de Resultados (mapas, gráficos, KPIs)

✅ **Panel de Control (Sidebar):**
- Filtros de Provincia, Año y Tipo de Evento
- Botones de control

✅ **4+ Modelos Disponibles:**
- 🌳 XGBoost (Modelo principal)
- 🌲 Random Forest
- 🧠 LSTM (Series temporales)
- ⚡ Gradient Boosting

✅ **Visualizaciones Interactivas:**
- 🗺️ Mapas de riesgo y heatmaps
- 📊 Gráficos de barras, líneas y comparativos
- 📈 Indicadores clave (KPIs)

---

## 🐛 Necesitas Ayuda?

1. **Lee el README completo:** [README_APP.md](README_APP.md)
2. **Revisa los ejemplos:** `examples.py`
3. **Verifica logs:** La consola mostrará errores detallados
4. **Actualiza dependencias:** `pip install --upgrade -r requirements.txt`

---

## 📝 Notas Importantes

- La **primera ejecución** puede tardar unos minutos en descargar modelos
- Los **mapas requieren conexión a internet** (OpenStreetMap)
- Los **datos de demostración son aleatorios** y solo para pruebas
- Para **datos reales**, asegúrate de que tengan el formato correcto

---

**¡Disfrutalo! 🎉**

¿Preguntas? Consulta [README_APP.md](README_APP.md) para documentación completa.
