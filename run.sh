#!/bin/bash
# Script de instalación y ejecución de la aplicación

echo "🚀 Sistema de Predicción de Desastres Naturales en Ecuador"
echo "=========================================================="
echo ""

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado"
    exit 1
fi

echo "✅ Python 3 encontrado"
python3 --version

echo ""
echo "📦 Instalando dependencias..."
pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "✅ Dependencias instaladas correctamente"
else
    echo "❌ Error al instalar dependencias"
    exit 1
fi

echo ""
echo "🎯 Iniciando aplicación Streamlit..."
echo "La aplicación estará disponible en: http://localhost:8501"
echo ""

streamlit run app.py
