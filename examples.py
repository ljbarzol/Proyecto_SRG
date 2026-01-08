"""
Script de ejemplo para uso programático del sistema (sin Streamlit).
Útil para automatización y scripts batch.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar ruta
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processor import DataProcessor
from src.models import MLModels
from src.visualizations import Visualizations
from utils.helpers import generate_sample_dataset, export_metrics_to_json


def example_basic_workflow():
    """
    Ejemplo básico del flujo completo de trabajo.
    """
    print("=" * 60)
    print("EJEMPLO: Flujo Básico del Sistema")
    print("=" * 60)
    
    # 1. Generar datos de demostración
    print("\n1️⃣ Generando datos de demostración...")
    df = generate_sample_dataset(500)
    print(f"✅ {len(df)} registros generados")
    print(f"Columnas: {list(df.columns)}")
    
    # 2. Procesar datos
    print("\n2️⃣ Procesando datos...")
    processor = DataProcessor()
    df_clean, stats = processor.clean_data(df)
    print(f"✅ Registros iniciales: {stats['registros_iniciales']}")
    print(f"✅ Registros finales: {stats['registros_finales']}")
    print(f"✅ Registros descartados: {stats['registros_descartados']}")
    
    # 3. Preparar para modelado
    print("\n3️⃣ Preparando datos para modelado...")
    df_processed = processor.preprocess_for_modeling(df_clean, target_col=None)
    print(f"✅ Datos procesados: {df_processed.shape}")
    
    # 4. Entrenar modelos
    print("\n4️⃣ Entrenando modelos...")
    numeric_features = df_processed.select_dtypes(include=[np.number]).columns
    X = df_processed[numeric_features].fillna(0).values
    y = np.random.randint(0, 2, len(X))  # Etiquetas aleatorias para demostración
    
    ml_models = MLModels()
    ml_models.prepare_data(X, y, test_size=0.2)
    
    # Entrenar XGBoost
    print("   • Entrenando XGBoost...")
    xgb_metrics = ml_models.train_xgboost()
    print(f"     Accuracy: {xgb_metrics['accuracy']:.4f}")
    
    # Entrenar Random Forest
    print("   • Entrenando Random Forest...")
    rf_metrics = ml_models.train_random_forest()
    print(f"     Accuracy: {rf_metrics['accuracy']:.4f}")
    
    # Entrenar Gradient Boosting
    print("   • Entrenando Gradient Boosting...")
    gb_metrics = ml_models.train_gradient_boosting()
    print(f"     Accuracy: {gb_metrics['accuracy']:.4f}")
    
    # 5. Obtener todas las métricas
    print("\n5️⃣ Resultados de Entrenamiento:")
    all_metrics = ml_models.get_all_metrics()
    for model_name, metrics in all_metrics.items():
        print(f"\n   {model_name.upper()}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"     • {metric_name}: {value:.4f}")
    
    # 6. Exportar métricas
    print("\n6️⃣ Exportando métricas...")
    export_metrics_to_json(all_metrics, "model_metrics.json")
    print("✅ Métricas guardadas en 'model_metrics.json'")
    
    # 7. Crear visualizaciones
    print("\n7️⃣ Generando visualizaciones...")
    kpis = Visualizations.create_kpi_cards(df)
    print(f"✅ KPIs calculados: {len(kpis)}")
    for kpi_name, value in kpis.items():
        print(f"   • {kpi_name}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Ejemplo completado exitosamente")
    print("=" * 60)


def example_data_filtering():
    """
    Ejemplo de filtrado de datos.
    """
    print("\n" + "=" * 60)
    print("EJEMPLO: Filtrado de Datos")
    print("=" * 60)
    
    # Generar datos
    df = generate_sample_dataset(1000)
    processor = DataProcessor()
    
    print(f"\nDataset original: {len(df)} registros")
    
    # Aplicar filtros
    filters = {
        'provincia': ['Pichincha', 'Guayas'],
        'year_range': (2015, 2017),
        'event_types': ['Inundación', 'Deslizamiento']
    }
    
    df_filtered = processor.filter_data(df, filters)
    
    print(f"Dataset filtrado: {len(df_filtered)} registros")
    print(f"\nFiltros aplicados:")
    print(f"  • Provincias: {filters['provincia']}")
    print(f"  • Años: {filters['year_range']}")
    print(f"  • Eventos: {filters['event_types']}")
    
    print("\n✅ Primeros registros filtrados:")
    print(df_filtered.head())


def example_model_comparison():
    """
    Ejemplo de comparación de modelos.
    """
    print("\n" + "=" * 60)
    print("EJEMPLO: Comparación de Modelos")
    print("=" * 60)
    
    # Generar datos sintéticos
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)
    
    ml_models = MLModels()
    ml_models.prepare_data(X, y, test_size=0.3)
    
    print("\nEntrenando múltiples modelos...\n")
    
    # Entrenar todos los modelos
    models = {
        'xgboost': ml_models.train_xgboost,
        'random_forest': ml_models.train_random_forest,
        'gradient_boosting': ml_models.train_gradient_boosting,
    }
    
    for model_name, train_func in models.items():
        metrics = train_func()
        print(f"{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}\n")
    
    # Comparación visual
    print("=" * 60)
    print("Resumen Comparativo:")
    print("=" * 60)
    
    all_metrics = ml_models.get_all_metrics()
    metrics_df = pd.DataFrame(all_metrics).T
    print(metrics_df.round(4).to_string())
    
    # Identificar mejor modelo
    best_model = max(all_metrics.items(), key=lambda x: x[1].get('f1', 0))
    print(f"\n🏆 Mejor modelo: {best_model[0].upper()} (F1: {best_model[1].get('f1', 0):.4f})")


def example_visualization_demo():
    """
    Ejemplo de generación de visualizaciones.
    """
    print("\n" + "=" * 60)
    print("EJEMPLO: Visualizaciones")
    print("=" * 60)
    
    df = generate_sample_dataset(300)
    
    # Generar KPIs
    print("\n1. Indicadores Clave (KPIs):")
    kpis = Visualizations.create_kpi_cards(df)
    for kpi_name, value in kpis.items():
        print(f"   • {kpi_name}: {value}")
    
    # Generar gráficos
    print("\n2. Generando gráficos...")
    
    fig_events = Visualizations.create_event_frequency_chart(df)
    print("   ✅ Gráfico de frecuencia de eventos")
    
    fig_provinces = Visualizations.create_province_comparison(df)
    print("   ✅ Gráfico comparativo de provincias")
    
    fig_timeline = Visualizations.create_timeline_chart(df)
    print("   ✅ Gráfico de línea temporal")
    
    # Generar mapas
    print("\n3. Generando mapas...")
    
    risk_map = Visualizations.create_risk_map()
    print("   ✅ Mapa de riesgo")
    
    heat_map = Visualizations.create_heatmap(df)
    print("   ✅ Mapa de calor")
    
    print("\n✅ Todas las visualizaciones generadas correctamente")


if __name__ == "__main__":
    """
    Ejecuta ejemplos de uso del sistema.
    
    Opciones:
    - python examples.py basic       : Flujo básico completo
    - python examples.py filter      : Ejemplo de filtrado
    - python examples.py comparison  : Comparación de modelos
    - python examples.py visual      : Demostración de visualizaciones
    - python examples.py all         : Ejecutar todos los ejemplos
    """
    
    import sys
    
    examples = {
        'basic': example_basic_workflow,
        'filter': example_data_filtering,
        'comparison': example_model_comparison,
        'visual': example_visualization_demo,
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
        if choice == 'all':
            for example_func in examples.values():
                example_func()
        elif choice in examples:
            examples[choice]()
        else:
            print(f"Opción no válida: {choice}")
            print(f"Opciones válidas: {', '.join(examples.keys())}, all")
    else:
        print("Sistema de Predicción de Desastres Naturales - Ejemplos")
        print("=" * 60)
        print("\nUso: python examples.py [opción]")
        print("\nOpciones disponibles:")
        for key in examples.keys():
            print(f"  • {key}")
        print("  • all (ejecuta todos los ejemplos)")
        print("\nEjemplo: python examples.py basic")
