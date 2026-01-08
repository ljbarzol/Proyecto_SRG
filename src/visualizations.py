"""
Módulo de visualizaciones: gráficos, mapas y KPIs.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import streamlit as st
from typing import Dict, List, Tuple


class Visualizations:
    """Clase para crear visualizaciones interactivas."""
    
    # Coordenadas aproximadas del centro de Ecuador
    ECUADOR_CENTER = [-1.8312, -78.1834]
    
    # Coordenadas de provincias ecuatorianas (aproximadas)
    PROVINCIA_COORDS = {
        'Pichincha': [-0.3522, -78.5249],
        'Guayas': [-2.2045, -79.8853],
        'Azuay': [-3.0197, -78.6147],
        'Tungurahua': [-1.2264, -78.6365],
        'Imbabura': [0.3522, -78.1200],
        'Carchi': [0.9021, -77.3768],
        'Sucumbíos': [0.1842, -76.3854],
        'Orellana': [-0.4615, -76.9914],
        'Napo': [-0.9796, -77.4914],
        'Pastaza': [-1.3399, -78.1191],
        'Morona Santiago': [-2.2985, -78.1493],
        'Zamora Chinchipe': [-4.0521, -78.9405],
        'El Oro': [-3.2981, -79.9667],
        'Loja': [-3.9899, -79.2060],
        'Cotopaxi': [-0.9264, -78.6350],
        'Los Ríos': [-1.2100, -79.4600],
        'Santa Elena': [-2.2368, -80.3740],
        'Manabí': [-0.9565, -80.1234],
        'Santo Domingo': [-0.2449, -79.1733],
        'Chimborazo': [-1.6705, -78.6460],
    }
    
    @staticmethod
    def create_kpi_cards(df: pd.DataFrame) -> Dict[str, int]:
        """
        Crea métricas clave (KPIs).
        
        Args:
            df: DataFrame con datos
            
        Returns:
            Diccionario con KPIs
        """
        kpis = {
            'total_eventos': len(df),
            'años_cubiertos': df.apply(lambda row: pd.to_datetime(row, errors='coerce')).min() if any(df.dtypes == 'datetime64[ns]') else 'N/A',
        }
        
        # Intentar obtener personas afectadas si existe la columna
        affected_cols = [col for col in df.columns if 'afectad' in col.lower() or 'personas' in col.lower()]
        if affected_cols:
            kpis['personas_afectadas'] = int(df[affected_cols[0]].sum()) if df[affected_cols[0]].dtype in ['int64', 'float64'] else 0
        else:
            kpis['personas_afectadas'] = 0
        
        # Intentar obtener viviendas dañadas
        house_cols = [col for col in df.columns if 'vivienda' in col.lower() or 'dañada' in col.lower()]
        if house_cols:
            kpis['viviendas_danadas'] = int(df[house_cols[0]].sum()) if df[house_cols[0]].dtype in ['int64', 'float64'] else 0
        else:
            kpis['viviendas_danadas'] = 0
        
        return kpis
    
    @staticmethod
    def create_event_frequency_chart(df: pd.DataFrame) -> go.Figure:
        """
        Gráfico de barras: Frecuencia de eventos por tipo.
        
        Args:
            df: DataFrame con datos
            
        Returns:
            Gráfico Plotly
        """
        # Detectar columna de tipo de evento
        event_cols = [col for col in df.columns if 'evento' in col.lower() or 'tipo' in col.lower() or 'peligro' in col.lower()]
        
        if not event_cols:
            return go.Figure().add_annotation(text="No hay datos de eventos disponibles")
        
        event_col = event_cols[0]
        event_counts = df[event_col].value_counts().reset_index()
        event_counts.columns = ['evento', 'frecuencia']
        
        fig = px.bar(
            event_counts,
            x='evento',
            y='frecuencia',
            labels={'evento': 'Tipo de Evento', 'frecuencia': 'Frecuencia'},
            title='Frecuencia de Eventos por Tipo',
            color='frecuencia',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_tickangle=-45,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_timeline_chart(df: pd.DataFrame) -> go.Figure:
        """
        Gráfico de línea: Tendencia de desastres a lo largo del tiempo.
        
        Args:
            df: DataFrame con datos
            
        Returns:
            Gráfico Plotly
        """
        # Detectar columna de fecha/año
        date_cols = [col for col in df.columns if any(x in col.lower() for x in ['fecha', 'año', 'year', 'date', 'time'])]
        
        if not date_cols:
            return go.Figure().add_annotation(text="No hay datos temporales disponibles")
        
        date_col = date_cols[0]
        
        try:
            # Convertir a datetime si es necesario
            if df[date_col].dtype == 'object':
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                df_temp['year'] = df_temp[date_col].dt.year
            else:
                df_temp = df.copy()
                if 'year' not in df_temp.columns:
                    df_temp['year'] = pd.to_datetime(df_temp[date_col], errors='coerce').dt.year
            
            events_by_year = df_temp.groupby('year').size()
            
            fig = px.line(
                x=events_by_year.index,
                y=events_by_year.values,
                labels={'x': 'Año', 'y': 'Número de Eventos'},
                title='Tendencia de Desastres por Año',
                markers=True
            )
            
            fig.update_layout(
                height=400,
                hovermode='x unified',
                plot_bgcolor='rgba(240,240,240,0.5)'
            )
            
            return fig
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error al procesar fechas: {str(e)}")
    
    @staticmethod
    def create_risk_map(risk_data: Dict[str, float] = None) -> folium.Map:
        """
        Crea mapa interactivo de riesgo de Ecuador.
        
        Args:
            risk_data: Diccionario con datos de riesgo por provincia
                       ej: {'Pichincha': 0.8, 'Guayas': 0.6}
        
        Returns:
            Mapa Folium
        """
        # Crear mapa centrado en Ecuador
        m = folium.Map(
            location=Visualizations.ECUADOR_CENTER,
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Si no hay datos de riesgo, generar datos aleatorios para demostración
        if risk_data is None:
            risk_data = {
                prov: np.random.uniform(0.2, 0.9) 
                for prov in Visualizations.PROVINCIA_COORDS.keys()
            }
        
        # Añadir marcadores por provincia
        for provincia, coords in Visualizations.PROVINCIA_COORDS.items():
            if provincia in risk_data:
                risk_level = risk_data[provincia]
                
                # Asignar color según nivel de riesgo
                if risk_level >= 0.7:
                    color = 'red'
                    risk_text = 'Alto'
                elif risk_level >= 0.4:
                    color = 'orange'
                    risk_text = 'Medio'
                else:
                    color = 'green'
                    risk_text = 'Bajo'
                
                popup_text = f"""
                <b>{provincia}</b><br>
                Nivel de Riesgo: {risk_text}<br>
                Score: {risk_level:.2f}
                """
                
                folium.CircleMarker(
                    location=coords,
                    radius=10,
                    popup=folium.Popup(popup_text, max_width=250),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
        
        return m
    
    @staticmethod
    def create_heatmap(df: pd.DataFrame) -> folium.Map:
        """
        Crea mapa de calor basado en eventos.
        
        Args:
            df: DataFrame con datos de eventos
            
        Returns:
            Mapa Folium con heatmap
        """
        m = folium.Map(
            location=Visualizations.ECUADOR_CENTER,
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Obtener columnas de coordenadas si existen
        lat_cols = [col for col in df.columns if 'lat' in col.lower()]
        lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'lng' in col.lower()]
        
        if lat_cols and lon_cols:
            # Usar coordenadas reales si existen
            heat_data = [
                [row[lat_cols[0]], row[lon_cols[0]]]
                for _, row in df.iterrows()
                if pd.notna(row[lat_cols[0]]) and pd.notna(row[lon_cols[0]])
            ]
        else:
            # Generar datos de heatmap basados en provincias
            heat_data = []
            for _, row in df.iterrows():
                provincia_cols = [col for col in df.columns if 'provincia' in col.lower()]
                if provincia_cols:
                    prov = row[provincia_cols[0]]
                    if prov in Visualizations.PROVINCIA_COORDS:
                        coords = Visualizations.PROVINCIA_COORDS[prov]
                        # Añadir ruido para dispersar puntos
                        heat_data.append([
                            coords[0] + np.random.uniform(-0.5, 0.5),
                            coords[1] + np.random.uniform(-0.5, 0.5)
                        ])
        
        if heat_data:
            HeatMap(heat_data, radius=20, blur=15, max_zoom=1).add_to(m)
        
        return m
    
    @staticmethod
    def create_province_comparison(df: pd.DataFrame) -> go.Figure:
        """
        Gráfico comparativo de provincias.
        
        Args:
            df: DataFrame con datos
            
        Returns:
            Gráfico Plotly
        """
        # Detectar columna de provincia
        prov_cols = [col for col in df.columns if 'provincia' in col.lower()]
        
        if not prov_cols:
            return go.Figure().add_annotation(text="No hay datos de provincias disponibles")
        
        prov_col = prov_cols[0]
        province_counts = df[prov_col].value_counts().head(10).reset_index()
        province_counts.columns = ['provincia', 'eventos']
        
        fig = px.bar(
            province_counts,
            x='eventos',
            y='provincia',
            orientation='h',
            labels={'eventos': 'Número de Eventos', 'provincia': 'Provincia'},
            title='Top 10 Provincias Afectadas',
            color='eventos',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig
    
    @staticmethod
    def create_model_comparison_chart(metrics: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Gráfico comparativo de rendimiento de modelos.
        
        Args:
            metrics: Diccionario con métricas de múltiples modelos
        
        Returns:
            Gráfico Plotly
        """
        model_names = list(metrics.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        
        fig = make_subplots(
            rows=1, cols=len(metrics_names),
            specs=[[{'type': 'bar'} for _ in metrics_names]],
            subplot_titles=metrics_names
        )
        
        for i, metric in enumerate(metrics_names, 1):
            values = [metrics[model].get(metric, 0) for model in model_names]
            fig.add_trace(
                go.Bar(x=model_names, y=values, name=metric, showlegend=False),
                row=1, col=i
            )
            fig.update_xaxes(tickangle=-45, row=1, col=i)
            fig.update_yaxes(range=[0, 1], row=1, col=i)
        
        fig.update_layout(height=400, title_text="Comparación de Rendimiento de Modelos")
        
        return fig
    
    @staticmethod
    def create_vulnerability_prediction_map(df: pd.DataFrame, selected_provinces: List[str] = None) -> folium.Map:
        """
        Crea mapa con predicciones de vulnerabilidad.
        Solo muestra las zonas seleccionadas en los filtros.
        
        Args:
            df: DataFrame con columna 'Vulnerabilidad_Predicha' y coordenadas
            selected_provinces: Lista de provincias seleccionadas (None = todas)
            
        Returns:
            Mapa Folium con marcadores coloreados por vulnerabilidad
        """
        # Crear mapa centrado en Ecuador
        m = folium.Map(
            location=Visualizations.ECUADOR_CENTER,
            zoom_start=7 if selected_provinces and len(selected_provinces) == 1 else 6,
            tiles='OpenStreetMap'
        )
        
        # Obtener columnas necesarias
        lat_cols = [col for col in df.columns if 'lat' in col.lower()]
        lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'lng' in col.lower()]
        prov_cols = [col for col in df.columns if 'provincia' in col.lower()]
        event_cols = [col for col in df.columns if 'evento' in col.lower() or 'tipo' in col.lower()]
        
        # Mapeo de colores por vulnerabilidad
        color_map = {
            'Alta': 'red',
            'Media': 'orange',
            'Baja': 'green'
        }
        
        # Si hay coordenadas reales, usarlas
        if lat_cols and lon_cols:
            for _, row in df.iterrows():
                if pd.notna(row[lat_cols[0]]) and pd.notna(row[lon_cols[0]]):
                    vuln = row.get('Vulnerabilidad_Predicha', 'Media')
                    color = color_map.get(vuln, 'gray')
                    
                    # Información del popup
                    popup_html = f"""
                    <div style='width: 200px'>
                        <h4 style='margin:0; color: {color}'>Vulnerabilidad: {vuln}</h4>
                        <hr style='margin: 5px 0'>
                    """
                    
                    if prov_cols:
                        popup_html += f"<b>Provincia:</b> {row[prov_cols[0]]}<br>"
                    if event_cols:
                        popup_html += f"<b>Tipo:</b> {row[event_cols[0]]}<br>"
                    
                    # Agregar otras columnas relevantes
                    for col in ['Personas_Afectadas', 'Viviendas_Dañadas', 'Canton', 'Parroquia']:
                        if col in df.columns and pd.notna(row.get(col)):
                            popup_html += f"<b>{col}:</b> {row[col]}<br>"
                    
                    popup_html += "</div>"
                    
                    folium.CircleMarker(
                        location=[row[lat_cols[0]], row[lon_cols[0]]],
                        radius=6,
                        popup=folium.Popup(popup_html, max_width=250),
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(m)
        
        # Si no hay coordenadas, usar centros provinciales
        elif prov_cols:
            # Agrupar por provincia y vulnerabilidad
            grouped = df.groupby([prov_cols[0], 'Vulnerabilidad_Predicha']).size().reset_index(name='count')
            
            for _, row in grouped.iterrows():
                provincia = str(row[prov_cols[0]])
                vuln = row['Vulnerabilidad_Predicha']
                count = row['count']
                
                if provincia in Visualizations.PROVINCIA_COORDS:
                    coords = Visualizations.PROVINCIA_COORDS[provincia]
                    color = color_map.get(vuln, 'gray')
                    
                    # Solo mostrar si está en las provincias seleccionadas o no hay filtro
                    if not selected_provinces or provincia in selected_provinces:
                        popup_html = f"""
                        <div style='width: 200px'>
                            <h4 style='margin:0'>{provincia}</h4>
                            <hr style='margin: 5px 0'>
                            <b>Vulnerabilidad:</b> <span style='color: {color}'>{vuln}</span><br>
                            <b>Zonas detectadas:</b> {count}<br>
                        </div>
                        """
                        
                        # Tamaño proporcional al conteo
                        radius = min(15 + (count / 10), 40)
                        
                        folium.CircleMarker(
                            location=coords,
                            radius=radius,
                            popup=folium.Popup(popup_html, max_width=250),
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.6,
                            weight=3
                        ).add_to(m)
        
        # Agregar leyenda
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; 
                    width: 180px; height: 120px; 
                    background-color: white; 
                    border:2px solid grey; 
                    z-index:9999; 
                    font-size:14px;
                    padding: 10px">
            <p style="margin: 0; font-weight: bold;">Nivel de Vulnerabilidad</p>
            <p style="margin: 5px 0;"><span style="color: red;">●</span> Alta</p>
            <p style="margin: 5px 0;"><span style="color: orange;">●</span> Media</p>
            <p style="margin: 5px 0;"><span style="color: green;">●</span> Baja</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
