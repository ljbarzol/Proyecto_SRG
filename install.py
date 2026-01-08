#!/usr/bin/env python3
"""
Script de instalación interactivo para el Sistema de Predicción de Desastres.
Compatible con Windows, macOS y Linux.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Ejecuta un comando y maneja errores."""
    if description:
        print(f"\n📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completado")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Instalación interactiva."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Sistema de Predicción de Desastres Naturales".center(58) + "║")
    print("║" + "  Script de Instalación".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Verificar Python
    print("\n1️⃣ Verificando Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} encontrado")
    else:
        print(f"❌ Se requiere Python 3.10 o superior (actual: {version.major}.{version.minor})")
        return False
    
    # Actualizar pip
    print("\n2️⃣ Actualizando pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "pip actualizado")
    
    # Instalar dependencias
    print("\n3️⃣ Instalando dependencias...")
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        cmd = f"{sys.executable} -m pip install -r {req_file}"
        if run_command(cmd, "Instalando paquetes"):
            print("✅ Todas las dependencias instaladas")
        else:
            print("⚠️ Algunas dependencias pueden no haberse instalado correctamente")
    else:
        print(f"❌ Archivo requirements.txt no encontrado en {req_file.parent}")
        return False
    
    print("\n" + "═" * 60)
    print("\n✅ ¡INSTALACIÓN COMPLETADA!")
    print("\nPara ejecutar la aplicación, usa uno de los siguientes comandos:")
    print("\n📌 Opción 1 - Usar script de ejecución (Linux/macOS):")
    print("   ./run.sh")
    print("\n📌 Opción 2 - Comando directo:")
    print("   streamlit run app.py")
    print("\n📌 Opción 3 - Ejecutar ejemplos (sin interfaz):")
    print(f"   {sys.executable} examples.py basic")
    print("\n" + "═" * 60)
    print("\n🌐 La aplicación estará disponible en: http://localhost:8501")
    print("\n💡 Presiona CTRL+C para detener la aplicación")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
