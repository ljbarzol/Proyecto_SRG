#!/usr/bin/env python3
"""
Script de verificación de la instalación.
Comprueba que todas las dependencias estén correctamente instaladas.
"""
import sys
from importlib import import_module
from pathlib import Path
import subprocess


def check_python_version():
    """Verifica la versión de Python."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (requiere 3.10+)")
        return False


def check_packages():
    """Verifica que los paquetes requeridos estén instalados."""
    packages = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-Learn',
        'xgboost': 'XGBoost',
        'tensorflow': 'TensorFlow',
        'plotly': 'Plotly',
        'folium': 'Folium',
        'geopandas': 'GeoPandas',
    }
    
    print("\n📦 Verificando paquetes:")
    all_ok = True
    
    for package, name in packages.items():
        try:
            module = import_module(package)
            version = getattr(module, '__version__', 'desconocida')
            print(f"   ✅ {name}: {version}")
        except ImportError:
            print(f"   ❌ {name}: NO INSTALADO")
            all_ok = False
    
    return all_ok


def check_project_structure():
    """Verifica la estructura del proyecto."""
    print("\n📁 Verificando estructura:")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'config.py',
        'examples.py',
    ]
    
    required_dirs = [
        'src',
        'data',
        'models',
        'utils',
        '.streamlit',
    ]
    
    all_ok = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}: NO ENCONTRADO")
            all_ok = False
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/: NO ENCONTRADO")
            all_ok = False
    
    return all_ok


def check_modules():
    """Verifica que los módulos del proyecto existan."""
    print("\n🔧 Verificando módulos del proyecto:")
    
    modules = [
        'src/data_processor.py',
        'src/models.py',
        'src/visualizations.py',
        'utils/helpers.py',
    ]
    
    all_ok = True
    
    for module in modules:
        if Path(module).exists():
            print(f"   ✅ {module}")
        else:
            print(f"   ❌ {module}: NO ENCONTRADO")
            all_ok = False
    
    return all_ok


def check_streamlit():
    """Verifica que Streamlit funcione."""
    print("\n🎯 Verificando Streamlit:")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'streamlit', 'version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"   ✅ Streamlit funcional")
            return True
        else:
            print(f"   ❌ Error en Streamlit")
            return False
    except Exception as e:
        print(f"   ❌ Error al verificar Streamlit: {e}")
        return False


def main():
    """Ejecuta todas las verificaciones."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " VERIFICACIÓN DE INSTALACIÓN ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    checks = [
        ("Versión de Python", check_python_version),
        ("Paquetes instalados", check_packages),
        ("Estructura de carpetas", check_project_structure),
        ("Módulos del proyecto", check_modules),
        ("Streamlit", check_streamlit),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{'=' * 60}")
        print(f"🔍 {name}")
        print('=' * 60)
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error durante la verificación: {e}")
            results.append((name, False))
    
    # Resumen final
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " RESUMEN ".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"║ {name:40} {status:15} ║")
        if not result:
            all_passed = False
    
    print("╚" + "═" * 58 + "╝")
    
    if all_passed:
        print("\n🎉 ¡VERIFICACIÓN EXITOSA!")
        print("\nPuedes ejecutar la aplicación con:")
        print("   streamlit run app.py")
        return 0
    else:
        print("\n❌ ALGUNAS VERIFICACIONES FALLARON")
        print("\nPara instalar dependencias, ejecuta:")
        print("   python install.py")
        print("   o")
        print("   pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
