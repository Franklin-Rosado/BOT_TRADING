import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. CONEXIÓN A METATRADER 5 
# ==========================================
print("Iniciando conexión con MetaTrader 5...")

if not mt5.initialize():
    print("Fallo al inicializar MT5. Asegúrate de tener la terminal abierta.")
    quit()

simbolo = "Bear1000" 
temporalidad = mt5.TIMEFRAME_D1 
velas_historia = 2000 

print(f"Descargando histórico de {simbolo}...")
rates = mt5.copy_rates_from_pos(simbolo, temporalidad, 0, velas_historia)
mt5.shutdown()

if rates is None or len(rates) == 0:
    raise RuntimeError(f"No se pudieron descargar datos de {simbolo}. Verifica el nombre del activo.")

# ==========================================
# 2. PREPARACIÓN DE DATOS E INDICADORES
# ==========================================
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df = df.rename(columns={'close': 'Adj Close'})

precios = df['Adj Close']
df['Log_Returns'] = np.log(precios / precios.shift(1))
df = df.dropna()

print("Datos de MetaTrader 5 extraídos y listos.")

# RSI (14 días)
ventana_rsi = 14
cambio = df['Adj Close'].diff()
ganancias = cambio.clip(lower=0)
perdidas = -1 * cambio.clip(upper=0)
promedio_ganancias = ganancias.ewm(com=ventana_rsi - 1, adjust=False).mean()
promedio_perdidas = perdidas.ewm(com=ventana_rsi - 1, adjust=False).mean()
rs = promedio_ganancias / promedio_perdidas
df['RSI_14'] = 100 - (100 / (1 + rs))

# SMA (20 y 50 días) y Tendencia
df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
df['Tendencia_SMA'] = df['SMA_20'] / df['SMA_50'] - 1

# Volatilidad (Anualizada a 365 días para sintéticos)
ventana_volatilidad = 20
df['Volatilidad_Diaria'] = df['Log_Returns'].rolling(window=ventana_volatilidad).std()
df['Volatilidad_Anualizada'] = df['Volatilidad_Diaria'] * np.sqrt(365)

# Limpieza y Target
df = df.dropna()
df['Retorno_Manana'] = df['Log_Returns'].shift(-1)
df['Target'] = np.where(df['Retorno_Manana'] > 0, 1, 0)
df = df.dropna()

caracteristicas = ['Log_Returns', 'Volatilidad_Anualizada', 'RSI_14', 'Tendencia_SMA'] 
X = df[caracteristicas]
y = df['Target'] 

print("\nMuestra de los datos preparados:")
print(df[['Log_Returns', 'Volatilidad_Anualizada', 'Retorno_Manana', 'Target']].tail())

# ==========================================
# 3. ENTRENAMIENTO DEL MODELO (RANDOM FOREST)
# ==========================================
indice_corte = int(len(df) * 0.80)
X_train = X.iloc[:indice_corte]
y_train = y.iloc[:indice_corte]
X_test = X.iloc[indice_corte:]
y_test = y.iloc[indice_corte:]

print(f"\nDías de historia para aprender (Train): {len(X_train)}")
print(f"Días para evaluar si el modelo funciona (Test): {len(X_test)}")

parametros_grid = {
    'n_estimators': [50, 100, 200],      
    'max_depth': [3, 5, 7, 10],          
    'min_samples_leaf': [10, 20, 30]     
}

tscv = TimeSeriesSplit(n_splits=3)
modelo_base = RandomForestClassifier(random_state=42, class_weight='balanced')
buscador = GridSearchCV(
    estimator=modelo_base,
    param_grid=parametros_grid,
    cv=tscv,               
    scoring='accuracy',    
    n_jobs=-1              
)

print("Buscando la mejor configuración... por favor espera.")
buscador.fit(X_train, y_train)

print("\n¡Búsqueda terminada!")
print(f"La mejor configuración encontrada fue: {buscador.best_params_}")

mejor_modelo = buscador.best_estimator_
predicciones_optimizadas = mejor_modelo.predict_proba(X_test)
prob_de_subida = predicciones_optimizadas[:, 1]

umbral_seguridad = 0.51
predicciones = np.where(prob_de_subida > umbral_seguridad, 1, 0) 

precision_estricta = accuracy_score(y_test, predicciones)
print(f"\nNueva Precisión con Umbral del {umbral_seguridad*100}%: {precision_estricta * 100:.2f}%")
print("\nReporte Detallado con Umbral Estricto:")
print(classification_report(y_test, predicciones))

# Ranking de Importancia
valores_importancia = mejor_modelo.feature_importances_
ranking = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': valores_importancia
}).sort_values(by='Importancia', ascending=False)

print("\nRanking de Importancia de Variables:")
print(ranking)
print("-" * 30)

# ==========================================
# 4. SIMULACIÓN Y MÉTRICAS DE RIESGO
# ==========================================
resultados = pd.DataFrame(index=X_test.index)
resultados['Retorno_Real'] = df.loc[X_test.index, 'Retorno_Manana'] 
resultados['Senal_Modelo'] = predicciones 
resultados['Retorno_Estrategia'] = resultados['Senal_Modelo'] * resultados['Retorno_Real']

costo_transaccion = 0.001
resultados['Cambio_Senal'] = resultados['Senal_Modelo'].diff().fillna(resultados['Senal_Modelo'].iloc[0])
resultados['Costo_Diario'] = np.abs(resultados['Cambio_Senal']) * (-np.log1p(-costo_transaccion))
resultados['Retorno_Estrategia_Neto'] = resultados['Retorno_Estrategia'] - resultados['Costo_Diario']

resultados['Capital_Mercado'] = np.exp(resultados['Retorno_Real'].cumsum())
resultados['Capital_Neto'] = np.exp(resultados['Retorno_Estrategia_Neto'].cumsum())

# Sharpe Ratio ajustado a 365 días
sharpe_mercado = np.sqrt(365) * (resultados['Retorno_Real'].mean() / resultados['Retorno_Real'].std())
sharpe_estrategia = np.sqrt(365) * (resultados['Retorno_Estrategia_Neto'].mean() / resultados['Retorno_Estrategia_Neto'].std())

pico_mercado = resultados['Capital_Mercado'].cummax()
pico_estrategia = resultados['Capital_Neto'].cummax()
drawdown_mercado = (resultados['Capital_Mercado'] / pico_mercado) - 1
drawdown_estrategia = (resultados['Capital_Neto'] / pico_estrategia) - 1
max_dd_mercado = drawdown_mercado.min()
max_dd_estrategia = drawdown_estrategia.min()

print("\n" + "="*40)
print("RADIOGRAFÍA DE RIESGO VS RETORNO")
print("="*40)
print(f"Mercado ({simbolo}):")
print(f" - Sharpe Ratio : {sharpe_mercado:.2f} (Mayor a 1.0 es bueno)")
print(f" - Max Drawdown : {max_dd_mercado*100:.2f}% (Peor caída sufrida)")
print("-" * 40)
print(f"Estrategia Neta:")
print(f" - Sharpe Ratio : {sharpe_estrategia:.2f}")
print(f" - Max Drawdown : {max_dd_estrategia*100:.2f}%")
print("="*40)

# ==========================================
# 5. EXPORTACIÓN DE RESULTADOS Y MODELO
# ==========================================
resultados['Precio_Activo'] = df.loc[X_test.index, 'Adj Close']
columnas_exportar = ['Precio_Activo', 'Senal_Modelo', 'Retorno_Real', 'Capital_Mercado', 'Capital_Neto']
tabla_auditoria = resultados[columnas_exportar]

nombre_csv = f'auditoria_{simbolo.lower()}_modelo.csv'
tabla_auditoria.to_csv(nombre_csv)
print(f"\n¡Archivo exportado exitosamente como {nombre_csv}!")

nombre_modelo = f'cerebro_{simbolo.lower()}.pkl'
joblib.dump(mejor_modelo, 'C:/Users/frank/PROYECT/' + nombre_modelo)
print(f"Modelo guardado exitosamente como {nombre_modelo}.")