import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def ejecutar_orden_mt5(simbolo, accion, volumen, precio_actual):
    tipo_orden = mt5.ORDER_TYPE_BUY if accion == 'buy' else mt5.ORDER_TYPE_SELL

    request_entrada = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": simbolo,
        "volume": float(volumen), 
        "type": tipo_orden,
        "price": precio_actual,
        "sl": 0.0,
        "tp": 0.0,
        "deviation": 20,
        "magic": 123456,
        "comment": "Bot_Entrada",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    resultado = mt5.order_send(request_entrada)
    
    if resultado.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Error al enviar orden: {resultado.comment} (Código: {resultado.retcode})")
        return 
        
    print(f"¡Orden ejecutada con éxito! Ticket: {resultado.order}")
    
    print("Asegurando la posición con Stop Loss y Take Profit...")
    
    if accion == 'buy':
        tp_precio = precio_actual * 1.08
        sl_precio = precio_actual * 0.97
    else:
        tp_precio = precio_actual * 0.92
        sl_precio = precio_actual * 1.03
        
    request_modificacion = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": simbolo,
        "position": resultado.order, 
        "sl": float(sl_precio),
        "tp": float(tp_precio),
        "magic": 123456
    }
    
    resultado_mod = mt5.order_send(request_modificacion)
    
    if resultado_mod.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"La compra entró, pero falló el SL/TP: {resultado_mod.comment}")
    else:
        print(f"¡Seguridad activa! TP: {tp_precio:.2f} | SL: {sl_precio:.2f}")

def main():
    activos = ['Bear1000'] 
    umbral_tecnico = 0.55

    print("Iniciando Bot MT5 para Índices Sintéticos...")
    if not mt5.initialize():
        print("Fallo al inicializar MT5.")
        return

    for simbolo in activos:
        print("\n" + "="*50)
        print(f"EVALUANDO: {simbolo}")
        print("="*50)
        
        try:
            ruta_modelo = Path(__file__).with_name(f'cerebro_{simbolo.lower()}.pkl')
            if not ruta_modelo.exists():
                print(f"No se encontró el modelo para {simbolo}.")
                continue
            modelo = joblib.load(ruta_modelo)

            rates = mt5.copy_rates_from_pos(simbolo, mt5.TIMEFRAME_D1, 0, 100)
            if rates is None or len(rates) == 0:
                print(f"Sin datos para {simbolo}.")
                continue
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            precios = df['close']

            df['Log_Returns'] = np.log(precios / precios.shift(1))
            cambio = precios.diff()
            ganancias = cambio.clip(lower=0)
            perdidas = -1 * cambio.clip(upper=0)
            rs = ganancias.ewm(com=13, adjust=False).mean() / perdidas.ewm(com=13, adjust=False).mean()
            df['RSI_14'] = 100 - (100 / (1 + rs))

            df['SMA_20'] = precios.rolling(window=20).mean()
            df['SMA_50'] = precios.rolling(window=50).mean()
            df['Tendencia_SMA'] = df['SMA_20'] / df['SMA_50'] - 1

            df['Volatilidad_Diaria'] = df['Log_Returns'].rolling(window=20).std()
            df['Volatilidad_Anualizada'] = df['Volatilidad_Diaria'] * np.sqrt(365) 

            caracteristicas = ['Log_Returns', 'Volatilidad_Anualizada', 'RSI_14', 'Tendencia_SMA']
            df = df.dropna(subset=caracteristicas)

            datos_hoy = df[caracteristicas].iloc[-1:]
            prob_subida = float(modelo.predict_proba(datos_hoy)[0][1])
            print(f"Certeza técnica del modelo: {prob_subida * 100:.2f}%")

            precio_actual = mt5.symbol_info_tick(simbolo).ask
            posiciones = mt5.positions_get(symbol=simbolo)
            tenemos_activo = len(posiciones) > 0

            if prob_subida > umbral_tecnico:
                print(f"Señal de COMPRA confirmada para {simbolo}.")
                if tenemos_activo:
                    print(f"Ya poseemos {simbolo}. Mantenemos.")
                else:
                    volumen_comprar = 0.03
                    print(f"Comprando {volumen_comprar} lotes a ~${precio_actual:.2f}...")
                    ejecutar_orden_mt5(simbolo, 'buy', volumen_comprar, precio_actual)
            else:
                print(f"Señal débil/bajista para {simbolo}. Manteniendo efectivo.")

        except Exception as e:
            print(f"Error evaluando {simbolo}: {e}")

    mt5.shutdown()
    print("\nCICLO FINALIZADO.")

if __name__ == "__main__":
    main()