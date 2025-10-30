import pandas as pd
import ccxt
from datetime import datetime, timezone
import time

def download_binance_btc(symbol='BTC/USDT', timeframe='1h', start_date='2020-01-01', end_date='2025-01-01'):
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,  # Required by Binance 
    })
    
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            
            if not ohlcv:
                break
            
            all_data.extend(ohlcv)
            
            current_ts = ohlcv[-1][0] + 1
            
            print(f"Downloaded {len(all_data)} candles... (up to {datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})")
            
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5) 
            continue
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    
    df = df[~df.index.duplicated(keep='first')]
    
    df.sort_index(inplace=True)
    
    print(f"\nDownload complete!")
    print(f"Total candles: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nDataFrame info:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    return df

def download_binance_eth(symbol='ETH/USDT', timeframe='1h', start_date='2020-01-01', end_date='2025-01-01'):
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,  # Required by Binance 
    })
    
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            
            if not ohlcv:
                break
            
            all_data.extend(ohlcv)
            
            current_ts = ohlcv[-1][0] + 1
            
            print(f"Downloaded {len(all_data)} candles... (up to {datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})")
            
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5) 
            continue
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    
    df = df[~df.index.duplicated(keep='first')]
    
    df.sort_index(inplace=True)
    
    print(f"\nDownload complete!")
    print(f"Total candles: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nDataFrame info:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    return df

if __name__ == "__main__":
    df_btc = download_binance_btc(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date='2020-01-01',
        end_date='2025-01-01'
    )
    output_file = './data/binance-BTCUSDT-1h.pkl'
    df_btc.to_pickle(output_file)
    print(f"\nData saved to: {output_file}")

    df_eth = download_binance_eth(
        symbol='ETH/USDT',
        timeframe='1h',
        start_date='2020-01-01',
        end_date='2025-01-01'
    )
    
    output_file = './data/binance-ETHUSDT-1h.pkl'
    df_eth.to_pickle(output_file)
    print(f"\nData saved to: {output_file}")