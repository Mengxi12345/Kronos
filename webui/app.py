import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
import time
warnings.filterwarnings('ignore')

# Import akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("Warning: akshare not available, please install with: pip install akshare")

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import DEFAULT_MODEL_KEY, DEFAULT_DEVICE, AUTO_LOAD_MODEL_ON_STARTUP

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None
current_model_key = None  # Store current loaded model key
current_device = None  # Store current device

# Global variables to store data
current_data_df = None  # Store current loaded data
current_symbol = None  # Store current stock symbol
current_lookback = None  # Store current lookback length

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_from_akshare(symbol: str, lookback: int = 400) -> tuple:
    """
    从akshare获取股票数据
    
    Args:
        symbol: 股票代码（如 '000001', '002594'）
        lookback: 需要获取的历史数据长度
    
    Returns:
        (df, error): DataFrame和错误信息
    """
    if not AKSHARE_AVAILABLE:
        return None, "akshare库未安装，请运行: pip install akshare"
    
    try:
        print(f"📥 正在从akshare获取 {symbol} 的日线数据...")
        
        max_retries = 3
        df = None
        
        # 重试机制
        for attempt in range(1, max_retries + 1):
            try:
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
                if df is not None and not df.empty:
                    break
            except Exception as e:
                print(f"⚠️ 尝试 {attempt}/{max_retries} 失败: {e}")
                if attempt < max_retries:
                    time.sleep(1.5)
        
        # 如果重试后仍然为空
        if df is None or df.empty:
            return None, f"无法获取股票 {symbol} 的数据，请检查股票代码是否正确"
        
        # 重命名列
        df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount"
        }, inplace=True)
        
        # 转换日期列
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # 转换数值列
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .replace({"--": None, "": None})
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 修复无效的开盘价
        open_bad = (df["open"] == 0) | (df["open"].isna())
        if open_bad.any():
            print(f"⚠️  修复了 {open_bad.sum()} 个无效的开盘价")
            df.loc[open_bad, "open"] = df["close"].shift(1)
            df["open"].fillna(df["close"], inplace=True)
        
        # 修复缺失的成交额
        if "amount" in df.columns:
            if df["amount"].isna().all() or (df["amount"] == 0).all():
                df["amount"] = df["close"] * df["volume"]
        
        # 重命名date为timestamps以保持一致性
        df.rename(columns={"date": "timestamps"}, inplace=True)
        
        # 确保有足够的行数
        if len(df) < lookback:
            return None, f"数据不足：只有 {len(df)} 行，需要至少 {lookback} 行"
        
        # 只取最后lookback行数据
        df = df.tail(lookback).reset_index(drop=True)
        
        # 移除包含NaN的行
        df = df.dropna()
        
        print(f"✅ 数据加载成功: {len(df)} 行, 时间范围: {df['timestamps'].min()} ~ {df['timestamps'].max()}")
        
        return df, None
        
    except Exception as e:
        return None, f"从akshare获取数据失败: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """从akshare加载股票数据"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip()
        lookback = int(data.get('lookback', 400))
        
        if not symbol:
            return jsonify({'error': '股票代码不能为空'}), 400
        
        if lookback < 100:
            return jsonify({'error': '历史数据长度至少需要100'}), 400
        
        if lookback > 2000:
            return jsonify({'error': '历史数据长度不能超过2000'}), 400
        
        # 从akshare获取数据
        df, error = load_data_from_akshare(symbol, lookback)
        if error:
            return jsonify({'error': error}), 400
        
        # 检测数据时间频率
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # 返回数据信息
        data_info = {
            'symbol': symbol,
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        # 将数据存储在全局变量中（用于预测）
        global current_data_df, current_symbol, current_lookback
        current_data_df = df
        current_symbol = symbol
        current_lookback = lookback
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'成功加载股票 {symbol} 的数据，共 {len(df)} 行'
        })
        
    except Exception as e:
        return jsonify({'error': f'加载数据失败: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """执行预测"""
    global current_data_df, current_symbol, current_lookback
    
    try:
        data = request.get_json()
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        # 检查数据是否已加载
        if current_data_df is None:
            return jsonify({'error': '请先加载股票数据'}), 400
        
        df = current_data_df
        lookback = current_lookback or len(df)
        
        if len(df) < lookback:
            return jsonify({'error': f'数据长度不足，需要至少 {lookback} 行'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Use latest lookback rows for prediction
                x_df = df.iloc[-lookback:][required_cols]
                x_timestamp = df.iloc[-lookback:]['timestamps']
                
                # Generate future timestamps for prediction
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[-1] - df['timestamps'].iloc[-2] if len(df) > 1 else pd.Timedelta(days=1)
                y_timestamp = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=pred_len,
                    freq=time_diff
                )
                y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                prediction_type = f"Kronos模型预测 (股票 {current_symbol})"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # 预测未来数据，没有实际数据用于比较
        actual_data = []
        actual_df = None
        
        # Create chart - use latest data
        historical_start_idx = max(0, len(df) - lookback)
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - use timestamps from y_timestamp
        # y_timestamp already contains the correct future timestamps
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            # Use timestamp from y_timestamp Series
            if isinstance(y_timestamp, pd.Series):
                timestamp = y_timestamp.iloc[i]
            elif isinstance(y_timestamp, pd.DatetimeIndex):
                timestamp = y_timestamp[i]
            else:
                timestamp = y_timestamp[i] if i < len(y_timestamp) else None
            
            # Convert timestamp to ISO format
            if hasattr(timestamp, 'isoformat'):
                timestamp_str = timestamp.isoformat()
            elif timestamp is not None:
                timestamp_str = str(timestamp)
            else:
                timestamp_str = f"T{i}"
            
            prediction_results.append({
                'timestamp': timestamp_str,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=f"akshare_{current_symbol}",
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'symbol': current_symbol,
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor, current_model_key, current_device
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        # Store current model info
        current_model_key = model_key
        current_device = device
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    global current_model_key, current_device
    
    if MODEL_AVAILABLE:
        if predictor is not None and current_model_key is not None:
            # Get model info from stored key
            model_config = AVAILABLE_MODELS.get(current_model_key, {})
            
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': model_config.get('name', 'Unknown'),
                    'params': model_config.get('params', 'Unknown'),
                    'device': current_device or str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

def load_model_on_startup():
    """在启动时加载模型"""
    global tokenizer, model, predictor, current_model_key, current_device
    
    if not MODEL_AVAILABLE:
        print("⚠️  Model library not available, skipping model loading")
        return False
    
    if not AUTO_LOAD_MODEL_ON_STARTUP:
        print("ℹ️  Auto-load model is disabled in config, skipping model loading")
        return False
    
    try:
        print(f"🔄 Loading model: {DEFAULT_MODEL_KEY} on {DEFAULT_DEVICE}...")
        
        if DEFAULT_MODEL_KEY not in AVAILABLE_MODELS:
            print(f"❌ Unsupported model: {DEFAULT_MODEL_KEY}")
            return False
        
        model_config = AVAILABLE_MODELS[DEFAULT_MODEL_KEY]
        
        # Load tokenizer and model
        print(f"📥 Loading tokenizer: {model_config['tokenizer_id']}...")
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        
        print(f"📥 Loading model: {model_config['model_id']}...")
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        print(f"🔧 Creating predictor on {DEFAULT_DEVICE}...")
        predictor = KronosPredictor(model, tokenizer, device=DEFAULT_DEVICE, max_context=model_config['context_length'])
        
        # Store current model info
        current_model_key = DEFAULT_MODEL_KEY
        current_device = DEFAULT_DEVICE
        
        print(f"✅ Model loaded successfully: {model_config['name']} ({model_config['params']}) on {DEFAULT_DEVICE}")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Kronos Web UI...")
    print("=" * 60)
    print(f"Model availability: {MODEL_AVAILABLE}")
    
    # 启动时预加载模型
    if AUTO_LOAD_MODEL_ON_STARTUP and MODEL_AVAILABLE:
        load_model_on_startup()
    elif MODEL_AVAILABLE:
        print("ℹ️  Auto-load model is disabled, model will not be loaded on startup")
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("⚠️  Model library not available, will use simulated data for demonstration")
    
    print("=" * 60)
    print("🌐 Web server starting on http://0.0.0.0:7070")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=7070)
