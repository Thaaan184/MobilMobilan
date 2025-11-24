from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import pickle
import json
import os
import math

app = Flask(__name__)

# --- KONFIGURASI PATH ---
# Pastikan path ini sesuai dengan struktur folder kamu
CSV_PATH = 'WebData/DatasetFinal.csv'
MODEL_PATH = 'WebData/car_prediction_model.pkl'
LABEL_PATH = 'WebData/cluster_labels.json'
ORDERS_PATH = 'WebData/orders.json'  # File baru untuk menyimpan data pembelian

# --- LOAD ASET SAAT APLIKASI DIMULAI ---
print("Loading Assets...")

# 1. Load Model Random Forest
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded.")
except FileNotFoundError:
    print(f"❌ Error: Model tidak ditemukan di {MODEL_PATH}")
    model = None

# 2. Load Label Mapping (JSON)
try:
    with open(LABEL_PATH, 'r') as f:
        # Load JSON dan ubah key dari string ke int agar cocok dengan prediksi model
        raw_labels = json.load(f)
        cluster_labels = {int(k): v for k, v in raw_labels.items()}
    print("✅ Labels loaded.")
except FileNotFoundError:
    print(f"❌ Error: JSON Labels tidak ditemukan di {LABEL_PATH}")
    cluster_labels = {}

# 3. Load Data CSV
def get_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        
        # Mapping Cluster Angka ke Nama (Text)
        # Jika cluster ada di JSON pakai namanya, jika tidak pakai angkanya
        df['cluster_name'] = df['cluster'].apply(lambda x: cluster_labels.get(x, f"Cluster {x}"))
        
        return df
    return None

# --- ROUTE HALAMAN UTAMA (TABEL) ---
@app.route('/')
def index():
    df = get_data()
    
    if df is None:
        return "Error: File CSV tidak ditemukan."

    # Konfigurasi Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 50
    total_rows = len(df)
    total_pages = math.ceil(total_rows / per_page)
    
    start = (page - 1) * per_page
    end = start + per_page
    
    # Filter Kolom yang mau ditampilkan 
    columns_to_show = [
        'Company Names', 'Cars Names', 'Engines', 'CC/Battery Capacity', 
        'HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H', 
        'Cars Prices', 'Fuel Types', 'Seats', 'Torque', 'cluster_name'
    ]
    
    # Ambil data untuk halaman ini
    data_page = df[columns_to_show].iloc[start:end]
    
    # Convert ke dictionary untuk dikirim ke HTML
    data_records = data_page.to_dict(orient='records')
    
    return render_template('index.html', 
                           data=data_records, 
                           columns=columns_to_show,
                           page=page, 
                           total_pages=total_pages)

# --- ROUTE HALAMAN BELI MOBIL (MODIFIKASI DARI PREDICT) ---
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = ""
    filtered_data = None
    columns_to_show = None
    page = request.args.get('page', 1, type=int)
    total_pages = 1

    # Ambil input dari POST atau GET (untuk pagination)
    if request.method == 'POST':
        input_price = request.form.get('price')
        input_seats = request.form.get('seats')
    else:
        input_price = request.args.get('price')
        input_seats = request.args.get('seats')

    last_price = input_price or ""
    last_seats = input_seats or ""

    if input_price and input_seats:
        try:
            input_price = float(input_price)
            input_seats = float(input_seats)
            
            # Prediksi dengan Model
            if model:
                pred_idx = model.predict([[input_price, input_seats]])[0]
                pred_label = cluster_labels.get(pred_idx, f"Cluster {pred_idx}")
                
                prediction_text = pred_label
                
                # Load data dan filter berdasarkan cluster prediksi
                df = get_data()
                if df is not None:
                    filtered_df = df[df['cluster'] == pred_idx]
                    
                    # Konfigurasi Pagination untuk hasil filter (10 per page)
                    per_page = 10
                    total_rows = len(filtered_df)
                    total_pages = math.ceil(total_rows / per_page)
                    
                    start = (page - 1) * per_page
                    end = start + per_page
                    
                    # Kolom yang ditampilkan, tambah kolom 'Aksi' untuk tombol Beli
                    columns_to_show = [
                        'Company Names', 'Cars Names', 'Engines', 'CC/Battery Capacity', 
                        'HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H', 
                        'Cars Prices', 'Fuel Types', 'Seats', 'Torque', 'cluster_name'
                    ]
                    
                    # Ambil data untuk halaman ini
                    data_page = filtered_df[columns_to_show].iloc[start:end]
                    filtered_data = data_page.to_dict(orient='records')
                    
            else:
                prediction_text = "Error: Model belum dimuat."
                
        except ValueError:
            prediction_text = "Input tidak valid. Masukkan angka."

    return render_template('predict.html', 
                           prediction=prediction_text,
                           last_price=last_price,
                           last_seats=last_seats,
                           data=filtered_data,
                           columns=columns_to_show,
                           page=page,
                           total_pages=total_pages)

# --- ROUTE UNTUK HANDLE PEMBELIAN (SIMPAN KE JSON) ---
@app.route('/buy', methods=['POST'])
def buy():
    try:
        data = request.json  # Ambil data dari AJAX
        nama = data.get('nama')
        alamat = data.get('alamat')
        no_telp = data.get('no_telp')
        car_name = data.get('car_name')  # Nama mobil yang dibeli
        
        # Load existing orders atau buat baru jika belum ada
        if os.path.exists(ORDERS_PATH):
            with open(ORDERS_PATH, 'r') as f:
                orders = json.load(f)
        else:
            orders = []
        
        # Tambah order baru
        new_order = {
            'nama': nama,
            'alamat': alamat,
            'no_telp': no_telp,
            'car_name': car_name,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        orders.append(new_order)
        
        # Simpan kembali ke JSON
        with open(ORDERS_PATH, 'w') as f:
            json.dump(orders, f, indent=4)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)