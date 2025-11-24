from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import pickle
import json
import os
import math
import re  # Untuk cleaning price

app = Flask(__name__)

# --- PATH ---
CSV_PATH = 'WebData/DatasetFinal.csv'
MODEL_PATH = 'WebData/car_prediction_model.pkl'
LABEL_PATH = 'WebData/cluster_labels.json'
ORDERS_PATH = 'WebData/orders.json'

# --- LOAD ASSET ---
print("Loading Assets...")

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded.")
except FileNotFoundError:
    print("Model tidak ditemukan!")
    model = None

try:
    with open(LABEL_PATH, 'r') as f:
        raw_labels = json.load(f)
        cluster_labels = {int(k): v for k, v in raw_labels.items()}
    print("Labels loaded.")
except FileNotFoundError:
    cluster_labels = {}

def get_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df['cluster_name'] = df['cluster'].apply(lambda x: cluster_labels.get(x, f"Cluster {x}"))
        return df
    return None

# Function to clean price (mirip dari KMEANS.ipynb)
def clean_price(price_str):
    if not isinstance(price_str, str):
        return None
    price_str = re.sub(r'[\$, ]', '', price_str)  # Hapus $, comma, spasi
    if '-' in price_str:
        try:
            low, high = map(float, price_str.split('-'))
            return (low + high) / 2
        except:
            return None
    try:
        return float(price_str)
    except:
        return None

# Function to clean seats
def clean_seats(seats_str):
    try:
        return float(seats_str)
    except:
        return None

# --- ROUTES EXISTING ---
@app.route('/')
def index():
    df = get_data()
    if df is None:
        return "Error: Dataset tidak ditemukan."

    page = request.args.get('page', 1, type=int)
    per_page = 50
    total = len(df)
    total_pages = math.ceil(total / per_page)
    start = (page - 1) * per_page
    end = start + per_page

    columns = ['Company Names', 'Cars Names', 'Engines', 'CC/Battery Capacity',
               'HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H',
               'Cars Prices', 'Fuel Types', 'Seats', 'Torque', 'cluster_name']

    data_page = df[columns].iloc[start:end].to_dict(orient='records')

    return render_template('index.html',
                           data=data_page,
                           columns=columns,
                           page=page,
                           total_pages=total_pages)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = ""
    filtered_data = None
    columns_to_show = None
    page = request.args.get('page', 1, type=int)
    total_pages = 1
    last_price = request.form.get('price') or request.args.get('price') or ""
    last_seats = request.form.get('seats') or request.args.get('seats') or ""

    if request.method == 'POST' or (request.args.get('price') and request.args.get('seats')):
        try:
            price = float(request.form.get('price') or request.args.get('price'))
            seats = float(request.form.get('seats') or request.args.get('seats'))

            if model:
                pred_idx = model.predict([[price, seats]])[0]
                prediction_text = cluster_labels.get(pred_idx, f"Cluster {pred_idx}")

                df = get_data()
                if df is not None:
                    filtered = df[df['cluster'] == pred_idx]
                    per_page = 10
                    total = len(filtered)
                    total_pages = math.ceil(total / per_page)
                    start = (page - 1) * per_page
                    end = start + per_page

                    columns_to_show = ['Company Names', 'Cars Names', 'Engines', 'CC/Battery Capacity',
                                       'HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H',
                                       'Cars Prices', 'Fuel Types', 'Seats', 'Torque', 'cluster_name']

                    data_page = filtered[columns_to_show].iloc[start:end]
                    filtered_data = data_page.to_dict(orient='records')
        except:
            prediction_text = "Input tidak valid."

    return render_template('predict.html',
                           prediction=prediction_text,
                           last_price=last_price,
                           last_seats=last_seats,
                           data=filtered_data,
                           columns=columns_to_show,
                           page=page,
                           total_pages=total_pages)

@app.route('/buy', methods=['POST'])
def buy():
    try:
        data = request.json
        nama = data.get('nama')
        alamat = data.get('alamat')
        no_telp = data.get('no_telp')
        car_full_name = data.get('Car_Full_Name')
        cluster_name = data.get('cluster')

        if os.path.exists(ORDERS_PATH):
            with open(ORDERS_PATH, 'r') as f:
                orders = json.load(f)
        else:
            orders = []

        orders.append({
            "nama": nama,
            "alamat": alamat,
            "no_telp": no_telp,
            "Car_Full_Name": car_full_name,
            "cluster": cluster_name,
            "timestamp": pd.Timestamp.now().isoformat()
        })

        with open(ORDERS_PATH, 'w') as f:
            json.dump(orders, f, indent=4)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/sales')
def sales():
    if not os.path.exists(ORDERS_PATH):
        orders = []
    else:
        with open(ORDERS_PATH, 'r') as f:
            orders = json.load(f)

    orders = sorted(orders, key=lambda x: x['timestamp'], reverse=True)

    page = request.args.get('page', 1, type=int)
    per_page = 20
    total = len(orders)
    total_pages = math.ceil(total / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = orders[start:end]

    return render_template('sales.html',
                           orders=page_data,
                           page=page,
                           total_pages=total_pages)

# --- ROUTE BARU: TAMBAH MOBIL ---
@app.route('/add_car', methods=['GET', 'POST'])
def add_car():
    error = None
    if request.method == 'POST':
        try:
            # Ambil input dari form
            company_names = request.form['company_names']
            cars_names = request.form['cars_names']
            engines = request.form['engines']
            cc_battery = request.form['cc_battery']
            horsepower = request.form['horsepower']
            total_speed = request.form['total_speed']
            performance = request.form['performance']
            cars_prices = request.form['cars_prices']
            fuel_types = request.form['fuel_types']
            seats = request.form['seats']
            torque = request.form['torque']

            # Generate Car_Full_Name
            car_full_name = f"{company_names.upper()} {cars_names.upper()}"

            # Clean price and seats
            price_cleaned = clean_price(cars_prices)
            seats_cleaned = clean_seats(seats)

            if price_cleaned is None or seats_cleaned is None:
                raise ValueError("Input harga atau kursi tidak valid!")

            # Predict cluster jika model ada
            if model:
                cluster_pred = model.predict([[price_cleaned, seats_cleaned]])[0]
            else:
                raise ValueError("Model tidak tersedia!")

            # Load CSV existing
            if os.path.exists(CSV_PATH):
                df = pd.read_csv(CSV_PATH)
            else:
                raise FileNotFoundError("CSV tidak ditemukan!")

            # Buat row baru
            new_row = pd.DataFrame({
                'Company Names': [company_names],
                'Cars Names': [cars_names],
                'Engines': [engines],
                'CC/Battery Capacity': [cc_battery],
                'HorsePower': [horsepower],
                'Total Speed': [total_speed],
                'Performance(0 - 100 )KM/H': [performance],
                'Cars Prices': [cars_prices],
                'Fuel Types': [fuel_types],
                'Seats': [seats],
                'Torque': [torque],
                'Car_Full_Name': [car_full_name],
                'Price_Cleaned': [price_cleaned],
                'Seats_Cleaned': [seats_cleaned],
                'cluster': [cluster_pred]
            })

            # Append dan simpan
            df_updated = pd.concat([df, new_row], ignore_index=True)
            df_updated.to_csv(CSV_PATH, index=False)

            return redirect(url_for('index'))  # Redirect ke list mobil setelah sukses
        except Exception as e:
            error = str(e)

    return render_template('add_car.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)