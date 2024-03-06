import streamlit as st
from streamlit.components.v1 import html as st_html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from branca.element import Figure

# Load Data
customer = pd.read_csv("brazilian-ecommerce/olist_customers_dataset.csv", delimiter=",")
order_items = pd.read_csv("brazilian-ecommerce/olist_order_items_dataset.csv", delimiter=",")
payments = pd.read_csv("brazilian-ecommerce/olist_order_payments_dataset.csv")
orders = pd.read_csv("brazilian-ecommerce/olist_orders_dataset.csv")
products = pd.read_csv("brazilian-ecommerce/olist_products_dataset.csv")
sellers = pd.read_csv("brazilian-ecommerce/olist_sellers_dataset.csv")
products_translation = pd.read_csv("brazilian-ecommerce/product_category_name_translation.csv")
geolocation = pd.read_csv("brazilian-ecommerce/olist_geolocation_dataset.csv")
geolocation = geolocation.drop_duplicates(subset=['geolocation_zip_code_prefix'])

# Merge Data
products = products.merge(products_translation, left_on='product_category_name', right_on='product_category_name', how='left')
df_product = products[["product_id", "product_category_name_english", "product_category_name"]]
df_order_items = order_items.merge(df_product, on='product_id', how='left')
df_order_items = df_order_items.merge(sellers, on='seller_id', how='left')
df_order_items = df_order_items.merge(geolocation, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left', validate='m:1')
df_order_items = df_order_items.drop_duplicates()  # Remove duplicates after merging
payments = payments.drop(columns=['payment_sequential', 'payment_installments'])
orders = orders.merge(payments, on='order_id', how='left')
customer = customer.drop(columns=['customer_unique_id'])
orders = orders.merge(customer, on='customer_id', how='left')
orders = orders.merge(geolocation, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')

# Mengubah tipe data beberapa kolom pada DataFrame orders
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'])
orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
orders['order_status'] = orders['order_status'].astype('category')

# Menambahkan kolom untuk analisis EDA pada DataFrame orders
orders['year'] = orders['order_purchase_timestamp'].dt.strftime('%Y')
orders['month'] = orders['order_purchase_timestamp'].dt.strftime('%m-%Y')
orders["lama_pengiriman_hari"] = (orders["order_delivered_customer_date"] - orders["order_delivered_carrier_date"]).dt.days
orders["hari_pembelian"] = orders["order_purchase_timestamp"].dt.strftime('%A')
orders['jam_pembelian'] = orders['order_purchase_timestamp'].apply(lambda x: x.hour)

# Mengelompokkan jam pembelian ke dalam kategori waktu hari
hours_bins = [-0.1, 6, 12, 18, 23]
hours_labels = ['Subuh', 'Pagi', 'Siang', 'Malam']
orders['waktu_hari_pembelian'] = pd.cut(orders['jam_pembelian'], hours_bins, labels=hours_labels)

# Mengelompokkan berdasarkan wilayah penjual dan menghitung jumlah penjual unik
unique_sellers_by_state = df_order_items.groupby(by="seller_state")["seller_id"].nunique().sort_values(ascending=False)

# Menggabungkan data customer dengan data seller berdasarkan order_id
customer_data = orders[["customer_city", "customer_state", "lama_pengiriman_hari", "order_id", "customer_id", 'customer_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']]
seller_data = df_order_items[["order_id", "seller_id", "seller_city", "seller_state", 'seller_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']]
cust_seller = customer_data.merge(seller_data, left_on='order_id', right_on='order_id', how='left')

# Cleaning Data
df_order_items['shipping_limit_date'] = pd.to_datetime(df_order_items['shipping_limit_date'])
df_order_items['product_category_name_english'].fillna('not defined', inplace=True)
df_order_items['product_category_name_english'] = np.where(df_order_items["product_category_name"] == 'pc_gamer', 'PC Gaming', df_order_items["product_category_name_english"])
df_order_items["product_category_name_english"] = np.where(df_order_items["product_category_name"] == 'portateis_cozinha_e_preparadores_de_alimentos', 'portable kitchen food preparers', df_order_items["product_category_name_english"])
df_order_items['year'] = df_order_items['shipping_limit_date'].dt.strftime('%Y')
df_order_items['month'] = df_order_items['shipping_limit_date'].dt.strftime('%m-%Y')
df_order_items.drop_duplicates(inplace=True)

orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'])
orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
orders['order_status'] = orders['order_status'].astype('category')
orders = orders.dropna(subset=["payment_type", "payment_value"])
orders = orders.drop_duplicates()

# Menghapus data dengan nilai lama_pengiriman_hari yang kurang atau sama dengan 0
orders = orders[orders["lama_pengiriman_hari"] > 0]

# Exploratory Data Analysis
category_orders = df_order_items.groupby(by="product_category_name_english")["product_id"].count().reset_index()
category_orders = category_orders.rename(columns={"product_category_name_english": "category", "product_id": "orders"})

# Mengelompokkan berdasarkan kota pembeli dan menghitung nilai median lama pengiriman
median_delivery_time_by_city = orders.groupby(by="customer_city")["lama_pengiriman_hari"].median().sort_values(ascending=False)

# Calculate delivery time between states and cities
delivery_time_between_states = cust_seller.groupby(['seller_state', 'customer_state'])['lama_pengiriman_hari'].median().sort_values(ascending=False).reset_index()
delivery_time_between_cities = cust_seller.groupby(['seller_city', 'customer_city'])['lama_pengiriman_hari'].median().sort_values(ascending=False).reset_index()

# Calculate delivery time in days
orders['lama_pengiriman_hari_state'] = (orders['order_delivered_customer_date'] - orders['order_approved_at']).dt.days

# Access the calculated delivery time columns in the customer_data DataFrame
customer_data = orders[["customer_city", "customer_state", "lama_pengiriman_hari_state", "order_id", "customer_id", 'customer_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']]

# Mengelompokkan berdasarkan state penjual dan pembeli, menghitung nilai median lama pengiriman, dan mengurutkan hasil
delivery_time_between_states = cust_seller.groupby(['seller_state', 'customer_state'])['lama_pengiriman_hari'].median().sort_values(ascending=False).reset_index()

# Menghitung rata-rata lama pengiriman antar state
df_delivery_state = cust_seller.groupby(['seller_state', 'customer_state'])['lama_pengiriman_hari'].mean().sort_values(ascending=False).reset_index()

# Menghitung rata-rata lama pengiriman antar city
df_delivery_city = cust_seller.groupby(['seller_city', 'customer_city', 'customer_zip_code_prefix', 'seller_zip_code_prefix'])['lama_pengiriman_hari'].mean().sort_values(ascending=False).reset_index()

# Mengelompokkan data berdasarkan jenis pembayaran dan menghitung rata-rata nilai pembayaran
df_payment = orders.groupby(by="payment_type")["payment_value"].mean().reset_index()

# Mengelompokkan data berdasarkan jenis pembayaran dan menghitung frekuensi penggunaan tiap tipe transaksi
df_payment_count = orders.groupby(by="payment_type")["order_id"].count().reset_index()

# Mengubah nama kolom untuk meningkatkan kejelasan  
df_payment = df_payment.rename(columns={"payment_type": "Payment Type", "payment_value": "Average Payment Value"})
df_payment_count = df_payment_count.rename(columns={"payment_type": "Payment Type", "order_id": "Transaction Count"})

# Menambahkan kolom 'nomor_bulan' sebagai angka bulan
orders['nomor_bulan'] = orders['order_purchase_timestamp'].dt.month

# Menghitung jumlah order per bulan untuk tahun 2017 dan 2018 (Januari-Agustus)
df_monthly_orders = orders.groupby(by=["nomor_bulan", "year"]).order_id.nunique().reset_index()

# Mengonversi tipe data 'nomor_bulan' menjadi int
df_monthly_orders["nomor_bulan"] = df_monthly_orders["nomor_bulan"].astype(int)

# Memfilter data untuk bulan Januari-Agustus
df_monthly_orders = df_monthly_orders[df_monthly_orders["nomor_bulan"] < 9]

# Menyusun nama bulan dalam Bahasa Indonesia
month_names = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'Mei',
    6: 'Jun',
    7: 'Jul',
    8: 'Agu'
}

# Mapping nomor bulan ke nama bulan dalam Bahasa Indonesia
df_monthly_orders['nama_bulan'] = df_monthly_orders['nomor_bulan'].map(month_names)

# Streamlit Dashboard
st.title("Brazilian E-Commerce Overview Dashboard")
st.sidebar.header("Navigation")
selected_page = st.sidebar.radio("Go to", ["Home", "Pertanyaan 1", "Pertanyaan 2", "Pertanyaan 3", "Pertanyaan 4", "Pertanyaan 5", "Pertanyaan 6", "Pertanyaan 7"])

if selected_page == "Home":
    st.header("Welcome to the E-Commerce Data Analysis Dashboard")
    st.subheader("Explore the insights from the Brazilian E-Commerce Public Dataset")
    st.image("https://storage.googleapis.com/kaggle-datasets-images/55151/105464/d59245a7014a35a35cc7f7b721de4dae/dataset-cover.png?t=2018-09-21-16-21-21", caption="E-Commerce Image", use_column_width=True)
    # Tambahkan teks selamat datang dan deskripsi dataset di home page
    st.markdown(
        "Welcome! This is a Brazilian ecommerce public dataset of orders made at Olist Store. "
        "The dataset contains information on 100k orders from 2016 to 2018, made at multiple marketplaces in Brazil. "
        "Its features allow viewing an order from various dimensions: from order status, price, payment, and freight performance "
        "to customer location, product attributes, and reviews written by customers. We have also released a geolocation dataset "
        "that relates Brazilian zip codes to lat/lng coordinates."
    )
    st.markdown(
        "This is real commercial data, it has been anonymized, and references to the companies and partners in the review text "
        "have been replaced with the names of Game of Thrones great houses."
    )

    st.markdown("This dashboard provides an overview and analysis of the Brazilian E-Commerce Public Dataset. "
                "Navigate through different sections to explore insights and answer business questions.")

elif selected_page == "Pertanyaan 1":
    st.header("Top Categories Analysis")
    st.subheader("Discover the most popular and least popular product categories.")

    # Membuat subplot untuk menampilkan dua grafik bar
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    # Warna untuk grafik bar
    colors = ["#4C72B0", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    # Grafik bar untuk kategori terlaris
    sns.barplot(x="orders", y="category", data=category_orders.sort_values(by="orders", ascending=False).head(5), palette=colors, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("Number of Orders")  # Menambah label sumbu x
    ax[0].set_title("Top Categories", loc="center", fontsize=15)
    ax[0].tick_params(axis='y', labelsize=12)

    # Grafik bar untuk kategori sedikit diminati
    sns.barplot(x="orders", y="category", data=category_orders.sort_values(by="orders", ascending=True).head(5), palette=colors, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("Number of Orders")  # Menambah label sumbu x
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Least Popular Categories", loc="center", fontsize=15)
    ax[1].tick_params(axis='y', labelsize=12)

    # Menampilkan judul utama
    st.pyplot(fig)  # Menggunakan st.pyplot untuk menampilkan plot pada dashboard Streamlit
    
    # Menambahkan conclusion
    st.subheader("Conclusion: Kategori Barang yang Paling Diminati dan Paling Kurang Diminati")
    st.write("Berdasarkan analisis, kategori yang paling diminati oleh konsumen adalah bed_bath_table, menunjukkan minat tinggi dalam produk-produk untuk keperluan kamar tidur dan kamar mandi. Sebaliknya, kategori security dan service menunjukkan minat yang lebih rendah, mencerminkan kemungkinan keterbatasan atau kurangnya permintaan untuk barang-barang dalam kategori tersebut.")

elif selected_page == "Pertanyaan 2":
    # Visualisasi Distribusi Kota Customer
    st.subheader("Kota dengan Customer Terbanyak:")
    top_customer_city = customer['customer_city'].value_counts().head(10)
    
    # Setel gaya seaborn untuk tampilan yang lebih menarik
    sns.set(style="whitegrid")

    # Buat gambar dan sumbu menggunakan matplotlib
    fig_top_customer = plt.figure(figsize=(10, 6))

    # Buat bar plot untuk top 10 sebaran pelanggan berdasarkan kota
    sns.barplot(x=top_customer_city.values, y=top_customer_city.index, palette='plasma')
    plt.title('Top 10 Sebaran Pelanggan Berdasarkan Kota', fontsize=12)
    plt.xlabel('Jumlah Pelanggan', fontsize=10)
    plt.ylabel('Kota', fontsize=10)

    # Tampilkan plot pertama pada dashboard Streamlit
    st.pyplot(fig_top_customer)

    st.subheader("Kota dengan Customer Paling Sedikit:")
    tail_customer_city = customer['customer_city'].value_counts().tail(10)

    # Setel ukuran gambar untuk visualisasi tail
    fig_tail_customer = plt.figure(figsize=(10, 6))

    # Buat bar plot untuk tail 10 sebaran pelanggan berdasarkan kota
    sns.barplot(x=tail_customer_city.values, y=tail_customer_city.index, palette='plasma')
    plt.title('Tail 10 Sebaran Pelanggan Berdasarkan Kota', fontsize=12)
    plt.xlabel('Jumlah Pelanggan', fontsize=10)
    plt.ylabel('Kota', fontsize=10)

    # Rotasi label pada sumbu y untuk meningkatkan kejelasan
    plt.xticks(rotation=45)

    # Tampilkan plot kedua pada dashboard Streamlit
    st.pyplot(fig_tail_customer)

    # Menambahkan conclusion
    st.subheader("Conclusion: Kota yang memiliki jumlah customer paling banyak dan paling sedikit")
    st.write("Sao Paulo menjadi kota dengan jumlah pelanggan paling tinggi, mencapai hampir 16.000 orang. Sebaliknya, lebih dari 5 kota hanya memiliki 1 pelanggan. Diperlukan upaya untuk meningkatkan jumlah pelanggan di kota-kota tersebut agar tidak terjadi penurunan di masa mendatang. Selain itu, perlu dilakukan penyelidikan lebih lanjut untuk memahami penyebab rendahnya jumlah pelanggan dari beberapa kota.")

elif selected_page == "Pertanyaan 3":
    st.header("Analisis Pengiriman Antar State dan Antar Kota")
    st.subheader("Pengiriman Antar State")
    
    # Menampilkan scatter plot untuk visualisasi rute pengiriman antar state dengan warna yang lebih menarik
    plt.figure(figsize=(8, 6))
    cmap = sns.color_palette("viridis", as_cmap=True)
    scatter = plt.scatter(df_delivery_state['seller_state'], df_delivery_state['customer_state'], c=df_delivery_state['lama_pengiriman_hari'], cmap=cmap, s=100, edgecolor='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, label='Lama Pengiriman (Hari)')
    plt.xlabel('State Penjual', fontsize=11)
    plt.ylabel('State Pembeli', fontsize=11)
    plt.title('Visualisasi Lama Pengiriman Antar State', fontsize=14)
    st.pyplot(plt)

    # Menampilkan boxplot untuk distribusi lama pengiriman antar state
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid", font_scale=1.2)
    color_palette = sns.color_palette("viridis")
    sns.boxplot(x='lama_pengiriman_hari', data=df_delivery_state, color=color_palette[2])  # Ganti df_delivery_state['lama_pengiriman_hari'] menjadi x='lama_pengiriman_hari'
    plt.xlabel('Lama Pengiriman (Hari)', fontsize=14)
    plt.title('Distribusi Lama Pengiriman Antar State', fontsize=12)
    st.pyplot(plt)

    st.subheader("Pengiriman Antar Kota")
    
    # Menampilkan boxplot untuk distribusi lama pengiriman antar city
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid", font_scale=1.2)
    color_palette = sns.color_palette("viridis")
    sns.boxplot(x='lama_pengiriman_hari', data=df_delivery_city, color=color_palette[2])  # Ganti df_delivery_city['lama_pengiriman_hari'] menjadi x='lama_pengiriman_hari'
    plt.xlabel('Lama Pengiriman (Hari)', fontsize=10)
    plt.title('Distribusi Lama Pengiriman Antar Kota', fontsize=12)
    st.pyplot(plt)

    # Peta Pengiriman Antar Kota Terlama
    st.subheader("Peta Pengiriman Antar Kota Terlama")

    # Create a Folium map
    fig = Figure(width=900, height=550)
    map_folium = folium.Map(location=[-4.56218, -37.766625], tiles='cartodbpositron', zoom_start=4)

    # Add markers to the map
    folium.Marker(location=[-4.56218, -37.766625], popup='Aracati, CE', tooltip='Customer', icon=folium.Icon(color='blue')).add_to(map_folium)
    folium.Marker(location=[-21.174925, -47.768616], popup='Ribeirao Preto, SP', tooltip='Seller', icon=folium.Icon(color='red')).add_to(map_folium)

    # Add the Folium map to the Streamlit app
    fig.add_child(map_folium)

    # Use st.components.v1.html to render the HTML directly
    st_html(fig._repr_html_(), width=900, height=550, scrolling=True)

    # Mengelompokkan berdasarkan waktu hari pembelian dan menghitung jumlah pesanan unik
    hourly_orders = orders.groupby(by="waktu_hari_pembelian")["order_id"].nunique().sort_values(ascending=False)

    # Mengelompokkan berdasarkan hari dan waktu pembelian, menghitung jumlah pesanan unik, dan mengurutkan hasil
    daily_hourly_orders = orders.groupby(["hari_pembelian", 'waktu_hari_pembelian'])['order_id'].nunique().sort_values(ascending=False).reset_index()

    # Menambahkan conclusion
    st.subheader("Conclusion: Durasi Pengiriman Paket Terlama dan Asal-Tujuan Pengiriman")
    st.write("Setelah mengidentifikasi dan menghilangkan outlier dalam data pengiriman, rata-rata pengiriman terlama antar kota mencapai 24.3 hari. Pengiriman terlama terjadi dari kota Sao Jose dos Campos ke Belem. Sementara itu, untuk pengiriman antar negara bagian, waktu pengiriman terlama adalah 27.69 hari dari state SP ke state RR. Informasi ini dapat menjadi dasar untuk peningkatan efisiensi dalam rantai pasok.")

# Menambahkan blok kode untuk pertanyaan 4 pada halaman yang dipilih
elif selected_page == "Pertanyaan 4":
    st.header("Pertanyaan 4: Analisis Pembayaran")

    # Menampilkan visualisasi menggunakan bar plot
    st.subheader("Rata-rata Nilai Pembayaran untuk Setiap Jenis Pembayaran")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Average Payment Value", y="Payment Type", data=df_payment.sort_values(by="Average Payment Value", ascending=False), palette="viridis", ax=ax)
    ax.set_xlabel("Rata-rata Nilai Pembayaran", fontsize=12, labelpad=10)
    ax.set_ylabel("Jenis Pembayaran", fontsize=12, labelpad=10)
    ax.set_title("Rata-rata Nilai Pembayaran untuk Setiap Jenis Pembayaran", loc="center", fontsize=14, pad=20)
    st.pyplot(fig)

    # Menampilkan distribusi jumlah transaksi untuk setiap jenis pembayaran menggunakan pie chart
    st.subheader("Distribusi Jumlah Transaksi untuk Setiap Jenis Pembayaran")

    # Menghitung jumlah pesanan unik untuk setiap tipe pembayaran
    df_payment = orders.groupby(by="payment_type")["order_id"].nunique().reset_index()

    # Mengatur palet warna
    palette_color = sns.color_palette('Set2')

    fig, ax = plt.subplots(figsize=(8, 8))
    explode = (0.1, 0, 0, 0)
    plt.pie(df_payment["order_id"], labels=df_payment["payment_type"], colors=palette_color, autopct='%.0f%%', explode=explode, startangle=140, wedgeprops=dict(width=0.3))
    plt.title("Distribusi Jumlah Transaksi untuk Setiap Jenis Pembayaran")
    plt.legend(title="Jenis Pembayaran", labels=df_payment["payment_type"], loc="upper left", bbox_to_anchor=(1, 0.8))
    st.pyplot(fig)

    # Menambahkan conclusion
    st.subheader("Conclusion: Rata-Rata Payment Value Tiap Tipe Transaksi dan Frekuensi Penggunaan")
    st.write("Dalam hal transaksi, konsumen cenderung lebih sering menggunakan tipe transaksi credit, yang mencakup 75% dari total transaksi. Rata-rata payment value untuk tipe transaksi ini sebesar 163.022616. Ini mungkin menunjukkan bahwa sebagian besar konsumen memilih pembayaran kredit untuk pembelian mereka.")

elif selected_page == "Pertanyaan 5":
    st.header("Perbandingan penjualan antara tahun 2017 dan 2018")

    # Palet warna yang diubah
    custom_palette = ["#FF6F61", "#43BFC7"]

    # Membuat bar plot menggunakan seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(x='nama_bulan', y='order_id', hue='year', data=df_monthly_orders, palette=custom_palette)
    
    # Menambahkan label pada sumbu-y
    plt.ylabel("Total Order", fontsize=12)
    
    # Menambahkan label pada sumbu-x
    plt.xlabel("Bulan", fontsize=12)

    # Menampilkan plot
    st.pyplot(plt)

    # Menghitung total order per tahun
    df_monthly_orders = df_monthly_orders.groupby("year")["order_id"].sum().reset_index()

    # Jumlah order pada tahun 2017
    orders_2017 = 21364

    # Jumlah order pada tahun 2018
    orders_2018 = 51461

    # Menghitung persentase kenaikan dari 2017 ke 2018
    percentage_increase = ((orders_2018 - orders_2017) / orders_2017) * 100

    # Data untuk visualisasi
    years = ['2017', '2018']
    orders_count = [orders_2017, orders_2018]

    # Setel gaya seaborn untuk tampilan yang lebih menarik
    sns.set(style="whitegrid")

    # Setel ukuran gambar
    plt.figure(figsize=(8, 5))

    # Gunakan barplot dengan warna yang kontras
    colors = sns.color_palette("husl", len(years))
    sns.barplot(x=years, y=orders_count, palette=colors)

    # Tambahkan label dan judul
    plt.xlabel('Tahun', fontsize=10)
    plt.ylabel('Jumlah Pesanan', fontsize=10)
    plt.title('Kenaikan Jumlah Pesanan dari 2017 ke 2018', fontsize=12)

    # Tampilkan nilai persentase kenaikan sebagai anotasi
    plt.text(0.5, 20000, f'{percentage_increase:.2f}% Increase', ha='center', va='bottom', fontsize=8, color='black')

    # Hapus spines pada sumbu y
    sns.despine(left=True)

    # Menampilkan plot
    st.pyplot(plt)

    # Menambahkan conclusion
    st.subheader("Conclusion: Perbandingan Penjualan Tahun 2017 dan 2018")
    st.write("Tren penjualan menunjukkan peningkatan yang signifikan pada tahun 2018 dibandingkan dengan tahun sebelumnya, yakni meningkat sebesar 140.87%. Hal ini dapat disebabkan oleh faktor-faktor seperti pertumbuhan pasar, strategi pemasaran yang berhasil, atau peningkatan kesadaran konsumen terhadap produk dan layanan yang ditawarkan.")


elif selected_page == "Pertanyaan 6":
    st.header("Bulan terjadi peningkatan penjualan yang paling signifikan dan faktor yang mungkin mempengaruhinya")

    # Mengelompokkan data berdasarkan bulan dan tahun serta menghitung jumlah order_id yang unik
    monthly_sales = orders.groupby(by=["month", "year"])["order_id"].nunique().reset_index()

    # Mengonversi kolom "month" ke format tanggal menggunakan pd.to_datetime
    monthly_sales["month"] = pd.to_datetime(monthly_sales["month"], format='%m-%Y')

    # Setel ukuran gambar
    plt.figure(figsize=(12, 6))

    # Buat line plot menggunakan seaborn dengan nuansa 'dark' dan marker 'o'
    ax = sns.lineplot(x='month', y='order_id', data=monthly_sales, estimator=None, linewidth=3, color='darkblue', marker='o')

    # Setel posisi x-ticks agar sesuai dengan nilai bulan
    ax.set(xticks=monthly_sales.month.values)

    # Atur judul dan label sumbu
    plt.title("Sales Growth Trend", loc="center", fontsize=14)
    plt.ylabel("Total Orders")
    plt.xlabel(None)

    # Matikan grid pada sumbu y
    ax.grid(False)

    # Rotasi label bulan agar lebih mudah dibaca
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    # Menampilkan plot
    st.pyplot(plt)

    # Menambahkan conclusion
    st.subheader("Conclusion: Bulan dengan Peningkatan Penjualan Tertinggi")
    st.write("Analisis menunjukkan bahwa bulan November 2017 menjadi bulan dengan penjualan tertinggi. Fokus pada tanggal 24 November 2017 mungkin menunjukkan adanya acara promosi atau penawaran khusus yang memicu peningkatan signifikan dalam aktivitas pembelian.")

elif selected_page == "Pertanyaan 7":
    st.header("Hari apa yang paling sering dipilih oleh pembeli untuk melakukan transaksi")

    # Mengelompokkan data berdasarkan bagian hari pembelian dan menghitung total order_id yang unik
    daily_orders = orders.groupby(by="waktu_hari_pembelian")["order_id"].nunique().reset_index()

    # Memberi nama ulang kolom order_id menjadi total_orders
    daily_orders.rename(columns={"order_id": "total_orders"}, inplace=True)

    # Setel gaya seaborn untuk tampilan yang lebih menarik
    sns.set(style="whitegrid")

    # Setel ukuran gambar yang lebih kecil
    plt.figure(figsize=(8, 4))

    # Warna yang lebih berbeda untuk memperjelas
    colors = ["#D3D3D3", "#D3D3D3", "#4C72B0", "#D3D3D3"]

    # Gunakan barplot dengan orientasi horizontal (barh)
    ax = sns.barplot(
        y="waktu_hari_pembelian",
        x="total_orders",
        data=daily_orders.sort_values(by="total_orders"),
        palette=colors
    )

    # Tambahkan anotasi pada setiap bar
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.0f}', (p.get_width() + 2, p.get_y() + p.get_height() / 2),
                    ha='center', va='center', color='black', fontsize=8)

    # Tambahkan judul dan label
    plt.title("Distribution of Purchases by Time of Day", loc="center", fontsize=10)
    plt.xlabel("Total Orders")
    plt.ylabel(None)

    # Hapus spines pada sumbu y
    sns.despine(left=True)

    # Tampilkan plot
    st.pyplot(plt)

    # Mengelompokkan data berdasarkan hari pembelian dan menghitung total order_id yang unik
    daywise_orders = orders.groupby(by="hari_pembelian")["order_id"].nunique().sort_values(ascending=False).reset_index()

    # Memberi nama ulang kolom order_id menjadi total_orders
    daywise_orders.rename(columns={"order_id": "total_orders"}, inplace=True)

    # Setel gaya seaborn untuk tampilan yang lebih menarik
    sns.set(style="whitegrid")

    # Buat subplot dengan ukuran tertentu
    plt.figure(figsize=(8, 4))

    # Urutkan DataFrame berdasarkan total_orders secara descending
    daywise_orders_sorted = daywise_orders.sort_values(by="total_orders", ascending=False)

    # Gunakan barplot dengan orientasi horizontal (barh) dan warna yang berbeda
    colors = ["#4C72B0", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    sns.barplot(
        y="hari_pembelian",
        x="total_orders",
        data=daywise_orders_sorted,
        palette=colors
    )

    plt.title("Distribution of Orders by Day of the Week", loc="center", fontsize=12)
    plt.xlabel("Total Orders")
    plt.ylabel("Day of the Week")
    plt.tick_params(axis='y', labelsize=10)
    
    # Tampilkan plot
    st.pyplot(plt)

    # Menambahkan conclusion
    st.subheader("Conclusion: Bagian Hari yang Sering Digunakan untuk Bertransaksi")
    st.write("Dari segi waktu, konsumen cenderung lebih aktif berbelanja pada hari Senin, menunjukkan bahwa awal minggu menjadi pilihan favorit untuk melakukan transaksi. Selain itu, penekanan pada aktivitas pembelian yang tinggi selama periode siang hari memberikan wawasan tambahan tentang kebiasaan belanja konsumen.")



