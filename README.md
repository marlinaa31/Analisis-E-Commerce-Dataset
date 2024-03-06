# Brazilian E-Commerce Public Dataset

# Analisis_dicoding

**Final Project Data Analysis**

## Setup Environment

```
conda create --name myenv python=3.9.18
conda activate myenv
pip install numpy pandas matplotlib seaborn streamlit
```

## Run Streamlit App

```
streamlit run dashboard.py
```

## Dataset Overview

### Introduction

This is a Brazilian ecommerce public dataset of orders made at Olist Store. The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allows viewing an order from multiple dimensions: from order status, price, payment and freight performance to customer location, product attributes and finally reviews written by customers. We also released a geolocation dataset that relates Brazilian zip codes to lat/lng coordinates.

This is real commercial data, it has been anonymised, and references to the companies and partners in the review text have been replaced with the names of Game of Thrones great houses.

## Data Structure

The Brazilian E-Commerce dataset from Kaggle consists of several CSV files that cover various aspects of the e-commerce system in Brazil. Here is an explanation of the main features in this dataset:

1. **Olist_Customers_Dataset.csv:**
   - `customer_id`: Unique customer ID.
   - `customer_unique_id`: Unique and anonymous ID for each customer.
   - `customer_zip_code_prefix`: Customer's zip code prefix.
   - `customer_city`: Customer's city.
   - `customer_state`: Customer's state.

2. **Olist_Geolocation_Dataset.csv:**
   - `geolocation_zip_code_prefix`: Zip code prefix.
   - `geolocation_lat`: Latitude of the geographic location.
   - `geolocation_lng`: Longitude of the geographic location.
   - `geolocation_city`: City.
   - `geolocation_state`: State.

3. **Olist_Order_Items_Dataset.csv:**
   - `order_id`: Unique order ID.
   - `order_item_id`: Unique item ID within an order.
   - `product_id`: Unique product ID.
   - `seller_id`: Unique seller ID.
   - `shipping_limit_date`: Shipping deadline.
   - `price`: Product price.
   - `freight_value`: Shipping cost.

4. **Olist_Order_Payments_Dataset.csv:**
   - `order_id`: Unique order ID.
   - `payment_sequential`: Payment sequence in one order.
   - `payment_type`: Payment method.
   - `payment_installments`: Number of payment installments.
   - `payment_value`: Payment value.

5. **Olist_Orders_Dataset.csv:**
   - `order_id`: Unique order ID.
   - `customer_id`: Unique customer ID.
   - `order_status`: Order status (processing, shipped, delivered, etc.).
   - `order_purchase_timestamp`: Order purchase time.
   - `order_approved_at`: Time of order approval.
   - `order_delivered_carrier_date`: Courier delivery time.
   - `order_delivered_customer_date`: Time the order was received by the customer.
   - `order_estimated_delivery_date`: Estimated delivery time.

6. **Olist_Products_Dataset.csv:**
   - `product_id`: Unique product ID.
   - `product_category_name`: Product category name.
   - `product_name_length`: Length of the product name.
   - `product_description_length`: Length of the product description.
   - `product_photos_qty`: Number of product photos.
   - `product_weight_g`: Product weight in grams.
   - `product_length_cm`: Product length in centimeters.
   - `product_height_cm`: Product height in centimeters.
   - `product_width_cm`: Product width in centimeters.

7. **Olist_Sellers_Dataset.csv:**
   - `seller_id`: Unique seller ID.
   - `seller_zip_code_prefix`: Seller's zip code prefix.
   - `seller_city`: Seller's city.
   - `seller_state`: Seller's state.

8. **Product_Category_Name_Translation.csv:**
   - `product_category_name`: Product category name in Portuguese.
   - `product_category_name_english`: Translation of the product category name into English.

This dataset provides broad insights into e-commerce transactions, customers, products, and sellers in Brazil. By combining data from various tables, comprehensive analysis can be conducted to understand trends and patterns in e-commerce activities.

## Merging Dataframes

In this project, there are two main dataframes that will be used, namely `df_order_items` and `orders`.

- `df_order_items`: A combination of the order_items, products_translation, and seller tables, providing details about item-orders, product translations, and seller information.

- `orders`: A combination of the orders, payments, and customer tables, offering a holistic overview of orders, payments, and customer data.

The integration of these two dataframes will enable a more comprehensive analysis of the entire project's processes and transactions.