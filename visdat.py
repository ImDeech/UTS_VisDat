# ======================================================
# BUSINESS DASHBOARD VISUALIZATION
# Dataset: Copy of finalProj_df - 2022.csv
# Author: Devin (adapted from GPT-5)
# ======================================================

# --- Import Library ---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load Dataset ---
df = pd.read_csv(r"D:\Kuliah\Semester5\VisualisasiData\buat-uts\Copy of finalProj_df - 2022.csv")

# --- Preprocessing ---
df['order_date_parsed'] = pd.to_datetime(df['order_date'], errors='coerce')
df['month'] = df['order_date_parsed'].dt.to_period('M')

# Buat kolom tambahan
df['revenue_after_discount'] = df['after_discount']
df['gross_margin'] = df['after_discount'] - df['cogs']

# --- KPI Summary ---
total_revenue = df['revenue_after_discount'].sum()
total_orders = df['id'].nunique()
aov = total_revenue / total_orders
total_discount = df['discount_amount'].sum()
gross_margin = df['gross_margin'].sum()

print("===== KPI SUMMARY =====")
print(f"Total Revenue: {total_revenue:,.0f}")
print(f"Total Orders: {total_orders}")
print(f"Average Order Value (AOV): {aov:,.0f}")
print(f"Total Discount: {total_discount:,.0f}")
print(f"Gross Margin: {gross_margin:,.0f}")

# --- Agregasi per bulan ---
monthly = df.groupby('month').agg({
    'revenue_after_discount':'sum',
    'id':'nunique',
    'gross_margin':'sum'
}).rename(columns={'id':'orders'})

# --- Top 10 SKU by Revenue ---
top_sku = df.groupby('sku_name')['revenue_after_discount'].sum().nlargest(10)

# --- Scatter Sample (Price vs Quantity) ---
df_scatter = df.sample(min(1000, len(df)), random_state=42)

# --- Correlation Heatmap Data ---
corr = df[['price','qty_ordered','discount_amount','after_discount','cogs','gross_margin']].corr()

# ======================================================
# STYLE SETTINGS
# ======================================================
plt.style.use('seaborn-v0_8-whitegrid')
primary_color = '#2E86C1'      # Biru profesional
secondary_color = '#AED6F1'    # Biru muda
accent_color = '#1F618D'       # Biru tua

# ======================================================
# 1️⃣ KPI Summary Bar Chart
# ======================================================
fig, ax = plt.subplots(figsize=(8,5))
kpi_labels = ['Revenue', 'Orders', 'AOV', 'Total Discount', 'Gross Margin']
kpi_values = [total_revenue, total_orders, aov, total_discount, gross_margin]

bars = ax.bar(kpi_labels, kpi_values, color=primary_color)
ax.set_title("KPI Summary", fontsize=14, fontweight='bold')
ax.bar_label(bars, fmt='%.0f', fontsize=8, rotation=45)
plt.tight_layout()
plt.show()

# ======================================================
# 2️⃣ Time Series (Revenue & Orders)
# ======================================================
fig, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(monthly.index.to_timestamp(), monthly['revenue_after_discount'], 
         color=primary_color, marker='o', label='Revenue')
ax1.set_ylabel('Revenue', color=primary_color)
ax2 = ax1.twinx()
ax2.plot(monthly.index.to_timestamp(), monthly['orders'], 
         color='#B03A2E', marker='s', label='Orders')
ax2.set_ylabel('Orders', color='#B03A2E')
ax1.set_title('Monthly Revenue & Orders Trend', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()

# ======================================================
# 3️⃣ Top 10 SKU by Revenue
# ======================================================
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.barh(top_sku.index[::-1], top_sku.values[::-1], color=accent_color)
ax.set_title("Top 10 SKU by Revenue", fontsize=14, fontweight='bold')
ax.set_xlabel("Revenue")
plt.tight_layout()
plt.show()

# ======================================================
# 4️⃣ Scatter Plot: Price vs Quantity Ordered
# ======================================================
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(df_scatter['price'], df_scatter['qty_ordered'], 
           alpha=0.6, color=secondary_color, edgecolor=accent_color)
ax.set_title("Price vs Quantity Ordered", fontsize=14, fontweight='bold')
ax.set_xlabel("Price")
ax.set_ylabel("Quantity Ordered")
plt.tight_layout()
plt.show()

# ======================================================
# 5️⃣ Correlation Heatmap
# ======================================================
fig, ax = plt.subplots(figsize=(6,5))
cax = ax.matshow(corr, cmap='Blues')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='left')
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)
ax.set_title("Correlation Heatmap", pad=20, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
