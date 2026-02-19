"""
app_dashboard.py
Streamlit Business Dashboard (blue-gray theme) for the provided CSV dataset.
Drop this file in the same folder as your CSV (Copy of finalProj_df - 2022.csv)
Run: streamlit run app_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Financial Performance & Revenue Analytics Dashboard", layout="wide", page_icon="📊")

# Helper utilities
def first_col(df, candidates):
    """Return the first column name in df that matches any of the candidates (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def detect_and_standardize(df):
    """Detect common columns and add standardized columns used by the app."""
    # lower-trim column labels mapping for robust access
    orig_cols = df.columns.tolist()
    colmap = {c.lower().strip(): c for c in orig_cols}

    # rename a few known variants to a normalized lowercase name in the dataframe
    rename_map = {}
    # product name
    for cand in ["product_name", "product", "sku_name", "sku", "item_name"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "product_name"
            break
    # category
    for cand in ["category", "cat", "product_category"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "category"
            break
    # price
    for cand in ["price", "unit_price", "base_price"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "price"
            break
    # qty
    for cand in ["qty_ordered", "quantity", "qty", "quantity_ordered"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "quantity"
            break
    # before/after discount
    for cand in ["before_discount", "amount_before_discount", "subtotal"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "before_discount"
            break
    for cand in ["after_discount", "amount_after_discount", "total_after_discount", "revenue"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "after_discount"
            break
    # discount
    for cand in ["discount", "discount_amount", "amount_discount"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "discount_amount"
            break
    # cogs
    for cand in ["cogs", "cost_of_goods_sold", "cost"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "cogs"
            break
    # customer id
    for cand in ["customer_id", "customer", "buyer_id"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "customer_id"
            break
    # order id
    for cand in ["id", "order_id", "order"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "order_id"
            break
    # order date
    for cand in ["order_date", "date", "transaction_date", "created_at"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "order_date"
            break
    # year
    for cand in ["year"]:
        if cand in colmap:
            rename_map[colmap[cand]] = "year"
            break

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure parsed date and year exist
    if "order_date" in df.columns:
        try:
            df["order_date_parsed"] = pd.to_datetime(df["order_date"], errors="coerce")
            df["year_detected"] = df["order_date_parsed"].dt.year
        except Exception:
            df["order_date_parsed"] = pd.NaT
            df["year_detected"] = np.nan
    else:
        df["order_date_parsed"] = pd.NaT
        df["year_detected"] = np.nan

    # If explicit year column exists, use it; otherwise use detected year from date
    if "year" in df.columns:
        try:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        except Exception:
            df["year"] = df["year"]
    else:
        df["year"] = df["year_detected"].astype("Int64")

    # Create revenue / after_discount if missing
    if "after_discount" not in df.columns:
        # try compute: price * quantity - discount_amount (if price and quantity exist)
        if "price" in df.columns and "quantity" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
            df["after_discount"] = df["price"] * df["quantity"]
            if "discount_amount" in df.columns:
                df["after_discount"] = df["after_discount"] - pd.to_numeric(df["discount_amount"], errors="coerce").fillna(0)
        # else try before_discount - discount_amount
        elif "before_discount" in df.columns and "discount_amount" in df.columns:
            df["after_discount"] = pd.to_numeric(df["before_discount"], errors="coerce").fillna(0) - pd.to_numeric(df["discount_amount"], errors="coerce").fillna(0)
        else:
            # fallback: 0
            df["after_discount"] = 0.0

    # Ensure numeric types for key columns
    for c in ["after_discount", "before_discount", "discount_amount", "price", "quantity", "cogs"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # compute net_profit (gross profit) if possible
    if "net_profit" not in df.columns and "after_discount" in df.columns:
        if "cogs" in df.columns:
            df["net_profit"] = df["after_discount"] - df["cogs"]
        else:
            # no COGS column -> leave net_profit as NaN (we can fallback to 0 in aggregates)
            df["net_profit"] = np.nan

    # unify customer counts
    if "customer_id" in df.columns:
        # ensure consistent dtype
        df["customer_id"] = df["customer_id"].astype(str)
    return df

def format_currency(amount):
    try:
        amount = float(amount)
    except Exception:
        amount = 0.0
    if abs(amount) >= 1_000_000_000:
        return f"${amount/1_000_000_000:,.2f} B"
    elif abs(amount) >= 1_000_000:
        return f"${amount/1_000_000:,.2f} M"
    elif abs(amount) >= 1_000:
        return f"${amount/1_000:,.2f} K"
    else:
        return f"${amount:,.2f}"

# Load data
df = pd.read_csv("Copy of finalProj_df - 2022.csv")

# Standardize / detect columns
df = detect_and_standardize(df)

# Main layout: Header + KPI cards
st.markdown(
    """
    <style>
    .big-title {font-size:42px; font-weight:700; color:#1E3A8A; margin-bottom:6px;}
    .subtle {color:#6B7280;}
    .kpi-card {background:#ffffff; border-radius:10px; padding:18px; box-shadow: 0 1px 3px rgba(15,23,42,0.06);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">📊 Business Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Product Sales and Customer Insight</div>', unsafe_allow_html=True)
st.markdown("---")

# Compute KPIs
st.markdown(
    "<h3 style='text-align: center;'>Sales Performance Metrics</h3>",
    unsafe_allow_html=True)
total_revenue = df["after_discount"].sum() if "after_discount" in df.columns else 0
total_orders = df["order_id"].nunique() if "order_id" in df.columns else len(df)
total_customers = df["customer_id"].nunique() if "customer_id" in df.columns else df["customer_id"].nunique() if "customer_id" in df.columns else None
total_discount = df["discount_amount"].sum() if "discount_amount" in df.columns else 0
# net profit: use net_profit if exists else after_discount - cogs if possible
if "net_profit" in df.columns and not df["net_profit"].isna().all():
    total_profit = df["net_profit"].sum()
elif "after_discount" in df.columns and "cogs" in df.columns:
    total_profit = (df["after_discount"] - df["cogs"]).sum()
else:
    total_profit = np.nan

aov = total_revenue / total_orders if total_orders and total_orders > 0 else 0

# KPI cards row 1
k1, k2, k3, k4, k5, k6 = st.columns([1.3,1,1,1,1,1])
k1.markdown(f"<div class='kpi-card'><small class='subtle'>Before Discount</small><h3>{format_currency(df['before_discount'].sum() if 'before_discount' in df.columns else total_revenue)}</h3></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi-card'><small class='subtle'>After Discount</small><h3>{format_currency(total_revenue)}</h3></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi-card'><small class='subtle'>Net Profit</small><h3>{format_currency(total_profit) if not np.isnan(total_profit) else 'N/A'}</h3></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi-card'><small class='subtle'>Quantity</small><h3>{int(df['quantity'].sum()) if 'quantity' in df.columns else df.shape[0]}</h3></div>", unsafe_allow_html=True)
k5.markdown(f"<div class='kpi-card'><small class='subtle'>Customer</small><h3>{int(total_customers) if total_customers is not None else 'N/A'}</h3></div>", unsafe_allow_html=True)
k6.markdown(f"<div class='kpi-card'><small class='subtle'>Average Order Value</small><h3>{format_currency(aov)}</h3></div>", unsafe_allow_html=True)

st.markdown("---")

# Table (top area) and left charts
st.markdown(
    "<h3 style='text-align: center;'>Transaction and Profit Analysis Report</h3>",
    unsafe_allow_html=True)

# Tentukan kolom yang ingin ditampilkan
display_cols = []
for c in ["product_name", "category", "before_discount", "after_discount", "net_profit", "quantity", "customer_id", "order_id"]:
    if c in df.columns:
        display_cols.append(c)

if not display_cols:
    display_cols = df.columns.tolist()[:10]

# Tampilkan tabel secara penuh (melebar)
st.dataframe(
    df[display_cols].head(100),
    use_container_width=True,  # ini yang bikin tabel melebar penuh
    height=400
)

st.markdown("---")

# Charts area
c1, c2 = st.columns([1.2,1])

with c1:
    st.subheader("Quantity Ordered and Unique Customer")
    # bar chart by year
    if "year" in df.columns and df["year"].notna().any():
        bar_df = df.groupby("year").agg(quantity=("quantity","sum") if "quantity" in df.columns else ("order_id","count"),
                                        customers=("customer_id","nunique") if "customer_id" in df.columns else (df.columns[0],"count"))
        bar_df = bar_df.reset_index()
        bar_df['quantity'] = bar_df['quantity'].astype(int)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=bar_df['year'], y=bar_df['quantity'], name='Quantity Ordered', marker_color='#2E86C1'))
        fig_bar.add_trace(go.Bar(x=bar_df['year'], y=bar_df['customers'], name='Customer', marker_color='#A8D0F0'))
        fig_bar.update_layout(barmode='group', xaxis_title='Year', yaxis_title='Count')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No year data available to build the yearly bar chart.")

with c2:
    st.subheader("Top 10 SKU by Revenue")
    if "product_name" in df.columns and "after_discount" in df.columns:
        top10 = df.groupby("product_name")["after_discount"].sum().nlargest(10).sort_values()
        fig_top = px.bar(x=top10.values, y=top10.index, orientation='h', color=top10.values,
                         color_continuous_scale="Blues", labels={'x':'Revenue','y':'Product'})
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("Insufficient data to show top products (need product_name & after_discount).")

st.markdown("---")
st.markdown(
    "<h3 style='text-align: center;'>Korelasi Antar Variabel Numerik</h3>",
    unsafe_allow_html=True)
numeric = df.select_dtypes(include=[np.number])
if not numeric.empty:
    corr = numeric.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("No numeric columns found for correlation matrix.")

st.markdown("---")
