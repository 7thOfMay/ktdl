import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="Product Recommendation System", layout="wide")

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("new_data_to_analysis.csv")
    # Drop duplicates for SKU to ensure unique products
    df = df.drop_duplicates(subset=["SKU"], keep="first")
    df = df.reset_index(drop=True)
    return df

df = load_data()

# === Preprocessing & Models ===

@st.cache_resource
def train_clustering_model(df):
    # Logic from Clutering.py
    cat_cols = [c for c in ["Category","Style","Size"] if c in df.columns]
    num_cols = [c for c in ["Amount"] if c in df.columns]
    
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )
    
    # Fit transform
    X = ct.fit_transform(df[cat_cols + num_cols])
    
    # Clustering
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=1024, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Return everything needed for prediction
    return X, labels, kmeans, ct

X_features, cluster_labels, kmeans_model, column_transformer = train_clustering_model(df)
df["Cluster"] = cluster_labels

# === Recommendation Functions ===

def recommend_apriori_logic(product_info, df, min_price, max_price, top_n=5):
    """
    Logic from Apriori.py: 
    - Similarity Score based on Category, Core, Size, Price Level.
    - Filter by Score >= 2.
    - Filter by Amount Diff <= price_tol (using min/max price here as range).
    """
    core = product_info["Core"]
    price_level = product_info["price_level"]
    size = product_info["Size"]
    category = product_info["Category"]
    amount = product_info["Amount"]
    sku = product_info.get("SKU", "INPUT_PRODUCT")

    # Filter candidates (exclude self)
    candidates = df[df["SKU"] != sku].copy()
    
    # Calculate Similarity Score
    # (candidates["Category"] == category).astype(int) + ...
    candidates["similarity_score"] = (
        (candidates["Category"] == category).astype(int) +
        (candidates["Core"] == core).astype(int) +
        (candidates["Size"] == size).astype(int) +
        (candidates["price_level"] == price_level).astype(int)
    )
    
    # Filter Score >= 2
    candidates = candidates[candidates["similarity_score"] >= 2]
    
    # Filter by Price Range (User Input)
    candidates = candidates[
        (candidates["Amount"] >= min_price) & 
        (candidates["Amount"] <= max_price)
    ]
    
    # Calculate Amount Diff
    candidates["amount_diff"] = abs(candidates["Amount"] - amount)
    
    # Sort by Score (Desc), Amount Diff (Asc)
    candidates = candidates.sort_values(by=["similarity_score", "amount_diff"], ascending=[False, True])
    
    return candidates.head(top_n)

def recommend_clustering_logic(product_info, df, X, ct, kmeans, min_price, max_price, top_n=5):
    """
    Logic from Clutering.py:
    - Predict cluster.
    - Filter same cluster.
    - Strict filter: Category, Size, Core must match.
    - Filter by Price Range.
    - Sort by Amount Diff (Asc), Cosine Sim (Desc).
    """
    # Prepare input for clustering
    # Create a DataFrame for the single input to pass to ColumnTransformer
    input_df = pd.DataFrame([product_info])
    
    # Transform input
    try:
        input_vec = ct.transform(input_df)
    except Exception as e:
        st.error(f"Error transforming input: {e}")
        return pd.DataFrame()

    # Predict cluster
    cluster_label = kmeans.predict(input_vec)[0]
    
    # Filter same cluster
    candidates = df[df["Cluster"] == cluster_label].copy()
    candidates = candidates[candidates["SKU"] != product_info.get("SKU", "INPUT")]
    
    # Strict Filter: Category, Size, Core
    candidates = candidates[
        (candidates["Category"] == product_info["Category"]) &
        (candidates["Size"] == product_info["Size"]) &
        (candidates["Core"] == product_info["Core"])
    ]
    
    # Filter by Price Range
    candidates = candidates[
        (candidates["Amount"] >= min_price) & 
        (candidates["Amount"] <= max_price)
    ]
    
    if candidates.empty:
        return pd.DataFrame()

    # Calculate Cosine Similarity
    # We need vectors for candidates. 
    # Since X corresponds to df index, we can fetch rows by index.
    # Ensure df index is aligned with X.
    candidate_indices = candidates.index
    candidate_vecs = X[candidate_indices]
    
    # Cosine Sim between input_vec and candidate_vecs
    # input_vec is (1, n_features), candidate_vecs is (m, n_features)
    cos_sim = cosine_similarity(input_vec, candidate_vecs).ravel()
    
    candidates["cosine_sim"] = cos_sim
    candidates["amount_diff"] = abs(candidates["Amount"] - product_info["Amount"])
    
    # Sort: Amount Diff (Asc), Cosine Sim (Desc)
    candidates = candidates.sort_values(by=["amount_diff", "cosine_sim"], ascending=[True, False])
    
    return candidates.head(top_n)

# === UI ===

st.title("Product Recommendation System")

# Increase font size for Tab labels
st.markdown("""
<style>
    /* Target the tab labels specifically */
    div[data-baseweb="tab-list"] p {
        font-size: 26px !important;
        font-weight: bold !important;
    }
    /* Fallback for older Streamlit versions or different structures */
    div[data-baseweb="tab-list"] button {
        font-size: 26px !important;
    }
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["Recommendation Demo", "Model Analysis"])

with tabs[0]:
    # Sidebar
    st.sidebar.header("Input Configuration")
    
    # Move Price Range to top of sidebar
    st.sidebar.subheader("Price Range")
    
    col_min, col_max = st.sidebar.columns(2)
    with col_min:
        min_p = st.number_input("Min Price", min_value=0.0, value=0.0)
    with col_max:
        max_p = st.number_input("Max Price", min_value=0.0, value=float(df["Amount"].max()))

    st.sidebar.markdown("---")

    # CSS to increase Radio Button size
    st.markdown("""
    <style>
    div[data-testid="stRadio"] > label {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] p {
        font-size: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    input_method = st.sidebar.radio("Choose Input Method", ["Select Existing Product", "Manual Input"])

    selected_product = None

    if input_method == "Select Existing Product":
        sku_list = df["SKU"].tolist()
        selected_sku = st.sidebar.selectbox("Select SKU", sku_list)
        if selected_sku:
            selected_product = df[df["SKU"] == selected_sku].iloc[0].to_dict()
            st.sidebar.write("Selected Product Details:")
            for key, value in selected_product.items():
                st.sidebar.text(f"{key}: {value}")

    else:
        st.sidebar.subheader("Enter Product Features")
        
        # Compact layout using columns
        sc1, sc2 = st.sidebar.columns(2)
        with sc1:
            category = st.selectbox("Category", df["Category"].unique())
            size = st.selectbox("Size", df["Size"].unique())
            price_level = st.selectbox("Price Level", df["price_level"].unique())
        with sc2:
            style = st.text_input("Style", "Unknown")
            core = st.selectbox("Core", df["Core"].unique())
            amount = st.number_input("Amount (Price)", min_value=0.0, value=500.0)
        
        selected_product = {
            "SKU": "MANUAL_INPUT",
            "Category": category,
            "Style": style,
            "Size": size,
            "Core": core,
            "price_level": price_level,
            "Amount": amount
        }

    # Main Content
    if selected_product:
        # CSS for centering
        st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                text-align: center !important;
            }
            [data-testid="stMetricLabel"] {
                text-align: center !important;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='text-align: center;'>Target Product</h3>", unsafe_allow_html=True)
        
        # 1. Visual Hierarchy & Grid: 4 separate cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            with st.container(border=True):
                st.metric(label="Category", value=selected_product["Category"])
        with c2:
            with st.container(border=True):
                st.metric(label="Size", value=selected_product["Size"])
        with c3:
            with st.container(border=True):
                st.metric(label="Core", value=selected_product["Core"])
        with c4:
            with st.container(border=True):
                st.metric(label="Price", value=f"{selected_product['Amount']:.2f}")

        st.markdown("<h3 style='text-align: center; margin-top: 30px;'>Recommendations</h3>", unsafe_allow_html=True)
        
        # 2. Grid Layout & Whitespace: Use columns with gap
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            # 3. Visual Hierarchy: Use container as a 'Card' for Method 1
            with st.container(border=True):
                st.markdown("<h4 style='text-align: center;'>Method 1: Content-Based</h4>", unsafe_allow_html=True)
                st.divider() # Whitespace/Separation
                
                recs_1 = recommend_apriori_logic(selected_product, df, min_p, max_p)
                if not recs_1.empty:
                    # Clean up dataframe display
                    display_df_1 = recs_1[["SKU", "Amount", "similarity_score", "amount_diff"]]
                    st.dataframe(
                        display_df_1, 
                        hide_index=True, 
                        use_container_width=True,
                        column_config={
                            "Amount": st.column_config.NumberColumn("Price", format="$%.2f"),
                            "similarity_score": st.column_config.ProgressColumn("Match Score", min_value=0, max_value=4, format="%d/4"),
                            "amount_diff": st.column_config.NumberColumn("Diff", format="$%.2f"),
                        }
                    )
                else:
                    st.warning("No matches found in this range.")

        with col2:
            # 3. Visual Hierarchy: Use container as a 'Card' for Method 2
            with st.container(border=True):
                st.markdown("<h4 style='text-align: center;'>Method 2: Clustering</h4>", unsafe_allow_html=True)
                st.divider()
                
                recs_2 = recommend_clustering_logic(selected_product, df, X_features, column_transformer, kmeans_model, min_p, max_p)
                if not recs_2.empty:
                    display_df_2 = recs_2[["SKU", "Amount", "cosine_sim", "amount_diff"]]
                    st.dataframe(
                        display_df_2, 
                        hide_index=True, 
                        use_container_width=True,
                        column_config={
                            "Amount": st.column_config.NumberColumn("Price", format="$%.2f"),
                            "cosine_sim": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1),
                            "amount_diff": st.column_config.NumberColumn("Diff", format="$%.2f"),
                        }
                    )
                else:
                    st.warning("No matches found in this range.")

with tabs[1]:
    st.header("Model Analysis")
    
    st.subheader("Apriori Association Rules")
    if st.button("Calculate Apriori Rules (This may take time)"):
        with st.spinner("Calculating..."):
            cols = ["Category", "Style", "Size", "Core", "price_level"]
            # Limit rows for speed in demo if needed, but let's try full
            df_hot = pd.get_dummies(df[cols])
            frequent_itemsets = apriori(df_hot, min_support=0.1, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules = rules.sort_values(by="lift", ascending=False)
            st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(20))
            
    st.markdown("---")
    st.subheader("Clustering Centers")
    st.write("Cluster Centers (Scaled):")
    st.write(kmeans_model.cluster_centers_)

