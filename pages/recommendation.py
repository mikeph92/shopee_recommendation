import os
import warnings
import logging
import gc
import requests
from PIL import Image
from io import BytesIO
import time
# Hide all warnings 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold(absl.logging.ERROR)
import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from utils.content_based_top20 import *
from utils.collaborative import get_top_n_recommendations

# Configure TensorFlow to use memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_cb_model():
    try:
        path = "models/content_based_model.pkl"
        if not os.path.exists(path):
            st.error("Model file not found.")
            return None
            
        with st.spinner('Loading content-based model...'):
            model = joblib.load(path)
            # Optimize memory usage and sanitize data
            if 'tfidf_matrix' in model:
                model['tfidf_matrix'] = model['tfidf_matrix'].tocsr()  # Convert to CSR format
            gc.collect()  # Force garbage collection
            return model
    except Exception as e:
        # Generic error message
        st.error("An error occurred while loading the content-based model.")
        return None

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_cf_model():
    try:
        path_model = "models/matrix_factorizer_keras_model.h5"
        path_meta = "models/id_mappings.pkl"

        if not os.path.exists(path_model) or not os.path.exists(path_meta):
            st.error("Model files not found.")
            return None, None
        
        with st.spinner('Loading collaborative filtering model...'):
            # Load model with memory optimization
            with tf.device('/CPU:0'):  # Force CPU usage
                model = load_model(path_model, compile=False)
                # Convert model to TF-Lite format for smaller memory footprint
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                
            # Load metadata with minimal data
            meta = joblib.load(path_meta)
            if isinstance(meta, dict):
                # Only keep necessary mappings
                meta = {
                    'user_mapping': meta.get('user_mapping', {}),
                    'product_mapping': meta.get('product_mapping', {}),
                    'mu': meta.get('mu', 0.0)
                }
            
            # Clear any unused memory
            gc.collect()
            
            return tflite_model, meta
    except Exception as e:
        # Generic error message
        st.error("An error occurred while loading the collaborative filtering model.")
        return None, None

@st.cache_data(ttl=3600)
def load_and_cache_image(image_url):
    try:
        if not image_url or pd.isna(image_url):
            return "images/no_image.jpg"
            
        response = requests.get(image_url, timeout=5)
        if response.status_code == 200:
            # Verify it's an image
            Image.open(BytesIO(response.content))
            return image_url
        return "images/no_image.jpg"
    except Exception:
        return "images/no_image.jpg"

# ====== Hiển thị sản phẩm gợi ý ======
def display_recommendations(result_df, is_cb=True):
    if result_df is None or result_df.empty:
        st.warning("🙁 Không tìm thấy sản phẩm phù hợp.")
        return

    # Add pagination
    items_per_page = 5
    total_items = len(result_df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        col1, col2 = st.columns([3, 1])
        with col1:
            current_page = st.number_input('Trang', min_value=1, max_value=total_pages, value=1)
        with col2:
            st.markdown(f"**Tổng số trang: {total_pages}**")
        
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        current_items = result_df.iloc[start_idx:end_idx]
    else:
        current_items = result_df

    # Display current page items
    for _, row in current_items.iterrows():
        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                image_url = load_and_cache_image(row['image'])
                st.image(image_url, width=120, use_column_width=True)

            with cols[1]:
                mota = str(row['description'])
                short_desc = mota[:200] + "..." if len(mota) > 200 else mota

                st.markdown(f"""
                **🧢 Tên sản phẩm:** {row['product_name']}  
                **📦 Loại sản phẩm:** {row['sub_category']}  
                **💸 Giá:** {int(row['price']):,}₫  
                **⭐ Đánh giá:** {float(row['rating']):.1f}  
                **📖 Mô tả:** {short_desc}
                """)

                if is_cb and 'similarity' in row:
                    st.markdown(f"📊 **Độ tương đồng:** {float(row['similarity']):.3f}")
                elif not is_cb and 'predict' in row and float(row['predict']) > 0:
                    st.markdown(f"📊 **Dự đoán:** {float(row['predict']):.1f}")
                    
                st.markdown(f"""
                            <a href="{row['link']}" target="_blank">👉 Xem chi tiết sản phẩm</a>
                            """, unsafe_allow_html=True)
            st.markdown("---")
            
    # Show pagination info
    if total_pages > 1:
        st.markdown(f"*Hiển thị {start_idx + 1}-{end_idx} trên tổng số {total_items} sản phẩm*")

def display_carousel(result_df, num_cols=3):
    if result_df is None or result_df.empty:
        st.warning("🙁 Khách hàng chưa có lịch sử đánh giá sản phẩm.")
        return

    # Chia dữ liệu thành từng "hàng" (chunk), mỗi hàng có tối đa num_cols sản phẩm
    rows = [result_df[i:i+num_cols] for i in range(0, len(result_df), num_cols)]

    for row_chunk in rows:
        cols = st.columns(num_cols)

        for i in range(num_cols):
            with cols[i]:
                if i < len(row_chunk):
                    row = row_chunk.iloc[i]

                    st.markdown("---")
                    st.markdown(
                        f"""
                        <div style='text-align: right; margin-bottom: 0.5rem;'>
                            <a href="{row['link']}" target="_blank">🔗 Xem chi tiết</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    try:
                        st.image(row['image'], use_container_width=True)
                    except:
                        st.image("images/no_image.jpg", use_container_width=True)

                    st.markdown(f"**{row['product_name']}**")
                    st.markdown(f"💸 **{int(row['price']):,}₫**")
                    st.markdown(f"⭐ {float(row['rating']):.1f}")

                    short_desc = str(row['description'])[:100] + "..." if row['description'] else "Không có mô tả"
                    st.caption(short_desc)
                else:
                    st.write("")  # Cột trống nếu không còn sản phẩm

# ====== Giao diện chính gợi ý ======
def product_recommendation(products_df, ratings_df):
    st.header("🎯 Hệ thống gợi ý sản phẩm")

    if products_df is None or ratings_df is None:
        st.error("Không thể tải dữ liệu. Vui lòng thử lại sau.")
        return

    method = st.selectbox("🔍 Chọn phương pháp gợi ý:", ["Gợi ý theo nội dung", "Gợi ý theo người dùng"])
    
    # Load models
    if method == "Gợi ý theo nội dung":
        model_cb = load_cb_model()
        if model_cb is None:
            st.error("Không thể tải mô hình gợi ý theo nội dung.")
            return
            
        search_mode = st.radio("Chọn cách tìm kiếm:", ["Từ khóa", "Mã sản phẩm"])

        if search_mode == "Từ khóa":
            keyword = st.text_input("Nhập từ khóa (ví dụ: áo thun)")
            if st.button("Gợi ý", key="btn_cb_keyword"):
                with st.spinner("Đang tìm kiếm sản phẩm..."):
                    result = search_and_recommend(model_cb, products_df, keyword, top_k=10)
                    display_recommendations(result, is_cb=True)

        elif search_mode == "Mã sản phẩm":
            my_dict = list(model_cb["cosine_similarity"].items())
            unique_ids = [x[0] for x in my_dict]

            product_id = st.selectbox("Chọn mã sản phẩm:", unique_ids)

            if st.button("Gợi ý", key="btn_cb_product"):
                try:
                    with st.spinner("Đang tìm kiếm sản phẩm tương tự..."):
                        result = recommend_by_product_id(model_cb, products_df, product_id, top_k=10)
                        product = products_df[products_df['product_id'] == product_id]
                        
                        st.markdown(f"### 🛒 <span style='color:#1f77b4'>Sản phẩm đang xem:</span> <strong>{product_id}</strong>", unsafe_allow_html=True)
                        display_recommendations(product, is_cb=True)
                        
                        st.markdown("### 🎯 <span style='color:#2ca02c'>Sản phẩm gợi ý:</span>", unsafe_allow_html=True)
                        display_recommendations(result, is_cb=True)
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {str(e)}")

    elif method == "Gợi ý theo người dùng":
        model_cf, meta = load_cf_model()
        if model_cf is None or meta is None:
            st.error("Không thể tải mô hình gợi ý theo người dùng.")
            return
            
        try:
            # Create user mapping DataFrame safely
            user_map_df = ratings_df[['user_id', 'user']].drop_duplicates().reset_index(drop=True)
            user_map = dict(zip(user_map_df['user_id'].tolist(), user_map_df['user'].tolist()))
            
            st.subheader("👥 Danh sách người dùng và mã ID")
            st.dataframe(user_map_df, use_container_width=True)
            
            user_ids = list(user_map.keys())
            st.markdown("""
            ### 🔑 <span style='color:#0e76a8;'>Nhập mã khách hàng</span>  
            <small><i>Ví dụ: <code>5</code>, <code>290</code>, <code>777</code>, <code>20000</code></i></small>
            """, unsafe_allow_html=True)

            selected_user = st.text_input(" ", key="user_input")

            if st.button("Gợi ý", key="btn_cf_user"):
                if selected_user:
                    try:
                        user_id = int(selected_user)
                        if user_id not in user_ids:
                            st.error("⚠️ Mã khách hàng không tồn tại trong hệ thống hoặc chưa có lịch sử đánh giá!")
                        else:
                            user_name = user_map.get(user_id, "Không xác định")
                            
                            st.markdown(f"""
                            <div style="font-size:20px; font-weight:600;">
                                👤 <span style="color:#0e76a8;">Tên người dùng:</span> {user_name}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("####")
                            st.subheader("🛍️ Sản phẩm đã đánh giá:")
                            user_rated_df = ratings_df[ratings_df['user_id'] == user_id]
                            rated_products = products_df[products_df['product_id'].isin(user_rated_df['product_id'])].copy()
                            
                            if not rated_products.empty:
                                display_carousel(rated_products)
                                
                                with st.spinner("Đang tìm kiếm sản phẩm phù hợp..."):
                                    result = get_top_n_recommendations(
                                        product_df=products_df,
                                        user_id=user_id,
                                        model=model_cf,
                                        user_mapping=meta['user_mapping'],
                                        product_mapping=meta['product_mapping'],
                                        mu=meta['mu'],
                                        rated_products=rated_products,
                                        n=10
                                    )
                                    
                                st.markdown("##")
                                st.subheader("🎁 Gợi ý sản phẩm dựa trên hành vi người dùng:")
                                display_recommendations(result, is_cb=False)
                            else:
                                st.warning("Người dùng chưa có lịch sử đánh giá sản phẩm.")
                                
                    except ValueError:
                        st.error("❌ Mã khách hàng phải là một số nguyên!")
                    except Exception as e:
                        st.error("Có lỗi xảy ra khi tìm kiếm gợi ý. Vui lòng thử lại sau.")
                        logging.error(f"Error in collaborative filtering: {str(e)}")
                else:
                    st.error("⚠️ Vui lòng nhập Mã khách hàng!")
                    
        except Exception as e:
            st.error("Có lỗi xảy ra khi xử lý dữ liệu người dùng.")
            logging.error(f"Error in user data processing: {str(e)}")