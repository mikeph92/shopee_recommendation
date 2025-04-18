import streamlit as st
import base64
import importlib
import pandas as pd
import gc

# Try to import memory monitoring, but continue if not available
try:
    from utils.memory_monitor import display_memory_usage, add_memory_cleanup_button
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False
    print("Memory monitoring not available")


# ===== Sidebar background =====
def sidebar_bg(side_bg_path):
    ext = side_bg_path.split('.')[-1]
    with open(side_bg_path, "rb") as f:
        side_bg = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/{ext};base64,{side_bg});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    try:
        # Load data in smaller chunks to manage memory
        chunk_size = 1000  # Reduced from 5000 to 1000 for better memory management
        
        # Load only necessary columns - minimize data exposure
        usecols = ['product_id', 'product_name', 'sub_category', 'price', 'rating', 'description', 'image', 'link']
        
        # Load products data with dtype optimization
        dtypes = {
            'product_id': 'int32',
            'price': 'float32',
            'rating': 'float32'
        }
        
        products_chunks = []
        try:
            for chunk in pd.read_csv('data/Products_ThoiTrangNam_clean_part1.csv', 
                                   chunksize=chunk_size, 
                                   usecols=usecols,
                                   dtype=dtypes):
                # Sanitize sensitive data
                chunk['description'] = chunk['description'].apply(lambda x: str(x)[:200] + '...' if len(str(x)) > 200 else str(x))
                products_chunks.append(chunk)
                gc.collect()  # Force garbage collection after each chunk
                
            for chunk in pd.read_csv('data/Products_ThoiTrangNam_clean_part2.csv', 
                                   chunksize=chunk_size,
                                   usecols=usecols,
                                   dtype=dtypes):
                # Sanitize sensitive data
                chunk['description'] = chunk['description'].apply(lambda x: str(x)[:200] + '...' if len(str(x)) > 200 else str(x))
                products_chunks.append(chunk)
                gc.collect()  # Force garbage collection after each chunk
                
            products_clean = pd.concat(products_chunks, ignore_index=True)
            
            # Clear chunks from memory
            del products_chunks
            gc.collect()
            
            # Load ratings data with optimized dtypes and minimal data
            rating_clean = pd.read_csv("data/Products_ThoiTrangNam_rating_clean.csv", 
                                     sep='\t',
                                     usecols=['user_id', 'product_id', 'rating'],
                                     dtype={'user_id': 'int32', 
                                           'product_id': 'int32',
                                           'rating': 'float32'})
            
            return products_clean, rating_clean
            
        except FileNotFoundError:
            st.error("Data files not found. Please check the data directory.")
            return None, None
            
    except Exception as e:
        # Generic error message to prevent data leakage
        st.error("An error occurred while loading data. Please try again later.")
        return None, None

# ===== Giao diện chính =====
def main():
    sidebar_bg("images/bg.png")

    # Ẩn stSidebarNav
    hide_sidebar_style = '''
        <style>
        [data-testid="stSidebarNav"] {
            display:none !important;
        }
        </style>
        '''
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)

    # Tải lên file dữ liệu
    products_clean, rating_clean = load_data()

    
    # Menu: ánh xạ Tên hiển thị → (file, hàm)
    menu = {
        "🏠 Trang chủ": ("home", "home", []),  # không có tham số
        "📊 Giới thiệu chung": ("general_content", "general_content", []),
        "📈 Khám phá dữ liệu": ("data_insight", "data_insight", [products_clean, rating_clean]),
        "🎯 Gợi ý sản phẩm": ("recommendation", "product_recommendation", [products_clean, rating_clean])
    }

    st.sidebar.title("📌 Chức năng")
    selected = st.sidebar.radio("Chọn trang:", list(menu.keys()))

    # Add memory monitoring if available
    if MEMORY_MONITOR_AVAILABLE:
        display_memory_usage()
        add_memory_cleanup_button()

    # Gọi đúng module và hàm theo lựa chọn
    module_name, function_name, params = menu[selected]
    module = importlib.import_module(f"pages.{module_name}")
    getattr(module, function_name)(*params)

    # Footer nhóm
    st.sidebar.markdown("""
    <div style="margin-top: 200px; background-color: #e0f2f1; padding: 15px; border-radius: 8px;">
        <strong>DL07 – K302 – April 2025</strong><br>
        Hàn Thảo Anh<br>
        Nguyễn Thị Thùy Trang<br>
        👩‍🏫 <strong>GVHD: Cô Khuất Thùy Phương</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()