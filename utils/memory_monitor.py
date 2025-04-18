import streamlit as st
import psutil
import os
import gc
import numpy as np
import pandas as pd
import shutil
import tempfile

MEMORY_THRESHOLD = 85  # Percentage

def get_process_memory():
    process = psutil.Process(os.getpid())
    # Get memory info in bytes and convert to MB
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    return memory_mb

def display_memory_usage():
    # Get memory usage in MB
    memory_usage_mb = get_process_memory()
    
    # Get total system memory in MB
    total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
    
    # Calculate percentage for progress bar
    memory_percentage = (memory_usage_mb / total_memory_mb) * 100
    
    # Create a progress bar for memory usage
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.progress(min(memory_percentage/100, 1.0))
    with col2:
        st.write(f"{memory_usage_mb:.1f} MB")
        
    # Automatic cleanup if memory usage is too high
    if memory_percentage > MEMORY_THRESHOLD:
        perform_memory_cleanup()
        st.sidebar.warning("🧹 Đã tự động dọn dẹp bộ nhớ!")

def perform_memory_cleanup():
    # Lưu trữ giá trị bộ nhớ trước khi dọn dẹp
    memory_before = get_process_memory()
    
    # 1. Force garbage collection
    gc.collect()
    
    # 2. Xóa cache Streamlit
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # 3. Xóa cache của các thư viện phổ biến
    try:
        # Xóa cache của pandas
        pd.core.common._maybe_cache_clear()
        
        # Xóa cache của numpy
        np.clear_cache()
    except:
        pass
    
    # 4. Xóa thư mục cache tạm thời của Streamlit
    try:
        cache_dir = os.path.join(tempfile.gettempdir(), "streamlit_cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
    except:
        pass
    
    # 5. Kiểm tra hiệu quả của việc dọn dẹp
    memory_after = get_process_memory()
    memory_saved = memory_before - memory_after
    
    # Trả về thông tin về hiệu quả dọn dẹp
    return memory_saved

def add_memory_cleanup_button():
    if st.sidebar.button("🧹 Dọn dẹp bộ nhớ"):
        memory_saved = perform_memory_cleanup()
        if memory_saved > 0:
            st.sidebar.success(f"✅ Đã dọn dẹp bộ nhớ! Giải phóng được {memory_saved:.1f} MB")
        else:
            st.sidebar.info("ℹ️ Không có bộ nhớ nào được giải phóng. Hệ thống đã tối ưu.") 