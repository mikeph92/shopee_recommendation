import streamlit as st
import psutil
import os
import gc
import numpy as np
import pandas as pd

MEMORY_THRESHOLD = 85  # Percentage

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_percent()

def display_memory_usage():
    # Get memory usage
    memory_usage = get_process_memory()
    
    # Create a progress bar for memory usage
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.progress(min(memory_usage/100, 1.0))
    with col2:
        st.write(f"{memory_usage:.1f}%")
        
    # Automatic cleanup if memory usage is too high
    if memory_usage > MEMORY_THRESHOLD:
        perform_memory_cleanup()
        st.sidebar.warning("🧹 Đã tự động dọn dẹp bộ nhớ!")

def perform_memory_cleanup():
    # Force garbage collection
    gc.collect()
    
    # Clear Streamlit cache if memory is still high
    if get_process_memory() > MEMORY_THRESHOLD:
        st.cache_data.clear()
        st.cache_resource.clear()

def add_memory_cleanup_button():
    if st.sidebar.button("🧹 Dọn dẹp bộ nhớ"):
        perform_memory_cleanup()
        st.sidebar.success("✅ Đã dọn dẹp bộ nhớ!") 