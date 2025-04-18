import streamlit as st
import psutil
import os
import gc
import numpy as np
import pandas as pd

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    return memory_usage_mb

def display_memory_usage():
    """Display current memory usage in Streamlit"""
    memory_usage = get_memory_usage()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Monitor")
    st.sidebar.markdown(f"Memory Usage: **{memory_usage:.1f} MB**")

def clear_memory():
    """Clear memory by running garbage collection"""
    # Clear all caches
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()
    
    # Run garbage collection
    gc.collect()
    
    # Get memory after cleanup
    memory_after = get_memory_usage()
    return memory_after

def add_memory_cleanup_button():
    """Add a button to clear memory cache"""
    if st.sidebar.button("🧹 Clear Memory Cache"):
        memory_before = get_memory_usage()
        memory_after = clear_memory()
        st.sidebar.success(f"Memory cleaned! ({memory_before:.1f}MB → {memory_after:.1f}MB)") 