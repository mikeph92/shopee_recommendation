import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objs as go
import numpy as np
import gc

@st.cache_data(ttl=3600)
def calculate_basic_stats(products_df, ratings_df):
    stats = {
        'num_products': len(products_df),
        'num_categories': products_df['sub_category'].nunique(),
        'num_ratings': len(ratings_df),
        'num_users': ratings_df['user_id'].nunique()
    }
    return stats

@st.cache_data(ttl=3600)
def calculate_user_stats(ratings_df, products_df):
    try:
        # Merge with minimal columns
        eda_merged = ratings_df.merge(
            products_df[['product_id', 'price']],
            on='product_id',
            how='left'
        )
        
        stats = {
            'top_reviewer': ratings_df['user_id'].value_counts().idxmax(),
            'top_reviewer_count': ratings_df['user_id'].value_counts().max(),
            'top_spender': eda_merged.groupby('user_id')['price'].sum().idxmax(),
            'top_spend_amount': eda_merged.groupby('user_id')['price'].sum().max(),
            'top_five_star_user': ratings_df[ratings_df['rating'] == 5]['user_id'].value_counts().idxmax(),
            'top_five_star_count': ratings_df[ratings_df['rating'] == 5]['user_id'].value_counts().max()
        }
        
        del eda_merged
        gc.collect()
        
        return stats
    except Exception as e:
        st.error(f"Error calculating user statistics: {str(e)}")
        return None

def create_figure(func):
    """Decorator to handle figure creation and cleanup"""
    def wrapper(*args, **kwargs):
        try:
            fig = func(*args, **kwargs)
            st.pyplot(fig)
            plt.close(fig)
            gc.collect()
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
    return wrapper

@create_figure
def plot_price_distribution(products_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=products_df, x='price', bins=50, ax=ax)
    ax.set_title('Phân bố giá sản phẩm')
    ax.set_xlabel('Giá (VNĐ)')
    ax.set_ylabel('Số lượng')
    return fig

@create_figure
def plot_category_distribution(products_df):
    category_counts = products_df['sub_category'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    category_counts.plot(kind='bar', ax=ax)
    ax.set_title('Top 10 danh mục sản phẩm')
    ax.set_xlabel('Danh mục')
    ax.set_ylabel('Số lượng sản phẩm')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def data_insight(products_df, ratings_df):
    if products_df is None or ratings_df is None:
        st.error("Không thể tải dữ liệu. Vui lòng thử lại sau.")
        return

    try:
        st.title("📊 Khám phá dữ liệu")
        
        # Basic Statistics
        stats = {
            'num_products': len(products_df),
            'num_categories': products_df['sub_category'].nunique(),
            'num_ratings': len(ratings_df),
            'num_users': ratings_df['user_id'].nunique()
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tổng số sản phẩm", f"{stats['num_products']:,}")
            st.metric("Số danh mục con", f"{stats['num_categories']:,}")
        with col2:
            st.metric("Tổng số đánh giá", f"{stats['num_ratings']:,}")
            st.metric("Số người dùng", f"{stats['num_users']:,}")

        # Sample data display
        st.markdown("### 🛍️ Dữ liệu sản phẩm mẫu")
        st.dataframe(products_df.head(10))
        st.markdown("### ⭐ Dữ liệu đánh giá mẫu")
        st.dataframe(ratings_df.head(10))

        # Basic visualizations with error handling
        try:
            st.header("📊 Phân bố giá sản phẩm")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=products_df, x='price', bins=50, ax=ax)
            ax.set_title('Phân bố giá sản phẩm')
            ax.set_xlabel('Giá (VNĐ)')
            ax.set_ylabel('Số lượng')
            st.pyplot(fig)
            plt.close(fig)
            gc.collect()
        except Exception as e:
            st.warning("Không thể hiển thị biểu đồ phân bố giá.")

        try:
            st.header("📊 Top danh mục sản phẩm")
            category_counts = products_df['sub_category'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(12, 6))
            category_counts.plot(kind='bar', ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            gc.collect()
        except Exception as e:
            st.warning("Không thể hiển thị biểu đồ danh mục sản phẩm.")

        try:
            st.header("📊 Phân bố đánh giá")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=ratings_df, x='rating', bins=20, ax=ax)
            ax.set_title('Phân bố điểm đánh giá')
            ax.set_xlabel('Điểm đánh giá')
            ax.set_ylabel('Số lượng')
            st.pyplot(fig)
            plt.close(fig)
            gc.collect()
        except Exception as e:
            st.warning("Không thể hiển thị biểu đồ phân bố đánh giá.")

    except Exception as e:
        st.error("Có lỗi xảy ra khi phân tích dữ liệu.")
        st.exception(e)
    finally:
        gc.collect()

@create_figure
def plot_description_distribution(products_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=products_df, x='desc_len', bins=50, ax=ax)
    ax.set_title('Phân bố độ dài mô tả sản phẩm')
    ax.set_xlabel('Độ dài mô tả (ký tự)')
    ax.set_ylabel('Số lượng')
    return fig

@create_figure
def plot_rating_distribution(ratings_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=ratings_df, x='rating', bins=20, ax=ax)
    ax.set_title('Phân bố điểm đánh giá')
    ax.set_xlabel('Điểm đánh giá')
    ax.set_ylabel('Số lượng')
    return fig

def display_user_stats(stats):
    st.header("👥 Thống kê người dùng")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Người dùng đánh giá nhiều nhất", 
                 f"User {stats['top_reviewer']}", 
                 f"{stats['top_reviewer_count']} đánh giá")
        st.metric("Người dùng chi tiêu nhiều nhất",
                 f"User {stats['top_spender']}", 
                 f"{stats['top_spend_amount']:,.0f} VNĐ")
                 
    with col2:
        st.metric("Người dùng đánh giá 5 sao nhiều nhất",
                 f"User {stats['top_five_star_user']}", 
                 f"{stats['top_five_star_count']} đánh giá")

    st.image("images/insight.jpeg", width=1000)
    st.title("Một số thông tin về dữ liệu")

    st.markdown("### 🛍️ Dữ liệu sản phẩm")
    st.dataframe(products_df.head(10))
    st.markdown("### ⭐ Dữ liệu đánh giá sản phẩm")
    st.dataframe(ratings_df.head(10))

    
    # Tính các chỉ số
    num_products = products_df['product_id'].nunique()
    num_users = ratings_df['user_id'].nunique()
    num_ratings = ratings_df.shape[0]
    
    # Merge để lấy thông tin price, product_name
    eda_merged = ratings_df.merge(products_df[['product_id', 'price', 'product_name']], on='product_id', how='left')

    # User đánh giá nhiều nhất
    top_reviewer = ratings_df['user_id'].value_counts().idxmax()
    top_reviewer_count = ratings_df['user_id'].value_counts().max()

    # User chi tiêu nhiều nhất
    user_spend = eda_merged.groupby('user_id')['price'].sum()
    top_spender = user_spend.idxmax()
    top_spend_amount = user_spend.max()

    # User rating 5 sao nhiều nhất
    one_star_users = ratings_df[ratings_df['rating'] == 5]['user_id'].value_counts()
    top_one_star_user = one_star_users.idxmax()
    top_one_star_count = one_star_users.max()

    # Sản phẩm bán chạy nhất (nhiều đánh giá nhất)
    top_product_id = ratings_df['product_id'].value_counts().idxmax()
    top_product_name = products_df.loc[products_df['product_id'] == top_product_id, 'product_name'].values[0]

    # Sản phẩm bán ít nhất (ít đánh giá nhất)
    least_product_id = ratings_df['product_id'].value_counts().idxmin()
    least_product_name = products_df.loc[products_df['product_id'] == least_product_id, 'product_name'].values[0]

    col1, col2, col3 = st.columns(3)

    # ---------- CỘT 1: THỐNG KÊ CHUNG ----------
    with col1:
        st.markdown("### 📊 Thống kê tổng quan")
        st.markdown(f"🛍️ **Số sản phẩm:**<br><span style='font-size:16px'>{num_products}</span>", unsafe_allow_html=True)
        st.markdown(f"👤 **Số người dùng:**<br><span style='font-size:16px'>{num_users}</span>", unsafe_allow_html=True)
        st.markdown(f"⭐ **Số lượt đánh giá:**<br><span style='font-size:16px'>{num_ratings}</span>", unsafe_allow_html=True)

    # ---------- CỘT 2: USER ----------
    with col2:
        st.markdown("### 🙋‍♂️ Người dùng nổi bật")
        st.markdown(f"✍️ **Review nhiều nhất:**<br><span style='font-size:15px'>{top_reviewer} ({top_reviewer_count} lần)</span>", unsafe_allow_html=True)
        st.markdown(f"💰 **Chi tiêu nhiều nhất:**<br><span style='font-size:15px'>{top_spender} ({top_spend_amount:,.0f} VNĐ)</span>", unsafe_allow_html=True)
        st.markdown(f"😍 **Rating 5⭐ nhiều nhất:**<br><span style='font-size:15px'>{top_one_star_user} ({top_one_star_count} lần)</span>", unsafe_allow_html=True)

    # ---------- CỘT 3: SẢN PHẨM ----------
    with col3:
        st.markdown("### 📦 Sản phẩm nổi bật")
        st.markdown(f"🔥 **Nhiều đánh giá nhất:**<br><span style='font-size:15px'>{top_product_name[:30]}...</span>", unsafe_allow_html=True)
        st.markdown(f"🥶 **Ít đánh giá nhất:**<br><span style='font-size:15px'>{least_product_name[:30]}...</span>", unsafe_allow_html=True)


    # Tính top nhóm hàng phổ biến
    top_subcat = products_df['sub_category'].value_counts().head(10)
    # Tiêu đề
    st.subheader("📦 Top 10 Nhóm Hàng Phổ Biến Nhất")

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=top_subcat.values,
        y=top_subcat.index,
        palette="Blues_d",
        ax=ax
    )
    ax.set_title("Top 10 nhóm hàng phổ biến")
    ax.set_xlabel("Số lượng sản phẩm")
    ax.set_ylabel("Nhóm hàng")
    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)
    
    # Tính top 20 tên sản phẩm lặp lại nhiều nhất
    top_names = products_df['product_name'].value_counts().head(20)

    # Hiển thị tiêu đề
    st.subheader("🛍️ Top 20 Tên Sản Phẩm Phổ Biến Nhất")

    # Vẽ biểu đồ với màu
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=top_names.values,
        y=top_names.index,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Top 20 tên sản phẩm phổ biến nhất")
    ax.set_xlabel("Số lần xuất hiện")
    ax.set_ylabel("Tên sản phẩm")
    plt.tight_layout()

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)
    
    st.subheader("🎻 Phân bố giá theo nhóm sản phẩm")

    # Vẽ biểu đồ violin
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=products_df,
        x='sub_category',
        y='price',
        inner='quartile',
        palette='pastel',
        ax=ax
    )

    # Tùy chỉnh trục
    ax.set_title("Phân bố giá theo nhóm sản phẩm (Violin Plot)")
    ax.set_xlabel("Nhóm sản phẩm")
    ax.set_ylabel("Giá (VNĐ)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)
    
    st.subheader("📦 Phân bố Rating theo Nhóm Sản Phẩm")

    # Vẽ biểu đồ boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=products_df,
        x='sub_category',
        y='rating',
        palette='Set3',
        ax=ax
    )

    # Tùy chỉnh trục
    ax.set_title("Phân bố rating theo từng nhóm sản phẩm")
    ax.set_xlabel("Nhóm sản phẩm")
    ax.set_ylabel("Rating")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)
    
    st.subheader("📈 Tương Quan giữa Giá và Rating theo Nhóm Sản Phẩm")

    # Vẽ biểu đồ scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=products_df,
        x='rating',
        y='price',
        hue='sub_category',
        alpha=0.7,
        palette='tab10',
        ax=ax
    )

    # Tùy chỉnh biểu đồ
    ax.set_title("Tương quan giữa giá và rating theo nhóm sản phẩm")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Giá sản phẩm (VNĐ)")
    ax.legend(title='Nhóm sản phẩm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)
    
    # Tính giá trung bình theo nhóm sản phẩm
    avg_price = products_df.groupby('sub_category')['price'].mean().sort_values()

    # Tiêu đề
    st.subheader("💰 Giá Trung Bình Theo Nhóm Sản Phẩm")

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=avg_price.values,
        y=avg_price.index,
        palette='viridis',
        ax=ax
    )
    ax.set_title("Giá trung bình theo nhóm sản phẩm")
    ax.set_xlabel("Giá trung bình (VNĐ)")
    ax.set_ylabel("Nhóm sản phẩm")
    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)
    
    # Tính rating trung bình theo nhóm sản phẩm
    avg_rating = products_df.groupby('sub_category')['rating'].mean().sort_values()

    # Tiêu đề
    st.subheader("⭐ Rating Trung Bình Theo Nhóm Sản Phẩm")

    # Vẽ biểu đồ barplot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=avg_rating.values,
        y=avg_rating.index,
        palette='magma',
        ax=ax
    )

    # Tùy chỉnh tiêu đề và nhãn
    ax.set_title("Rating trung bình theo nhóm sản phẩm")
    ax.set_xlabel("Rating trung bình")
    ax.set_ylabel("Nhóm sản phẩm")
    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)

        
    # Tính toán tỷ lệ đánh giá
    values = ratings_df.rating.value_counts()
    labels = values.index
    colors = ['red', 'blue', 'green', 'yellow', 'black']

    # Tạo pie chart
    trace = go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors)
    )

    # Tạo layout cho pie chart
    layout = go.Layout(
        title='Biểu đồ Ratings theo phần trăm'
    )

    # Tạo figure và vẽ biểu đồ
    fig = go.Figure(data=trace, layout=layout)

    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig)
    
    # Lọc các sản phẩm có rating = 5
    five_star_ratings = ratings_df[ratings_df['rating'] == 5]
    five_star_ratings = five_star_ratings.merge(products_df[['product_id', 'product_name', 'sub_category']], on='product_id', how='left')

    # Nhóm sản phẩm và đếm số lần đánh giá 5 sao
    top_product = five_star_ratings['product_name'].value_counts().reset_index()
    top_product.columns = ['product_name', 'count']

    # Sắp xếp sản phẩm theo số lượng rating 5 sao giảm dần
    top_product = top_product.sort_values(by='count', ascending=False)

    # Chọn top N sản phẩm có số lượng rating 5 sao nhiều nhất (ví dụ top 10)
    top_N_product = top_product.head(10)

    # Tiêu đề cho phần trong Streamlit
    st.subheader("⭐ Top Sản Phẩm Nhận Được Nhiều Đánh Giá 5 Sao")

    # Vẽ biểu đồ barplot
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(
        data=top_N_product,
        x='count',
        y='product_name',
        palette='viridis',
        ax=ax
    )

    # Tùy chỉnh biểu đồ
    ax.set_xlabel('Count of 5-Star Ratings')
    ax.set_ylabel('Product Name')
    ax.set_title('Product with the Most 5-Star Ratings')

    # Hiển thị biểu đồ trong Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    
    # Nhóm theo sub_category và đếm số lần đánh giá 5 sao
    top_sub_category = five_star_ratings['sub_category'].value_counts().reset_index()
    top_sub_category.columns = ['sub_category', 'count']

    # Sắp xếp các nhóm sản phẩm theo số lượng rating 5 sao giảm dần
    top_sub_category = top_sub_category.sort_values(by='count', ascending=False)

    # Chọn top N nhóm sản phẩm có số lượng rating 5 sao nhiều nhất (ví dụ top 10)
    top_N_sub_category = top_sub_category.head(10)

    # Tiêu đề cho phần trong Streamlit
    st.subheader("⭐ Top Nhóm Sản Phẩm Nhận Được Nhiều Đánh Giá 5 Sao")

    # Vẽ biểu đồ barplot
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(
        data=top_N_sub_category,
        x='count',
        y='sub_category',
        palette='viridis',
        ax=ax
    )

    # Tùy chỉnh biểu đồ
    ax.set_xlabel('Count of 5-Star Ratings')
    ax.set_ylabel('Sub Category')
    ax.set_title('Sub Category with the Most 5-Star Ratings')

    # Hiển thị biểu đồ trong Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    
    # Lọc các sản phẩm có rating = 1
    one_star_ratings = ratings_df[ratings_df['rating'] == 1]

    # Kết hợp dữ liệu từ eda_df và products_df để có được product_name
    one_star_ratings = one_star_ratings.merge(products_df[['product_id', 'product_name']], on='product_id', how='left')

    # Nhóm sản phẩm và đếm số lần đánh giá 1 sao
    top_product = one_star_ratings['product_name'].value_counts().reset_index()
    top_product.columns = ['product_name', 'count']

    # Sắp xếp sản phẩm theo số lượng rating 1 sao giảm dần
    top_product = top_product.sort_values(by='count', ascending=False)

    # Chọn top N sản phẩm có số lượng rating 1 sao nhiều nhất (ví dụ top 10)
    top_N_product = top_product.head(10)

    # Tiêu đề cho phần trong Streamlit
    st.subheader("⚠️ Top Sản Phẩm Nhận Được Nhiều Đánh Giá 1 Sao")

    # Vẽ biểu đồ barplot
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(
        data=top_N_product,
        x='count',
        y='product_name',
        palette='viridis',
        ax=ax
    )

    # Tùy chỉnh biểu đồ
    ax.set_xlabel('Count of 1-Star Ratings')
    ax.set_ylabel('Product Name')
    ax.set_title('Product with the Most 1-Star Ratings')

    # Hiển thị biểu đồ trong Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("📊 Phân bố độ dài mô tả sản phẩm")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(products_df['desc_len'], bins=50, kde=True, ax=ax)
    ax.set_title("Phân bố độ dài mô tả sản phẩm")
    ax.set_xlabel("Số từ")
    ax.set_ylabel("Số sản phẩm")
    plt.tight_layout()

    st.pyplot(fig)
    
    st.markdown("### ☁️ Wordcloud mô tả sản phẩm")
    st.image("images/wordcloud.png", width=1000)