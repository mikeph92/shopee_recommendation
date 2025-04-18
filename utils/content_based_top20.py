import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
import re

# ========== Hàm tiền xử lý ==========
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

def search_and_recommend(model_dict, product_df, keyword, top_k=10):
    vectorizer = model_dict["tfidf_vectorizer"]
    tfidf_matrix = model_dict["tfidf_matrix"]
    
    # Preprocess + Tokenize query như dữ liệu gốc
    query_processed = preprocess_text(keyword)
    query_tokenized = word_tokenize(query_processed, format="text")

    # Vector hóa từ khóa và tính similarity
    query_vector = vectorizer.transform([query_tokenized])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Tạo DataFrame với tất cả các sản phẩm và similarity scores
    result = product_df.copy()
    result["similarity"] = similarity_scores
    result = result[result['similarity'] > 0]

    # Group by product_id and calculate mean rating
    result = result.groupby('product_id').agg({
        'product_name': 'first',
        'sub_category': 'first',
        'price': 'first',
        'rating': 'mean',
        'description': 'first',
        'similarity': 'first',
        'image': 'first',
        'link': 'first'
    }).reset_index()

    # Sort by similarity and get top-k
    result = result.sort_values('similarity', ascending=False)
    return result[["product_id", "product_name", "sub_category", "price", "rating", "description", "similarity", "image", "link"]].head(top_k)


def recommend_by_product_id(model_dict, product_df, product_id, top_k=10):
    # Chuyển đổi kiểu dữ liệu của product_id sang số nguyên
    try:
        product_id = int(product_id)
    except ValueError:
        st.error("❌ Mã sản phẩm không hợp lệ. Vui lòng nhập lại mã sản phẩm dạng số nguyên.")
        return None
    
    cosine_sim = model_dict["cosine_similarity"]

    if product_id not in cosine_sim.keys():
        st.error("❌ Mã sản phẩm không tồn tại trong dữ liệu.")
        return None
    
    # Lấy similarity scores và product IDs từ cosine_sim
    sim_scores = cosine_sim[product_id]
    result_ids = [x[0] for x in sim_scores]
    similarities = [x[1] for x in sim_scores]
    
    # Tạo DataFrame chỉ với các sản phẩm có trong similarity scores
    result = product_df[product_df['product_id'].isin(result_ids)].copy()
    
    # Tạo dictionary ánh xạ product_id với similarity score
    sim_dict = dict(zip(result_ids, similarities))
    
    # Gán similarity scores cho đúng product_id
    result['similarity'] = result['product_id'].map(sim_dict)
    
    # Group by product_id and calculate mean rating
    result = result.groupby('product_id').agg({
        'product_name': 'first',
        'sub_category': 'first',
        'price': 'first',
        'rating': 'mean',
        'description': 'first',
        'similarity': 'first',
        'image': 'first',
        'link': 'first'
    }).reset_index()

    # Sort by similarity and get top-k
    result = result.sort_values('similarity', ascending=False)
    return result[["product_id", "product_name", "sub_category", "price", "rating", "description", "similarity", "image", "link"]].head(top_k)