import numpy as np
import pandas as pd
import logging


def get_top_n_recommendations(product_df, user_id, model, user_mapping, product_mapping, mu, rated_products, n):
    try:
        # Check if user_id exists in user_mapping
        if user_id not in user_mapping:
            logging.error(f"User ID {user_id} not found in user_mapping")
            raise ValueError(f"User ID {user_id} not found in user_mapping")
            
        user_idx = user_mapping[user_id]

        # Lấy danh sách product_id đã đánh giá
        rated_product_ids = set(rated_products['product_id'].unique())

        # Lấy tất cả product_id hợp lệ
        all_products = product_df['product_id'].unique()
        candidate_products = [
            pid for pid in all_products
            if pid in product_mapping and pid not in rated_product_ids
        ]

        if not candidate_products:
            logging.warning("No candidate products found for recommendation")
            print("⚠️ Không còn sản phẩm nào để gợi ý cho user.")
            return pd.DataFrame()

        # Mapping product_id -> index
        try:
            product_indices = [product_mapping[pid] for pid in candidate_products]
        except KeyError as e:
            logging.error(f"Product ID {e} not found in product_mapping")
            raise ValueError(f"Product ID {e} not found in product_mapping")

        # Tạo input cho model
        user_array = np.array([user_idx] * len(product_indices))
        product_array = np.array(product_indices)

        # Dự đoán rating
        try:
            preds = model.predict([user_array, product_array], verbose=0).flatten() + mu
        except Exception as e:
            logging.error(f"Error during model prediction: {str(e)}")
            raise ValueError(f"Error during model prediction: {str(e)}")

        # Lấy top n sản phẩm
        top_n_idx = preds.argsort()[::-1][:n]
        top_product_ids = [candidate_products[i] for i in top_n_idx]
        top_scores = [preds[i] for i in top_n_idx]

        # Tạo DataFrame kết quả
        result_df = pd.DataFrame({
            'product_id': top_product_ids,
            'predict': top_scores
        })

        # Merge với product_df để lấy thông tin chi tiết
        try:
            merged_df = result_df.merge(product_df, on='product_id', how='left')
        except Exception as e:
            logging.error(f"Error merging result with product_df: {str(e)}")
            raise ValueError(f"Error merging result with product_df: {str(e)}")

        # Trả về các cột mong muốn
        return merged_df[[
            "product_id", "product_name", "sub_category", "price",
            "rating", "description", "image", "link", "predict"
        ]]
    except Exception as e:
        logging.error(f"Error in get_top_n_recommendations: {str(e)}")
        raise