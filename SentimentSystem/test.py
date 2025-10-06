import os
import shutil
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def load_data():
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    path = kagglehub.dataset_download("samantas2020/online-retail-xlsx")
    
    for filename in os.listdir(path):
        shutil.copy(os.path.join(path, filename), data_dir)
    
    df = pd.read_excel(os.path.join(data_dir, "Online_Retail.xlsx"))
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Original shape: {df.shape}")
    print("Missing values:\n", df.isnull().sum())
    
    df = df.dropna(subset=['Description'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[~df['StockCode'].str.contains('POST|BANK|FEE|TEST|AMAZONFEE|DCGS|DOT|PADS|CRUK', case=False, na=False)]
    
    df['Description'] = (
        df['Description']
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r'[^A-Z0-9\s]', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
    )
    
    df = df[df['Description'] != '']
    df = df[~df['Description'].str.contains('UNKNOWN|ADJUST|DISCOUNT|POSTAGE|CARRIAGE|SAMPLES|DAMAGED|LOST|CRUSHED|SMASHED', na=False)]
    
    product_counts = df['StockCode'].value_counts()
    valid_products = product_counts[product_counts >= 5].index
    df = df[df['StockCode'].isin(valid_products)]
    
    df = df.reset_index(drop=True)
    
    print(f"Cleaned shape: {df.shape}")
    print(f"Unique products: {df['StockCode'].nunique()}")
    return df


def extract_product_features(df):
    df['Keywords'] = df['Description'].apply(lambda x: ' '.join(x.split()[:3]))
    
    color_keywords = ['WHITE', 'RED', 'PINK', 'BLUE', 'GREEN', 'BLACK', 'CREAM', 'IVORY']
    material_keywords = ['METAL', 'WOOD', 'GLASS', 'CERAMIC', 'FABRIC', 'PAPER']
    type_keywords = ['HEART', 'HOLDER', 'FRAME', 'BOX', 'BAG', 'LIGHT', 'HANGING']
    
    def extract_attrs(text):
        attrs = []
        for color in color_keywords:
            if color in text:
                attrs.append(color)
        for material in material_keywords:
            if material in text:
                attrs.append(material)
        for typ in type_keywords:
            if typ in text:
                attrs.append(typ)
        return attrs
    
    df['Attributes'] = df['Description'].apply(extract_attrs)
    return df


def prepare_transactions(df: pd.DataFrame):
    df_sorted = df.sort_values(['InvoiceNo', 'InvoiceDate'])
    transactions = df_sorted.groupby('InvoiceNo')['StockCode'].apply(list).tolist()
    transactions = [t for t in transactions if 3 <= len(t) <= 30]
    
    print(f"Training transactions: {len(transactions)}")
    print(f"Sample: {transactions[0][:10]}")
    return transactions


def train_word2vec(transactions):
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=transactions,
        vector_size=300,
        window=15,
        min_count=5,
        sg=1,
        workers=4,
        epochs=100,
        negative=25,
        ns_exponent=0.75,
        alpha=0.05,
        min_alpha=0.00001,
        sample=1e-3,
        hs=0
    )
    print(f"Vocabulary size: {len(model.wv.index_to_key)}")
    return model


def analyze_similarity_distribution(model):
    all_products = model.wv.index_to_key[:500]
    sample_size = min(100, len(all_products))
    sample_products = np.random.choice(all_products, sample_size, replace=False)
    
    similarities = []
    for prod in sample_products:
        if prod in model.wv:
            try:
                similar = model.wv.most_similar(prod, topn=10)
                similarities.extend([sim for _, sim in similar])
            except KeyError:
                continue
    
    if len(similarities) == 0:
        print("No valid similarities found")
        return
    
    print(f"\nSimilarity Distribution Analysis:")
    print(f"Mean: {np.mean(similarities):.4f}")
    print(f"Std: {np.std(similarities):.4f}")
    print(f"Min: {np.min(similarities):.4f}")
    print(f"Max: {np.max(similarities):.4f}")
    print(f"Median: {np.median(similarities):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Product Similarities')
    plt.axvline(np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_embeddings(model, n_words=150):
    words = model.wv.index_to_key[:n_words]
    vectors = model.wv[words]
    
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(vectors)
    
    plt.figure(figsize=(14, 10))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, s=30)
    
    for i in range(min(50, len(words))):
        plt.annotate(words[i], xy=(reduced[i, 0], reduced[i, 1]), fontsize=7, alpha=0.7)
    
    plt.title("Word2Vec Product Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()


def compute_diversity(recommendations, df):
    if len(recommendations) == 0:
        return 0
    
    all_attrs = []
    for _, row in recommendations.iterrows():
        prod_data = df[df['StockCode'] == row['StockCode']]
        if len(prod_data) > 0 and 'Attributes' in prod_data.columns:
            attrs = prod_data.iloc[0]['Attributes']
            all_attrs.extend(attrs)
    
    if len(all_attrs) == 0:
        return 0
    
    unique_attrs = len(set(all_attrs))
    total_attrs = len(all_attrs)
    diversity = unique_attrs / total_attrs if total_attrs > 0 else 0
    return diversity


def recommend_similar(model, product_code, df, top_n=10, similarity_threshold=0.5):
    if product_code not in model.wv:
        return f"Product {product_code} not in vocabulary"
    
    similar = model.wv.most_similar(product_code, topn=top_n * 2)
    similar = [(code, score) for code, score in similar if score >= similarity_threshold][:top_n]
    
    results = []
    for code, score in similar:
        desc_match = df[df['StockCode'] == code]['Description']
        desc = desc_match.iloc[0] if len(desc_match) > 0 else 'N/A'
        
        if 'Attributes' in df.columns:
            attrs_match = df[df['StockCode'] == code]['Attributes']
            attrs = attrs_match.iloc[0] if len(attrs_match) > 0 else []
            attrs_str = ', '.join(attrs) if attrs else 'N/A'
        else:
            attrs_str = 'N/A'
        
        results.append({
            'Rank': len(results) + 1,
            'StockCode': code,
            'Description': desc,
            'Attributes': attrs_str,
            'Similarity': f"{score:.4f}"
        })
    
    result_df = pd.DataFrame(results)
    diversity = compute_diversity(result_df, df)
    print(f"Recommendation Diversity: {diversity:.4f}")
    
    return result_df


def recommend_from_basket(model, product_codes, df, top_n=10, similarity_threshold=0.5):
    product_codes = [str(p) for p in product_codes]
    valid = [p for p in product_codes if p in model.wv]
    
    if not valid:
        return "No valid products in vocabulary"
    
    print(f"Valid products: {len(valid)}/{len(product_codes)}")
    
    basket_attrs = []
    for p in valid:
        prod_data = df[df['StockCode'] == p]
        if len(prod_data) > 0 and 'Attributes' in prod_data.columns:
            basket_attrs.extend(prod_data.iloc[0]['Attributes'])
    
    if basket_attrs:
        print(f"Basket attributes: {Counter(basket_attrs).most_common()}")
    
    avg_vec = np.mean([model.wv[p] for p in valid], axis=0).reshape(1, -1)
    
    all_products = model.wv.index_to_key
    all_vecs = np.array([model.wv[w] for w in all_products])
    
    sims = cosine_similarity(avg_vec, all_vecs)[0]
    
    for p in valid:
        if p in all_products:
            idx = all_products.index(p)
            sims[idx] = -1
    
    candidate_idx = np.where(sims >= similarity_threshold)[0]
    if len(candidate_idx) == 0:
        print("Warning: No products above similarity threshold, lowering threshold")
        similarity_threshold = 0.3
        candidate_idx = np.where(sims >= similarity_threshold)[0]
    
    candidate_sims = sims[candidate_idx]
    top_candidate_idx = candidate_idx[np.argsort(candidate_sims)[-top_n:][::-1]]
    
    results = []
    for idx in top_candidate_idx:
        code = all_products[idx]
        desc_match = df[df['StockCode'] == code]['Description']
        desc = desc_match.iloc[0] if len(desc_match) > 0 else 'N/A'
        
        if 'Attributes' in df.columns:
            attrs_match = df[df['StockCode'] == code]['Attributes']
            attrs = attrs_match.iloc[0] if len(attrs_match) > 0 else []
            attrs_str = ', '.join(attrs) if attrs else 'N/A'
        else:
            attrs_str = 'N/A'
        
        results.append({
            'Rank': len(results) + 1,
            'StockCode': code,
            'Description': desc,
            'Attributes': attrs_str,
            'Similarity': f"{sims[idx]:.4f}"
        })
    
    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        diversity = compute_diversity(result_df, df)
        print(f"Recommendation Diversity: {diversity:.4f}")
    
    return result_df


def evaluate_model(model, test_df, top_n=10):
    test_transactions = test_df.groupby('InvoiceNo')['StockCode'].apply(list).tolist()
    test_transactions = [t for t in test_transactions if len(t) >= 3]
    
    hits = 0
    total = 0
    precisions = []
    mrr_scores = []
    
    for transaction in test_transactions:
        if len(transaction) < 3:
            continue
        
        split_point = len(transaction) - 1
        basket = transaction[:split_point]
        target = transaction[split_point]
        
        valid = [p for p in basket if p in model.wv]
        
        if not valid or target not in model.wv:
            continue
        
        avg_vec = np.mean([model.wv[p] for p in valid], axis=0).reshape(1, -1)
        all_products = model.wv.index_to_key
        all_vecs = np.array([model.wv[w] for w in all_products])
        
        sims = cosine_similarity(avg_vec, all_vecs)[0]
        
        for p in valid:
            if p in all_products:
                sims[all_products.index(p)] = -1
        
        top_indices = np.argsort(sims)[-top_n:][::-1]
        recommended = [all_products[i] for i in top_indices]
        
        if target in recommended:
            hits += 1
            rank = recommended.index(target) + 1
            mrr_scores.append(1.0 / rank)
            precisions.append(1.0 / top_n)
        else:
            precisions.append(0.0)
        
        total += 1
    
    hit_rate = hits / total if total > 0 else 0
    avg_precision = np.mean(precisions) if precisions else 0
    mrr = np.mean(mrr_scores) if mrr_scores else 0
    
    print(f"\nEvaluation Results:")
    print(f"Hit Rate@{top_n}: {hit_rate:.4f} ({hits}/{total})")
    print(f"Average Precision@{top_n}: {avg_precision:.4f}")
    print(f"MRR: {mrr:.4f}")
    
    return hit_rate, avg_precision


if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    df = extract_product_features(df)
    
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")
    
    transactions = prepare_transactions(train_df)
    model = train_word2vec(transactions)
    
    analyze_similarity_distribution(model)
    visualize_embeddings(model)
    
    test_product = '85123A'
    if test_product in model.wv:
        original_desc = train_df[train_df['StockCode'] == test_product]['Description'].iloc[0]
        
        print(f"\n{'='*70}")
        print(f"Single Product Recommendation")
        print(f"{'='*70}")
        print(f"Product: {test_product} - {original_desc}\n")
        print(recommend_similar(model, test_product, train_df, top_n=10, similarity_threshold=0.4))
    
    sample_basket = ['84406B', '20679', '85123A']
    sample_basket = [str(x) for x in sample_basket]
    print(f"\n{'='*70}")
    print(f"Basket-Based Recommendation")
    print(f"{'='*70}")
    print(f"Basket: {sample_basket}\n")
    
    for item in sample_basket:
        if item in model.wv:
            item_match = train_df[train_df['StockCode'] == item]
            if len(item_match) > 0:
                desc = item_match['Description'].iloc[0]
                print(f"  {item}: {desc}")
    
    print(f"\nRecommended products:\n")
    print(recommend_from_basket(model, sample_basket, train_df, top_n=10, similarity_threshold=0.4))
    
    evaluate_model(model, test_df, top_n=10)
    # evaluate_model(model, test_df, top_n=20)
    # evaluate_model(model, test_df, top_n=50)