from flask import Flask, request, jsonify, render_template, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from synonyms import expand_query

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key

# Load dataset
df = pd.read_csv('Netflix_movies_and_tv_shows_clustering.csv')
df = df[['type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']]
df.dropna(inplace=True)

# Create TF-IDF Vectorizer with adjusted settings
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['title'])

def search(query, page=1, per_page=10):
    expanded_query = expand_query(query)
    query_vec = vectorizer.transform([expanded_query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = similarity.argsort()[::-1]
    results = df.iloc[indices]
    results['similarity'] = similarity[indices]

    # Filter out results with similarity score of 0
    results = results[results['similarity'] > 0]

    total_results = len(results)
    total_pages = (total_results + per_page - 1) // per_page  # Calculate the total number of pages

    # Paginate results
    start = (page - 1) * per_page
    end = min(start + per_page, total_results)
    paginated_results = results[start:end]
    
    return paginated_results, expanded_query, total_results, total_pages, start + 1, end

@app.route('/')
def home():
    query = session.get('query', '')
    results = session.get('results', [])
    return render_template('index.html', query=query, results=results)

@app.route('/search', methods=['GET'])
def search_api():
    query = request.args.get('query')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))

    results, expanded_query, total_results, total_pages, start, end = search(query, page, per_page)
    session['query'] = query
    session['results'] = results.to_dict(orient='records')

    return jsonify({
        'results': results.to_dict(orient='records'),
        'expanded_query': expanded_query,
        'total_results': total_results,
        'total_pages': total_pages,
        'current_page': page,
        'start': start,
        'end': end
    })

@app.route('/detail', methods=['GET'])
def detail():
    title = request.args.get('title')
    item = df[df['title'] == title].iloc[0]
    return render_template('detail.html', item=item)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
