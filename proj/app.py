# Import necessary libraries
import logging
import matplotlib.pyplot as plt
import seaborn as sbs

# Options for plots (global configuration)
plt.rcParams['figure.figsize'] = (10, 6)
sbs.set('paper')

# Set logging level to CRITICAL (global configuration)
logging.getLogger().setLevel(logging.CRITICAL)

from flask import Flask, render_template, request, redirect, url_for
import utils

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    """Render the main input page."""
    return render_template('index.html')

@app.route('/stats')
def stats():
    # # Load the docs_remaining and docs_crossref using the load_docs_from_pickle function
    # docs_remaining = utils.load_docs_from_pickle('data/docs_remaining.pkl')
    # docs_crossref = utils.load_docs_from_pickle('data/docs_crossref.pkl')

    # # Generate histograms and save them in /static/graphs
    # utils.plot_histograms(docs_remaining)

    # # Generate the co-citation and coupling networks and save them in /static/graphs
    # utils.plot_cocitation_network(docs_crossref)
    # utils.plot_coupling_network(docs_crossref)

    # # Generate the stats
    # utils.generate_stats(docs_remaining)

    # Prepare the template data for the generated graphs and tables
    histograms = [
        "/static/graphs/year_histogram.png",
        "/static/graphs/affiliation_histogram.png",
        "/static/graphs/author_histogram.png",
        "/static/graphs/number_authors_histogram.png"
    ]
    
    networks = [
        {"name": "Co-citation Network", "path": "/static/graphs/cocitation_network.html"},
        {"name": "Coupling Network", "path": "/static/graphs/coupling_network.html"}
    ]

    word_dist_img = "/static/graphs/word_distribution.png"
    word_dist_table_img = "/static/graphs/word_distribution_table.png"
    topic_table_img = "/static/graphs/topic_modeling_table.png"
    topic_cloud_img = "/static/graphs/topic_cloud.png"

    # Render the stats page with the generated graphs and tables
    return render_template('stats.html', histograms=histograms, networks=networks, word_dist_img=word_dist_img, 
                           word_dist_table_img=word_dist_table_img, topic_table_img=topic_table_img, topic_cloud_img=topic_cloud_img)

# @app.route('/add_papers', methods=['POST'])
# def add_papers():
#     """Handle adding new papers."""
#     uploaded_files = request.files.getlist('file')  # Get all uploaded files

#     # Process and merge the documents using the function from utils.py
#     docs_remaining = load_and_process_files(uploaded_files)
#     print(len(docs_remaining), "papers remaining")

#     return redirect(url_for('stats'))

@app.route('/search', methods=['GET'])
def search():
    """Handle search functionality for documents based on a keyword."""
    
    # Get the query string from the GET parameters
    query = request.args.get('query', '').strip()

    print(f"Search query: {query}")

    if not query:
        # Redirect back to the index page if the query is empty
        return redirect(url_for('index'))

    # Filter documents based on the query
    filtered_docs = utils.filter_by_title(query)

    # Pass the filtered documents and query back to the template
    return render_template(
        'search.html',
        query=query,
        results=filtered_docs[:15],  # Limit to top 15 results
        total_results=len(filtered_docs),
    )


if __name__ == '__main__':
    app.run(debug=True)
