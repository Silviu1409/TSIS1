import logging
import matplotlib.pyplot as plt
import seaborn as sbs
import os
from threading import Thread, Lock, Event
from werkzeug.utils import secure_filename
from datetime import datetime

# Options for plots (global configuration)
plt.rcParams['figure.figsize'] = (10, 6)
sbs.set_theme('paper')

from flask import Flask, render_template, request, redirect, url_for, jsonify
import utils

app = Flask(__name__, static_folder='static')
app.logger.setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Shared state to track task completion
task_status = {}
upload_in_progress = False
stats_in_progress = False
process_lock = Lock()
utils.process_cancel_event = Event()

# Utility function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'ris'}


@app.route('/')
def index():
    """Render the main input page."""
    return render_template('index.html')


@app.route('/stats')
def stats():
    folder = '/static/graphs/'

    # Prepare the template data for the generated graphs and tables
    histograms = [
        {"name": "Year Histogram", "path": folder + "year_histogram.png"},
        {"name": "Affiliation Histogram", "path": folder + "affiliation_histogram.png"},
        {"name": "Author Histogram", "path": folder + "author_histogram.png"},
        {"name": "Number of authors Histogram", "path": folder + "number_authors_histogram.png"},
        {"name": "Publication source Histogram", "path": folder + "publication_source_histogram.png"}
    ]
    
    networks = [
        {"name": "Co-citation Network", "path": folder + "cocitation_network.html"},
        {"name": "Coupling Network", "path": folder + "coupling_network.html"}
    ]

    word_dists = [
        {"name": "Word Distribution Graph", "path": folder + "word_distribution.png"},
        {"name": "Word Distribution Table", "path": folder + "word_distribution_table.png"}
    ]

    topics = [
        {"name": "Topic Modeling Table", "path": folder + "topic_modeling_table.png"},
        {"name": "Topic Cloud", "path": folder + "topic_cloud.png"}
    ]

    # Render the stats page with the generated graphs and tables
    return render_template('stats.html', histograms=histograms, networks=networks, word_dists=word_dists, topics=topics)


@app.route('/add_papers', methods=['POST'])
def add_papers():
    """Handle adding new papers."""
    global upload_in_progress, stats_in_progress, process_lock

    if 'file' not in request.files:
        return "No file part", 400
    
    # Cancel ongoing stats processing (if any)
    utils.process_cancel_event.set()

    with process_lock:
        if upload_in_progress:
            return "File upload already in progress. Please wait.", 409
        if stats_in_progress:
            # Allow new uploads but cancel ongoing stats processing
            print("Stats generation in progress; canceling to allow new upload.")

        # Mark upload as in progress
        upload_in_progress = True

    uploaded_files = request.files.getlist('file')
    if not uploaded_files or all(file.filename == '' for file in uploaded_files):
        with process_lock:
            upload_in_progress = False
        return "No files selected", 400
    
    saved_files = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join("data/uploads/", filename)

            if os.path.exists(file_path):
                base, ext = os.path.splitext(filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{base}_{timestamp}{ext}"
                file_path = os.path.join("data/uploads/", filename)

            file.save(file_path)
            saved_files.append(file_path)
        else:
            with process_lock:
                upload_in_progress = False
            return "Invalid file type. Please upload a CSV file.", 400

    # Start a new background process for file processing
    def process_files():
        global upload_in_progress, stats_in_progress

        try:
            utils.load_and_process_files(saved_files)
        finally:
            with process_lock:
                upload_in_progress = False

    thread = Thread(target=process_files)
    thread.daemon = True
    thread.start()

    return redirect(url_for('stats'))


@app.route('/search', methods=['GET'])
def search():
    """Handle search functionality for documents based on a keyword."""

    global task_status
    
    # Get the query string from the GET parameters
    query = request.args.get('query', '').strip()

    if not query:
        # Redirect back to the index page if the query is empty
        return redirect(url_for('index'))
    
    # Prepare initial response for the user
    folder = '/static/graphs/search/'
    task_status[query] = "in-progress"
    
    # Filter documents based on the query
    filtered_docs, filtered_docs_crossref = utils.filter_by_keyword(query)

    results = [{
            'title': doc.title,
            'authors': doc.entry['Authors'].replace(";", ","),
            'publication_source': doc.entry['Publication Title'],
            'publisher': doc.publisher,
            'publication_year': doc.publication_year,
            'abstract': doc.abstract,
            'doi': doc.id.doi,
            'link': doc.entry.get('PDF Link', '')
        } for doc in filtered_docs]

    def generate_custom_stats(query):
        # Delete previous stats
        for file_name in os.listdir('./static/graphs/search'):
            file_path = os.path.join('./static/graphs/search', file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Generate histograms and save them in /static/graphs/search
        utils.plot_histograms(filtered_docs, filtered_docs_crossref, "static/graphs/search/")

        # Generate the co-citation and coupling networks and save them in /static/graphs
        utils.plot_cocitation_network(filtered_docs_crossref, "static/graphs/search/")
        utils.plot_coupling_network(filtered_docs_crossref, "static/graphs/search/")

        # Generate the stats
        utils.generate_stats(filtered_docs, "static/graphs/search/")

        # Prepare template data for the generated graphs and tables
        histograms = [
            {"name": "Year Histogram", "path": folder + "year_histogram.png"},
            {"name": "Affiliation Histogram", "path": folder + "affiliation_histogram.png"},
            {"name": "Author Histogram", "path": folder + "author_histogram.png"},
            {"name": "Number of authors Histogram", "path": folder + "number_authors_histogram.png"},
            {"name": "Publication source Histogram", "path": folder + "publication_source_histogram.png"}
        ]
        
        networks = [
            {"name": "Co-citation Network", "path": folder + "cocitation_network.html"},
            {"name": "Coupling Network", "path": folder + "coupling_network.html"}
        ]

        word_dists = [
            {"name": "Word Distribution Graph", "path": folder + "word_distribution.png"},
            {"name": "Word Distribution Table", "path": folder + "word_distribution_table.png"}
        ]

        topics = [
            {"name": "Topic Modeling Table", "path": folder + "topic_modeling_table.png"},
            {"name": "Topic Cloud", "path": folder + "topic_cloud.png"}
        ]

        # Save paths in the task_status dictionary
        task_status[query] = {
            "status": "complete",
            "paths": {
                "histograms": histograms,
                "networks": networks,
                "word_dists": word_dists,
                "topics": topics
            }
        }

    # Start the background thread for heavy computation
    thread = Thread(target=generate_custom_stats, args=(query,))
    thread.daemon = True
    thread.start()
    
    return render_template(
        'search.html',
        query=query,
        results=results,
        total_results=len(filtered_docs)
    )


@app.route('/task-status/<query>', methods=['GET'])
def task_status_route(query):
    """API to check the status of the search task."""
    
    status = task_status.get(query, "not-found")
    return jsonify({"status": status})


@app.route('/stats/<query>', methods=['GET'])
def stats_page(query):
    """Render the stats page with the generated diagram paths."""
    
    paths = task_status.get(query, {}).get("paths", {})
    return render_template('stats.html', query=query, histograms=paths['histograms'], networks=paths['networks'], word_dists=paths['word_dists'], topics=paths['topics'])


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
