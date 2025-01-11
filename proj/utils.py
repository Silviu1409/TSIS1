import pickle
import matplotlib.pyplot as plt
import seaborn as sbs
import litstudy
import os
import pandas as pd
from pandas.plotting import table
import shutil
import logging
from pyvis.network import Network
import spacy
from functools import reduce
from operator import or_
from threading import Event
import app


if not os.path.exists('static/graphs'):
    os.makedirs('static/graphs')

if not os.path.exists('data/uploads'):
    os.makedirs('data/uploads')

plt.rcParams['figure.figsize'] = (10, 6)
sbs.set_theme('paper')
logging.getLogger().setLevel(logging.CRITICAL)

net = Network(notebook=True, cdn_resources='remote')
net.repulsion()
nlp = spacy.load("en_core_web_md")
process_cancel_event = Event()


def load_docs_from_pickle(pickle_file_path):
    """Loads docs from the pickle file if it exists."""
    
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            docs = pickle.load(f)
            print(len(docs), 'papers loaded from pickle')
    else:
        docs = set()
    
    return docs


def save_docs_to_pickle(docs, pickle_file_path):
    """Saves docs_remaining to the pickle file."""
    
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(docs, f)
        print(len(docs), 'papers saved to pickle')


def load_and_process_files(files, docs_file_path='data/docs_remaining.pkl', docs_crossref_file_path='data/docs_crossref.pkl'):
    """Loads and merges documents from CSV files, excludes documents based on RIS files, and saves them."""
    
    global process_cancel_event

    process_cancel_event.clear()

    # Load existing docs
    docs_remaining = load_docs_from_pickle(docs_file_path)
    docs_crossref = load_docs_from_pickle(docs_crossref_file_path)

    docs_csv = litstudy.DocumentSet(docs=[])
    docs_exclude = litstudy.DocumentSet(docs=[])

    # Iterate through the uploaded files
    for file in files:
        if file.rsplit('.', 1)[1].lower() == 'csv':
            docs_csv |= litstudy.load_csv(file)
        elif file.rsplit('.', 1)[1].lower() == 'ris':
            docs_exclude |= litstudy.load_ris_file(file)

    # Exclude documents from the RIS files
    docs_remaining_new = docs_csv - docs_exclude
    docs_crossref_new, _ = litstudy.refine_crossref(docs_remaining_new, timeout=0.1)

    docs_remaining_new = docs_remaining | docs_remaining_new
    docs_crossref_new = docs_crossref | docs_crossref_new

    save_docs_to_pickle(docs_remaining_new, 'data/docs_remaining.pkl')
    save_docs_to_pickle(docs_crossref_new, 'data/docs_crossref.pkl')

    # Mark stats generation as in progress
    app.stats_in_progress = True

    # Perform stats processing if not canceled
    try:
        if not process_cancel_event.is_set():
            plot_histograms(docs_remaining_new, docs_crossref_new, "static/graphs/")

        if not process_cancel_event.is_set():
            plot_cocitation_network(docs_crossref_new, "static/graphs/")

        if not process_cancel_event.is_set():
            plot_coupling_network(docs_crossref_new, "static/graphs/")
        
        if not process_cancel_event.is_set():
            generate_stats(docs_remaining_new, "static/graphs/")
    finally:
        app.stats_in_progress = False
    
    return


def annotate_bars(ax):
    for bar in ax.patches:
        if isinstance(bar, plt.Rectangle):
            height = bar.get_height() if bar.get_height() else bar.get_width()
            width = bar.get_width() if bar.get_width() else bar.get_height()

            # Only annotate if the value is greater than 0
            if height > 0:
                if bar.get_height() > bar.get_width():  # Vertical bars
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 5),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)
                else:  # Horizontal bars
                    ax.annotate(f'{int(width)}',
                                xy=(width, bar.get_y() + bar.get_height() / 2),
                                xytext=(5, 0),
                                textcoords="offset points",
                                ha='left', va='center', fontsize=9)


def plot_histograms(docs, docs_crossref, folder):
    """Generate and save the histogram plots."""

    try:
        # Plot Year Histogram
        _, ax = plt.subplots()
        litstudy.plot_year_histogram(docs, vertical=True, ax=ax)
        annotate_bars(ax)
        plt.tight_layout()
        plt.savefig(folder + 'year_histogram.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Plot Affiliation Histogram
        _, ax = plt.subplots()
        litstudy.plot_affiliation_histogram(docs, limit=15, ax=ax)
        annotate_bars(ax)
        plt.tight_layout()
        plt.savefig(folder + 'affiliation_histogram.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Get docs that only have authors
        docs_filtered = docs.filter_docs(lambda d: d.authors and d.authors[0].name != '')

        # Plot Author Histogram
        _, ax = plt.subplots()
        litstudy.plot_author_histogram(docs_filtered, ax=ax)
        annotate_bars(ax)
        plt.tight_layout()
        plt.savefig(folder + 'author_histogram.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Plot Number of Authors Histogram
        _, ax = plt.subplots()
        litstudy.plot_number_authors_histogram(docs, ax=ax)
        annotate_bars(ax)
        plt.tight_layout()
        plt.savefig(folder + 'number_authors_histogram.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Plot Publication Source Histogram
        _, ax = plt.subplots()
        litstudy.plot_source_histogram(docs_crossref, limit=15, ax=ax)
        annotate_bars(ax)
        plt.tight_layout()
        plt.savefig(folder + 'publication_source_histogram.png', bbox_inches='tight', dpi=300)
        plt.close()

        logging.info("Histograms saved successfully.")
    except Exception as e:
        logging.error(f"Error generating histograms: {e}")


def plot_cocitation_network(docs_crossref, folder):
    """Generate and save the cocitation network graph to the desired location."""
    
    try:
        # Generate the network graph, which outputs 'citation.html'
        litstudy.plot_cocitation_network(docs_crossref)

        # Source file generated by litstudy
        source_file = 'citation.html'

        # Destination path
        output_path = folder + 'cocitation_network.html'

        # Ensure the source file exists
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Expected file {source_file} not found.")

        # Move and rename the file to the desired location
        shutil.move(source_file, output_path)
        logging.info(f"Co-citation network successfully moved to {output_path}.")
    except Exception as e:
        logging.error(f"Error generating or moving co-citation network: {e}")


def plot_coupling_network(docs_crossref, folder, max_edges=5000):
    """Generate and save the coupling network graph to the desired location."""
    
    try:
        # Generate the network graph, which outputs 'citation.html'
        litstudy.plot_coupling_network(docs_crossref, max_edges=max_edges)

        # Source file generated by litstudy
        source_file = 'citation.html'

        # Destination path
        output_path = folder + 'coupling_network.html'

        # Ensure the source file exists
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Expected file {source_file} not found.")

        # Move and rename the file to the desired location
        shutil.move(source_file, output_path)
        logging.info(f"Coupling network successfully moved to {output_path}.")
    except Exception as e:
        logging.error(f"Error generating or moving coupling network: {e}")


def save_table_as_png(dataframe, path, figsize=(8, 6), scale_factor=1.2):
    _, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    if dataframe.empty:
        return
    
    tbl = table(ax, dataframe, loc='center', colWidths=[0.3] * len(dataframe.columns))

    # Set font size and scale
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(scale_factor, scale_factor)  # Adjust the scale for better fit

    # Adjust layout to center the table
    plt.tight_layout(pad=2.0)  # Add padding to center the table

    # Save the table as PNG
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()


def generate_stats(docs, save_dir, num_topics=15, ngram_threshold=0.8):
    # Build the corpus
    corpus = litstudy.build_corpus(docs, ngram_threshold=ngram_threshold)

    if (len(corpus.dictionary) == 0):
        return
    
    # 1. Save word distribution plot (graph)
    word_dist_path = os.path.join(save_dir, "word_distribution.png")
    _, ax = plt.subplots()
    litstudy.plot_word_distribution(corpus, limit=50, title="Top words", vertical=True, label_rotation=45, ax=ax)
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(word_dist_path, bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Save word distribution table as PNG
    word_dist_table_path = os.path.join(save_dir, "word_distribution_table.png")
    word_dist_table = litstudy.compute_word_distribution(corpus).filter(like='_', axis=0).sort_index().sort_values(by='count', ascending=False)
    save_table_as_png(word_dist_table, word_dist_table_path, figsize=(12, 6), scale_factor=1.5)

    # Train the topic model
    topic_model = litstudy.train_nmf_model(corpus, num_topics=num_topics, max_iter=100)
    
    # 3. Save topic modeling table as PNG
    topic_table_path = os.path.join(save_dir, "topic_modeling_table.png")
    topics_data = []
    for i in range(num_topics):
        best_tokens = topic_model.best_tokens_for_topic(i, 10)
        # Limit token length to avoid overlap
        truncated_tokens = ", ".join(best_tokens[:5]) + ("..." if len(best_tokens) > 5 else "")
        topics_data.append({"Top Tokens": truncated_tokens})

    # Create the DataFrame and set the index to start from 1
    topic_table = pd.DataFrame(topics_data, index=range(1, len(topics_data) + 1))
    save_table_as_png(topic_table, topic_table_path, figsize=(14, 6), scale_factor=1.5)

    # 4. Save topic clouds plot (graph)
    topic_cloud_path = os.path.join(save_dir, "topic_cloud.png")
    plt.figure(figsize=(15, 5))
    litstudy.plot_topic_clouds(topic_model, ncols=5)
    plt.savefig(topic_cloud_path, bbox_inches='tight', dpi=300)
    plt.close()


def get_similar_words_spacy(word, similarity_threshold):
    similar_words = set()
    
    # Add original word and its lemmatized form
    doc = nlp(word)
    similar_words.add(word)
    for token in doc:
        similar_words.add(token.lemma_)
    
    # Add similar words using word vectors
    token = nlp(word)[0]
    for other_token in nlp.vocab:
        if other_token.has_vector and other_token.is_lower and other_token.is_alpha:
            similarity = token.similarity(other_token)
            if similarity >= similarity_threshold:
                similar_words.add(other_token.text)
    
    return similar_words


def filter_by_keyword(query):
    """Filter documents based on a keyword."""
    
    docs = load_docs_from_pickle('data/docs_remaining.pkl')
    docs_crossref = load_docs_from_pickle('data/docs_crossref.pkl')
    query_lower = query.lower().split()

    search_words = reduce(or_, [get_similar_words_spacy(x, 0.5) for x in query_lower])

    filtered_docs = docs.filter_docs(
        lambda d: any(
            q in d.title.lower().strip() or
            any(
                q in split_word.lower() 
                for sublist in (d.keywords or []) 
                for keyword in sublist 
                for split_word in keyword.split()
            )
            for q in search_words
        )
    )

    import re
    filtered_dois = {
        re.search(r"'doi': '([^']+)'", str(doc.__dict__['_identifier'])).group(1).strip()
        for doc in filtered_docs
        if re.search(r"'doi': '([^']+)'", str(doc.__dict__['_identifier']))
    }
    filtered_docs_crossref = docs_crossref.filter_docs(lambda d: d.id.doi.strip() in filtered_dois)
    filtered_dois = {
        doc.id.doi.strip()
        for doc in filtered_docs_crossref
    }
    filtered_docs = filtered_docs.filter_docs(lambda d: d.id.doi and d.id.doi.strip() in filtered_dois)
    
    return filtered_docs, filtered_docs_crossref
