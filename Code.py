print("Importing Libraries....")
# Importing Libraries
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
import docx2txt
import re
from scipy.spatial.distance import cosine
import openai
from docx import Document
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate
import tiktoken


"""
This script is designed to perform the following tasks:

1. Download necessary NLTK data packages:
    - 'stopwords' for removing common words that do not contribute to the meaning of a sentence.
    - 'punkt' for tokenizing sentences and words.
    - 'wordnet' for lexical database to find the meanings of words and for lemmatization purposes.

2. Initialize a set of stop words in English using NLTK's stopwords corpus.

3. Initialize a WordNetLemmatizer, which is used to reduce words to their base or root form.
"""
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    print("Downloads and initializations were successful.")

except Exception as e:
    print(f"An error occurred: {e}")


"""
Preprocesses a text document by performing the following steps:
1. Removes punctuation and special characters.
2. Tokenizes the text.
3. Converts words to lowercase, removes stopwords, and lemmatizes them.
4. Joins the processed words back into a single string.

Parameters:
    document (str): The text document to be preprocessed.

Returns:
    str: The preprocessed text.
"""
def preprocess(document):
    try:
        # Remove punctuation and special characters
        document = re.sub('[^a-zA-Z0-9]', ' ', document)
        # Tokenize, remove stopwords, and lemmatize
        words = word_tokenize(document)
        words = [lemmatizer.lemmatize(
            word.lower()) for word in words if word.isalpha() and word not in stop_words]

        return ' '.join(words)
    except Exception as e:
        print("An error occurred during preprocessing:", e)


"""
Reads and preprocesses .docx files from a given directory.

Steps:
1. Initializes lists to store raw documents, processed documents, and filenames.
2. Iterates through files in the directory, processing only .docx files.
3. Reads the content of each .docx file.
4. Appends the raw content, preprocessed content, and filenames to their respective lists.
5. Handles any exceptions and prints an error message if one occurs.

Parameters:
    directory (str): The path to the directory containing .docx files.

Returns:
    tuple: Contains lists of raw documents, processed documents, and filenames.
           Returns (None, None, None) if an error occurs.
"""
def read_data(directory):
    try:
        print('Reading Files....')
        raw_documents = []  # Save the Raw Documents
        documents = []  # Save the Processed Documents
        filenames = []  # Save the filenames

        for filename in os.listdir(directory):
            if filename.endswith(".docx"):
                file_path = os.path.join(directory, filename)
                content = docx2txt.process(file_path)
                # Save a copy of the original document
                raw_documents.append(content)
                # Save the preprocessed document
                documents.append(preprocess(content))
                filenames.append(filename)  # Save the filename

        return raw_documents, documents, filenames

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None  # Return None values if an error occurs


"""
Converts a list of documents into TF-IDF vectors.

Steps:
1. Initializes a TfidfVectorizer.
2. Fits and transforms the documents into TF-IDF vectors.
3. Handles any exceptions and prints an error message if one occurs.

Parameters:
    documents (list of str): The list of preprocessed documents.

Returns:
    scipy.sparse.csr.csr_matrix: The TF-IDF vectors.
    Returns None if an error occurs.
"""
def vectors(documents):
    try:
        print('Creating Vectors....')
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(documents)
        return X
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Return None if an error occurs


"""
Clusters documents using Affinity Propagation.

Steps:
1. Initializes an AffinityPropagation model.
2. Fits the model and predicts cluster assignments for the input data.
3. Handles any exceptions and prints an error message if one occurs.

Parameters:
    X (scipy.sparse.csr.csr_matrix): The TF-IDF vectors of the documents.

Returns:
    numpy.ndarray: An array of cluster labels for each document.
    Returns None if an error occurs.
"""
def clusters(X):
    try:
        print('Creating Clusters....')
        affinity_propagation = AffinityPropagation()
        clusters = affinity_propagation.fit_predict(X)
        return clusters
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Return None if an error occurs


"""
Groups documents and filenames by their cluster assignments.

Steps:
1. Initializes dictionaries to store grouped documents and filenames.
2. Iterates over the cluster assignments and groups the corresponding raw documents and filenames.
3. Handles any exceptions and prints an error message if one occurs.

Parameters:
    clusters (numpy.ndarray): An array of cluster labels for each document.
    raw_documents (list of str): The list of raw document texts.
    filenames (list of str): The list of filenames corresponding to the documents.

Returns:
    tuple: Contains two dictionaries:
        - grouped_documents: Groups of raw documents by cluster.
        - grp_doc: Groups of filenames by cluster.
    Returns (None, None) if an error occurs.
"""
def group_doc_by_cluster(clusters, raw_documents, filenames):
    try:
        print('Grouping Files by Cluster.....')
        grouped_documents = defaultdict(list)
        grp_doc = defaultdict(list)

        for i, cluster in enumerate(clusters):
            grouped_documents[cluster].append(raw_documents[i])
            grp_doc[cluster].append(filenames[i])

        return grouped_documents, grp_doc

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None  # Return None values if an error occurs

"""
Displays documents grouped by their cluster assignments.

Steps:
1. Iterates over the cluster groups.
2. Prints the cluster number and the filenames of the documents in that cluster.
3. Handles any exceptions and prints an error message if one occurs.

Parameters:
    grp_doc (dict): A dictionary where keys are cluster labels and values are lists of filenames.

Returns:
    None
"""
def display_clusters(grp_doc):
    try:
        for cluster, docs in grp_doc.items():
            print(f'Cluster {cluster}:')
            for i, doc in enumerate(docs):
                print(f'Document {i+1}: {doc}')
        print('Clusters Created Successfully!')
    except Exception as e:
        print(f"An error occurred: {e}")

"""
Removes duplicate sentences from a text based on similarity.

Steps:
1. Initializes a SentenceTransformer model.
2. Splits the text into sentences.
3. Encodes the sentences into embeddings.
4. Compares each sentence with others to check for similarity above a threshold (0.70).
5. Keeps only unique sentences.
6. Handles any exceptions and prints an error message if one occurs.

Parameters:
    text (str): The input text from which duplicate sentences need to be removed.

Returns:
    str: The text with duplicate sentences removed.
    Returns None if an error occurs.
"""
def remove_duplicates(text):
    try:
        print('Removing Duplicates....')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = text.split('.')
        sentence_embeddings = model.encode(sentences)
        unique_sentences = []

        for i in range(len(sentences)):
            is_unique = True
            for j in range(i + 1, len(sentences)):
                similarity = 1 - \
                    cosine(sentence_embeddings[i], sentence_embeddings[j])
                if similarity > 0.70:  # Similarity Threshold
                    is_unique = False
                    break
            if is_unique:
                unique_sentences.append(sentences[i])

        return '. '.join(unique_sentences)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Return None if an error occurs


"""
Corrects the grammar of a given text using OpenAI's GPT-3.5-turbo model.

Steps:
1. Splits the text into chunks that fit within the token limit.
2. Sends each chunk to OpenAI's API for grammar correction.
3. Combines the corrected chunks back into a single text.

Helper Functions:
- split_into_chunks(text, max_tokens): Splits text into chunks based on the token limit.
- correct_chunk(chunk): Sends a chunk to OpenAI's API for grammar correction.

Handles exceptions and prints error messages if any occur.

Parameters:
    text (str): The input text to be grammatically corrected.

Returns:
    str: The grammatically corrected text.
"""
def correct_grammar(text):
    print('Sending Files to OpenAI....')

    def split_into_chunks(text, max_tokens):
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            for i in range(0, len(tokens), max_tokens):
                yield encoding.decode(tokens[i:i + max_tokens])
        except Exception as e:
            print(f"Error during tokenization or splitting: {e}")
            yield text  # Fallback to yield the original text if there's an error

    def correct_chunk(chunk):
        try:
            prompt = f"This is a task to correct the grammar of the following text:\n{chunk}\n\nInstructions for the system:\nPlease ensure that each sentence is grammatically correct and that the meaning of the text is preserved. Proper punctuation should also be used.\n\nCorrected text:"
            completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {"role": "assistant",
                     "content": prompt,
                     },
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error during API call: {e}")
            return chunk  # Fallback to return the original chunk if there's an error

    max_tokens = 4000
    corrected_text = []

    for chunk in split_into_chunks(text, max_tokens):
        corrected_text.append(correct_chunk(chunk))

    return ' '.join(corrected_text)


"""
Extracts the company name from a given text using OpenAI's GPT-3.5-turbo model.

Steps:
1. Constructs a prompt to instruct the model to extract the company name.
2. Sends the prompt to OpenAI's API.
3. Retrieves the company name from the API response.
4. Handles exceptions and prints an error message if one occurs.

Parameters:
    text (str): The input text from which to extract the company name.

Returns:
    str: The extracted company name.
    Returns None if an error occurs.
"""
def recognize_title(text):
    try:
        prompt = f"Extract and provide only the name of the company specifically mentioned in the provided text:\n{text}\n\nCompany Name:"

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "assistant",
                 "content": prompt,
                 },
            ]
        )

    # return response.choices[0].text.strip()
        return completion.choices[0].message.content

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Return None if an error occurs


"""
Processes a list of texts to correct grammar, punctuation, and extract company titles, then formats the results into JSON-compatible dictionaries.

Steps:
1. Iterates over each text in the input list.
2. Corrects the grammar and punctuation of the text.
3. Extracts the company title from the corrected text.
4. Appends the corrected text and extracted title to a list of dictionaries.
5. Handles exceptions and prints an error message if one occurs.

Parameters:
    duplicate (list of str): The list of texts to be processed.

Returns:
    list of dict: A list of dictionaries containing the corrected text and extracted title.
    Returns None if an error occurs.
"""
def json_doc(duplicate):
    try:
        print('Checking for Grammar, Punctuation and Extracting Titles.....')
        corrected_texts = []
        for text in duplicate:
            corrected_text = correct_grammar(text)
            if corrected_text is None:
                continue  # Skip this iteration if an error occurred

            title = recognize_title(corrected_text)
            if title is None:
                continue  # Skip this iteration if an error occurred

            corrected_texts.append({
                "title": title,
                "corrected_text": corrected_text
            })
        return corrected_texts

    except Exception as e:
        print(f"An error occurred in json_doc: {e}")
        return None  # Return None if an error occurs

"""
Saves a list of corrected texts and titles to individual .docx files.

Steps:
1. Checks if the output directory exists; if not, creates it.
2. Iterates over the data list.
3. Creates a new .docx document for each entry with the corrected text.
4. Saves the document with a filename based on the extracted title.
5. Handles exceptions during both the directory creation and file saving processes, and prints error messages if any occur.

Parameters:
    output_dir (str): The directory where the files will be saved.
    data_list (list of dict): A list of dictionaries containing corrected texts and titles.

Returns:
    None
"""
def saving_file(output_dir, data_list):
    try:
        print('Saving Files....')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for data in data_list:
            try:
                # Create a new Document
                doc = Document()
                doc.add_paragraph(data['corrected_text'])

                # Save the document
                file_name = f'{data["title"]}_consolidated.docx'
                doc.save(os.path.join(output_dir, file_name))
            except Exception as e:
                print(
                    f"An error occurred while processing data: {data}. Error: {e}")

        print("Files Saved Successfully...")

    except Exception as e:
        print(f"An error occurred in saving_file: {e}")

"""
Extracts corrected texts and titles from a list of dictionaries, handling potential errors.

Steps:
1. Initializes lists for deduplicated content and titles.
2. Iterates over the data list.
3. Appends the 'corrected_text' and 'title' from each dictionary to their respective lists.
4. Handles KeyError if a dictionary is missing the 'corrected_text' or 'title' key.
5. Handles any unexpected exceptions and prints error messages if any occur.

Parameters:
    data_list (list of dict): A list of dictionaries containing corrected texts and titles.

Returns:
    tuple: Two lists - one with deduplicated content (corrected texts) and one with titles.
    Returns (None, None) if an error occurs.
"""
def refined_text(data_list):
    try:
        deduplicated_content = []
        titles = []
        for data in data_list:
            try:
                deduplicated_content.append(data['corrected_text'])
                titles.append(data['title'])
            except KeyError as e:
                print(
                    f"KeyError: {e}. Missing 'corrected_text' in data: {data}")
            except Exception as e:
                print(
                    f"An unexpected error occurred while processing data: {data}. Error: {e}")

        return deduplicated_content, titles

    except Exception as e:
        print(f"An error occurred in refined_text: {e}")
        return None, None  # Return None if an error occurs

"""
Calculates the cosine similarity between two text strings.

Steps:
1. Vectorizes the input texts using TF-IDF.
2. Converts the vectorized texts to arrays.
3. Computes the cosine similarity between the two vectors.
4. Handles any exceptions and prints an error message if one occurs.

Parameters:
    text1 (str): The first text string.
    text2 (str): The second text string.

Returns:
    float: The cosine similarity score between text1 and text2.
    Returns None if an error occurs.
"""
def calculate_cosine_similarity(text1, text2):
    try:
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        csim = cosine_similarity(vectors)
        return csim[0, 1]

    except Exception as e:
        print(f"An error occurred in calculate_cosine_similarity: {e}")
        return None  # Return None if an error occurs

"""
Evaluates the similarity between input and output texts and compiles the results into a DataFrame.

Steps:
1. Ensures the input and output lists have the same length.
2. Calculates cosine similarity scores between corresponding input and output texts.
3. Computes the lengths of input and output texts.
4. Creates a DataFrame containing company names, similarity scores, and text lengths.
5. Handles any exceptions and prints an error message if one occurs.

Parameters:
    input_file (list of str): The list of original input texts.
    output_file (list of str): The list of processed output texts.
    titles (list of str): The list of company names corresponding to the texts.

Returns:
    pd.DataFrame: A DataFrame with columns for company names, similarity scores, input lengths, and output lengths.
    Returns None if an error occurs.
"""
def evaluation(input_file, output_file, titles):
    try:
        list1 = input_file
        list2 = output_file

        # Ensure that the lengths of the input and output lists are the same
        if len(list1) != len(list2):
            raise ValueError(
                "The length of input_file and output_file must be the same.")

        # Calculate similarity scores and lengths
        similarity_scores = [calculate_cosine_similarity(
            list1[i], list2[i]) for i in range(len(list1))]
        lengths = [(len(list1[i]), len(list2[i])) for i in range(len(list1))]

        # Create a DataFrame
        data = {
            'Company Name': titles,
            'Similarity Score': similarity_scores,
            'Input Length': [length[0] for length in lengths],
            'Output Length': [length[1] for length in lengths]
        }

        df = pd.DataFrame(data)
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    
    try:
        load_dotenv()
        input_directory = os.getenv('INPUT_FOLDER_PATH')
        output_directory = os.getenv('OUTPUT_FOLDER_PATH')
        openai.api_key = os.getenv('API_KEY')
    except Exception as e:
        print(f"An error occurred while loading environment variables: {e}")
        exit(1)  # Exit if environment variables cannot be loaded

    raw_documents, documents, filenames = read_data(input_directory)

    X = vectors(documents)
    cluster = clusters(X)

    grouped_documents, grp_doc = group_doc_by_cluster(
        cluster, raw_documents, filenames)
    display_clusters(grp_doc)

    print('Merging Files....')
    merged_content = []
    for doc in grouped_documents:
        text = ' '.join(grouped_documents[doc])
        merged_content.append(text)

    duplicated_text = []
    for text in merged_content:
        duplicated_text.append(remove_duplicates(text))

    data_list = json_doc(duplicated_text)

    saving_file(output_directory, data_list)

    refined, titles = refined_text(data_list)
    df = evaluation(merged_content, refined, titles)
    print('Evaluation and Results:')
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False)) # printing dataframe
