# import packages
import json
import os
import logging

from flask_cors import CORS
from flask import Flask, request, jsonify
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import clean_wiki_text, convert_files_to_docs
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes.retriever import DensePassageRetriever, ElasticsearchRetriever
from haystack.nodes.reader import FARMReader

#application settings
app = Flask(__name__)
CORS(app)

# Application directory for inputs and training
app.config["input"] = "/usr/src/app/data/input"
app.config["train_model"] = "/usr/src/app/data/train_model"
app.config["squad_data"] = "/usr/src/app/data/squad20"

# ElasticSearch server host information
app.config["host"] = "0.0.0.0"
app.config["username"] = ""
app.config["password"] = ""
app.config["port"] = "9200"

@app.route('/')
def home():
    """Return a friendly HTTP greeting."""
    return 'Hello QNA API is running'

#endpoint to update embedded method
@app.route('/set_embedded', methods=['POST'])
def set_embedded():
    """Return a friendly HTTP greeting."""
    index = request.form['index']
    document_store = ElasticsearchDocumentStore(host=app.config["host"],
                                                port=app.config["port"],
                                                username=app.config["username"],
                                                password=app.config["password"],
                                                index=index,
                                                embedding_field="embedding",
                                                embedding_dim=768)
    retriever = DensePassageRetriever(document_store=document_store,
                                      embedding_model="dpr-bert-base-nq",
                                      do_lower_case=True, use_gpu=False)
    #Now update the retriever embedded to the elasticsearch document
    document_store.update_embeddings(retriever)
    return json.dumps({'status':'Susccess','message': 'Sucessfully embedded method updated in ElasticSearch Document', 'result': []})

@app.route('/update_document', methods=['POST'])
def update_document():
    """Return a the url of the index document."""
    if request.files:
        # index is the target document where queries need to sent.
        index = request.form['index']
        # uploaded document for target source
        doc = request.files["doc"]

        file_path = os.path.join(app.config["input"], doc.filename)

        # saving the file to the input directory
        doc.save(file_path)
        #initialization of the Haystack Elasticsearch document storage
        document_store = ElasticsearchDocumentStore(host=app.config["host"],
                                                    port=app.config["port"],
                                                    username=app.config["username"],
                                                    password=app.config["password"],
                                                    index=index)
        # convert the pdf files into dictionary and update to ElasticSearch Document
        docs = convert_files_to_docs(
            app.config["input"],
            clean_func=clean_wiki_text,
            split_paragraphs=False)
        document_store.write_documents(docs)
        os.remove(file_path)
        return json.dumps(
            {'status':'Susccess','message':
                'document available at http://'+ app.config["host"] +':'
                + app.config["port"] +'/' + index + '/_search',
                'result': []})
    else:
        return json.dumps({'status':'Failed','message': 'No file uploaded', 'result': []})


@app.route('/qna', methods=['POST'])
def qna():
    """Return the n answers."""

    question = request.form['question']
    # index is the target document where queries need to sent.
    index = request.form['index']

    # to select train or untrained model
    mode = request.form['mode']

    #initialization of the Haystack Elasticsearch document storage
    document_store = ElasticsearchDocumentStore(
        host=app.config["host"],
        username=app.config["username"],
        password=app.config["password"],
        index=index)

    if mode == 'trained':
        # base on the search mode train_model
        reader = FARMReader(model_name_or_path=app.config["train_model"] ,
                            use_gpu=False)
    else:
        # base on the search mode pre_train
        reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad",
                            use_gpu=False)

    #initialization of ElasticRetriever
    retriever = ElasticsearchRetriever(document_store= document_store)
    # ExtractiveQAPipeline sticks together reader and retriever
    # in a pipeline to answer our actual questions.
    qa_pipe = ExtractiveQAPipeline(reader=reader, retriever=retriever)

    # predict n answers
    n = int(request.form['n'])
    prediction = qa_pipe.run(query=question, top_k_retriever=10, top_k_reader=n)
    answer = []
    for res in prediction['documents']:
        answer.append(res['documents'])

    return json.dumps({'status':'success','message': 'Process succesfully', 'result': answer})


@app.route('/qna_pretrain', methods=['POST'])
def qna_pretrain():
    """Return the n answers."""

    question = request.form['question']
    # index is the target document where queries need to sent.
    index = request.form['index']

    #initialization of the Haystack Elasticsearch document storage
    document_store = ElasticsearchDocumentStore(
        host=app.config["host"],
        username=app.config["username"],
        password=app.config["password"],
        index=index)

    # using pretrain model
    reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad",
                            use_gpu=False)

    #initialization of ElasticRetriever
    retriever = ElasticsearchRetriever(document_store= document_store)
    # ExtractiveQAPipeline sticks together reader and retriever
    # in a pipeline to answer our actual questions.
    qa_pipe = ExtractiveQAPipeline(reader=reader, retriever=retriever)

    # predict n answers
    n = int(request.form['n'])
    prediction = qa_pipe.run(query=question, top_k_retriever=10, top_k_reader=n)
    answer = []
    for res in prediction['documents']:
        answer.append(res['documents'])

    return json.dumps({'status':'success','message': 'Process succesfully', 'result': answer})


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return json.dumps({'status':'failed','message':
        """An internal error occurred: <pre>{}</pre>See logs for full stacktrace.""".format(e),
                       'result': []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8777)
