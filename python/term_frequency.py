import re

# returns list of documents where term appears
def documents_with(term,documents):
    docs = []

    for d in documents:
        if term in documents[d]:
            docs.append(d)

    return docs

# returns list of position where term appears in doc
def document_positions(doc,term):

    return [i for i, x in enumerate(doc) if x == term]