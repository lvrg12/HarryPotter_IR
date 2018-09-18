import re

def documents_with(term,documents):
    docs = []

    for d in documents:
        if term in documents[d]:
            docs.append(d)

    return docs


def document_positions(doc,term):

    return [i for i, x in enumerate(doc) if x == term]