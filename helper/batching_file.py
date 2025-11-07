MAX_VECTOR_SIZE_BYTES = 1536 * 4
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024  
SAFE_VECTOR_COUNT = MAX_PAYLOAD_BYTES // MAX_VECTOR_SIZE_BYTES

def batch_documents_by_size(documents, ids, max_total_chars=200000):
    batches = []
    current_docs = []
    current_ids = []
    current_total = 0

    for doc, doc_id in zip(documents, ids):
        content_len = len(doc.page_content)
        if current_total + content_len > max_total_chars:
            batches.append((current_docs, current_ids))
            current_docs = []
            current_ids = []
            current_total = 0

        current_docs.append(doc)
        current_ids.append(doc_id)
        current_total += content_len

    if current_docs:
        batches.append((current_docs, current_ids))

    return batches