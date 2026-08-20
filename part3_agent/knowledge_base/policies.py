"""
Flipkart-style policy knowledge base.
Each doc_id maps to a short (2-4 sentence) policy document.
Chunking strategy: sentence-wise (one chunk per sentence), not fixed-size/overlapping --
this keeps each retrievable unit topically atomic, which suits short policy sentences better
than arbitrary character windows.
"""

import re

POLICY_DOCS = {
    "return_window_apparel": (
        "Apparel and footwear items can be returned within 14 days of delivery. "
        "The item must be unused, unwashed, and returned with original tags and packaging. "
        "Refunds are issued after the returned item passes a quality check at the warehouse."
    ),
    "return_window_electronics": (
        "Electronics items have a 7-day return window from the date of delivery. "
        "Items must include all original accessories, manuals, and undamaged packaging. "
        "Physical damage or missing accessories will result in a rejected return."
    ),
    "return_window_home": (
        "Home and furniture items can be returned within 10 days of delivery. "
        "Large furniture returns require a reverse-pickup slot to be scheduled in advance. "
        "Assembled furniture must be disassembled by the customer before pickup."
    ),
    "cod_refund_timeline": (
        "For Cash on Delivery orders, refunds are issued to the customer's bank account, not back as cash. "
        "COD refunds typically take 7-10 business days to reflect after the return is approved. "
        "Customers must provide valid bank account details through the returns portal to receive the refund."
    ),
    "prepaid_refund_timeline": (
        "For prepaid orders, refunds are credited back to the original payment method. "
        "Card and UPI refunds usually reflect within 3-5 business days after approval. "
        "Wallet refunds are typically instant once the return is approved."
    ),
    "delivery_sla_standard": (
        "Standard delivery orders are delivered within 4-7 business days depending on the destination pincode. "
        "Delivery times may extend during sale events or in remote serviceable areas. "
        "Customers receive SMS and app notifications at each delivery milestone."
    ),
    "delivery_sla_express": (
        "Express delivery orders are delivered within 1-2 business days in serviceable metro pincodes. "
        "Express delivery carries an additional shipping fee shown at checkout. "
        "Express delivery is not available for large or heavy furniture items."
    ),
    "reverse_pickup_eligibility": (
        "Reverse pickup is available for most categories except perishable goods and personal care items opened after delivery. "
        "A pickup agent will collect the item from the same address the order was delivered to. "
        "If reverse pickup is unavailable in a pincode, customers must self-ship the item and upload the courier receipt."
    ),
    "damaged_item_policy": (
        "If an item arrives damaged or defective, customers must report it within 48 hours of delivery. "
        "Photo or video evidence of the damage is required to process the claim. "
        "Damaged-item claims are exempt from the standard return window and processed on priority."
    ),
    "wrong_item_policy": (
        "If a customer receives the wrong item, a free replacement or full refund is offered. "
        "Wrong-item complaints must be raised within 3 days of delivery for fastest resolution. "
        "No return shipping charge applies to wrong-item cases."
    ),
    "cancellation_policy": (
        "Orders can be cancelled for free anytime before they are shipped. "
        "Once an order is shipped, cancellation is not possible and the customer must use the return process instead. "
        "Cancelled prepaid orders are refunded via the prepaid refund timeline policy."
    ),
    "cod_availability": (
        "Cash on Delivery is available for orders under a certain value threshold in most serviceable pincodes. "
        "High-value electronics above a set price cap may not be eligible for COD due to fraud-prevention rules. "
        "COD eligibility is shown on the product page before checkout."
    ),
    "warranty_claims": (
        "Manufacturer warranty claims for electronics are handled directly by the brand's authorized service center, not by Flipkart's return process. "
        "Flipkart provides the original invoice needed to register a warranty claim. "
        "Extended warranty plans purchased at checkout are serviced through the third-party warranty provider."
    ),
    "installation_services": (
        "Large appliances such as air conditioners and washing machines include free standard installation in serviceable pincodes. "
        "Installation is scheduled separately after delivery, typically within 48 hours. "
        "Installation charges may apply for non-standard setups like additional piping or wiring."
    ),
}


def chunk_sentence_wise(doc_id: str, text: str):
    """Split a document into sentence-level chunks. Returns list of
    (chunk_id, chunk_text, parent_doc_id) tuples. Multi-sentence docs
    yield multiple chunks, satisfying the 'more than one chunk per doc' requirement."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return [
        (f"{doc_id}__s{i}", sentence, doc_id) for i, sentence in enumerate(sentences)
    ]


def build_all_chunks():
    """Returns: list of (chunk_id, chunk_text, parent_doc_id), and a
    chunk_id -> parent_doc_id lookup map for document-level retrieval scoring."""
    all_chunks = []
    for doc_id, text in POLICY_DOCS.items():
        all_chunks.extend(chunk_sentence_wise(doc_id, text))
    chunk_to_doc = {chunk_id: doc_id for chunk_id, _, doc_id in all_chunks}
    return all_chunks, chunk_to_doc


# Retrieval-evaluation answer key: query -> set of relevant doc_ids (document-level, Task 10)
RETRIEVAL_EVAL_QUERIES = [
    {
        "query": "How long do I have to return a shirt I bought?",
        "relevant_docs": {"return_window_apparel"},
    },
    {
        "query": "When will I get my refund if I paid cash on delivery?",
        "relevant_docs": {"cod_refund_timeline"},
    },
    {
        "query": "Can I get same-day or next-day delivery?",
        "relevant_docs": {"delivery_sla_express"},
    },
    {
        "query": "My laptop arrived broken, what do I do?",
        "relevant_docs": {"damaged_item_policy", "return_window_electronics"},
    },
    {
        "query": "Will someone come pick up my return or do I have to ship it myself?",
        "relevant_docs": {"reverse_pickup_eligibility"},
    },
    {
        "query": "Can I cancel my order after it's already been shipped?",
        "relevant_docs": {"cancellation_policy"},
    },
]


if __name__ == "__main__":
    all_chunks, chunk_to_doc = build_all_chunks()
    print(f"Total policy documents: {len(POLICY_DOCS)}")
    print(f"Total sentence-wise chunks: {len(all_chunks)}")
    print(f"Avg chunks per doc: {len(all_chunks) / len(POLICY_DOCS):.2f}")
    print(f"\nSample chunk -> parent doc mapping:")
    for chunk_id, text, doc_id in all_chunks[:5]:
        print(f"  {chunk_id:35s} -> {doc_id}")
    print(f"\nRetrieval eval queries defined: {len(RETRIEVAL_EVAL_QUERIES)}")
