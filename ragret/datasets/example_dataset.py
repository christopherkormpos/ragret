example_dataset = [
    {
        "user_query": "What are your store hours?",
        "retrieved_documents": [
            "Our store is open Monday through Friday from 9am to 6pm.",
            "On weekends our store is open from 10am to 4pm.",
            "Our store carries a wide variety of products across multiple categories."
        ],
        "llm_answer": "We are open Monday to Friday 9am-6pm and weekends 10am-4pm. We are closed on public holidays."
    },
    {
        "user_query": "Do you offer free shipping?",
        "retrieved_documents": [
            "Orders over $50 qualify for free standard shipping within the US.",
            "Expedited and international shipping are always charged at standard rates.",
            "Free shipping is automatically applied at checkout when the order total meets the threshold."
        ],
        "llm_answer": "Yes, orders over $50 get free standard shipping within the US, applied automatically at checkout."
    },
    {
        "user_query": "What is your return policy?",
        "retrieved_documents": [
            "Items can be returned within 30 days of purchase for a full refund.",
            "Products must be unused and in their original packaging to be eligible for a return.",
            "Digital downloads and gift cards are non-refundable."
        ],
        "llm_answer": "You can return unused items in original packaging within 30 days for a full refund. Digital downloads and gift cards cannot be returned."
    }
]