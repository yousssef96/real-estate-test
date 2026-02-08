#  Real Estate Assistant (Arabic RAG-Powered Search)  

##  Overview  
Traditional real estate platforms (like Nawy, Aqarmap, or dubizzle) often rely on complex, multi-step filtering systems. This can be a significant barrier for non-technical users, particularly older generations.

This project introduces a "Google-style" natural language interface for real estate. Using Retrieval-Augmented Generation (RAG) and Vector Search, users can find their dream home by simply typing their requirements in plain Arabic or English, bypassing the need for tedious manual filters.

##  Key Features  
Natural Language Search: No more checkboxes. Search for "Apartment in New Cairo under 5M with 3 bedrooms" directly.

Multi-Language Support: Optimized for Arabic queries to serve the local Egyptian/MENA market.

Smart Matching & Alternatives: If an exact match doesn't exist, it retrieves the 2–3 closest alternatives.

Automated Comparison: The LLM provides a side-by-side feature comparison of suggested units to help users make informed decisions.

Live Data Sync: Seamlessly integrates with Airtable.

## 🛠️ Tech Stack  
LLM Framework: LangChain / OpenAI

Vector Database: MongoDB Atlas Vector Search

Data Source: Airtable API

Embeddings:  OpenAI text-embedding-3-small 

Try the Live Demo: 
https://xweuxdkzfqcbxfoeawkwte.streamlit.app/
