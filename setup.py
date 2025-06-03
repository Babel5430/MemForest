import setuptools
import os

def read_readme():
    try:
        readme_path = os.path.join(os.path.dirname(__file__), "README.md")
        if not os.path.exists(readme_path):
             readme_path = os.path.join(os.path.dirname(__file__), "README_en.md") # Fallback
        
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    except IOError:
        return "MemForest: A hierarchical memory system for roleplay chatbots with persistence, summarization, and multiple vector store backends."

install_requires = [
    "numpy>=1.21.0",
    "langchain-core>=0.1.0",
    "aiosqlite>=0.19.0",   
    "tokenizers>=0.19.0",
    "onnxruntime>=1.18.0",

    # Optimal
    "pymilvus>=2.3.0,<2.5.0", # For Milvus/Zilliz Cloud
    "qdrant-client>=1.9.0",   # For Qdrant
    "chromadb>=0.5.0",        # For ChromaDB
    "sqlite-vec>=0.2.1",      # For SQLite-vec extension

    # For comman vector embeding
    "sentence-transformers>=2.2.0",
    "torch>=1.9.0"
]

setuptools.setup(
    name="MemForest",
    version="1.1.0",
    author="Babel5430",
    author_email="babel5430@gmail.com", 
    description="A hierarchical memory system for roleplay chatbots with async operations, persistence, summarization, and multiple vector store options.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Babel5430/MemForest",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Development Status :: 4 - Beta"
    ],
    keywords="chatbot memory nlp ai rag langchain milvus qdrant chroma sqlite roleplay hierarchical memory async",
    project_urls={
        'Bug Reports': 'https://github.com/Babel5430/MemForest/issues',
        'Source': 'https://github.com/Babel5430/MemForest/',
    },
)
