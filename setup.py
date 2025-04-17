import setuptools
import os

def read_readme():
    try:
        with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
            return f.read()
    except IOError:
        return ""
# langchain-core might be sufficient, or you might need the full langchain package
# torch dependency might vary based on user's system (CPU/GPU) - consider advising users
install_requires = [
    "numpy>=1.21.0",                
    "langchain-core>=0.1.0",        
    # Add full 'langchain' or specific components if needed, e.g., 'langchain-openai'
    "pymilvus>=2.3.0,<2.4.0",       
    "sentence-transformers>=2.2.0", 
    "torch>=1.9.0"                 
]

setuptools.setup(
    name="MemForest",
    version="1.0.0", 
    author="Babel5430", 
    author_email="babel5430@gmail.com>", 
    description="A hierarchical memory system for chatbots with persistence and summarization.",
    long_description=read_readme(),
    long_description_content_type="text/markdown", 
    url="https://github.com/Babel5430/MemForest", 
    packages=setuptools.find_packages(), 
    install_requires=install_requires,
    python_requires=">=3.8", 
    classifiers=[ 
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
    ],
    keywords="chatbot memory nlp ai rag langchain milvus", 
)
