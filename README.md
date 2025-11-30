AI Emotion Detection Project 

    This project analyzes diary/journal style text and assigns it to an emotional cluster using text embeddings and clustering.
    FastAPI backend processes the text, and a simple webpage allows users to test the system.

How to run the code:

        1. Download the project and make sure you have app.py, index.html, style.css, and the entire data folder
        
        2. Install dependencies using, pip install fastapi uvicorn pandas numpy scikit-learn joblib sentence-transformers pydantic
        
        3. start the FastAPI server using, uvicorn app:app
        
        4. open index.html in your browser to use the interface


Overview:

    -Converts text into embeddings
    
    -Reduces dimensions with PCA
    
    -Assigns the text to the closest emotional cluster
    
    -Returns the emotion label, and valence/arousal
    
    -Includes a interface to test the system

Files:

     -app.py - FastAPI server

     -data - files created from backend 

     -notebook - preprocessing, clustering, and visualization notebooks

     -index.html - frontend

     -style.css - styling for the interface 

Team Members:

    Laiba Baig

    Daniel Mondragon

    Nicholas Alberto

    David Osemene

    Alan Trujillo




