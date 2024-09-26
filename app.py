from flask import Flask, request, jsonify
import pandas as pd
import re
from flask_cors import CORS
from groq import Groq
import traceback

app = Flask(__name__)
CORS(app)

# Initialize the Groq API client with your API key
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# HTML form for file upload
@app.route('/')
def upload_form():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis File Upload</title>
    </head>
    <body>
        <h1>Upload CSV or XLSX File for Sentiment Analysis</h1>
        <form action="/analyze" method="POST" enctype="multipart/form-data">
            <label for="file">Choose file:</label>
            <input type="file" id="file" name="file" accept=".xlsx, .csv" required><br><br>
            <input type="submit" value="Analyze">
        </form>
    </body>
    </html>
    '''

# Function to read reviews from uploaded file
def read_reviews(file):
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        raise ValueError("File format not supported. Please upload a CSV or XLSX file.")

    if 'Review' not in data.columns:
        raise ValueError("Missing 'Review' column in the uploaded file.")

    return data['Review'].dropna().tolist()  # Return a list of non-empty reviews

# Function to analyze sentiment using Groq API
def get_sentiment(review):
    messages = [
        {"role": "user", "content": "Not worth the price at all.🙁"},
        {"role": "assistant", "content": "Negative"},
        {"role": "user", "content": "Better than expected!"},
        {"role": "assistant", "content": "Neutral"},
        {"role": "user", "content": "Would definitely recommend to others."},
        {"role": "assistant", "content": "Positive"},
        {"role": "user", "content": "Does the job, no complaints."},
        {"role": "assistant", "content": "Neutral"},
        {"role": "user", "content": "Very underwhelming experience."},
        {"role": "assistant", "content": "Negative"},
        {"role": "user", "content": "Great value for money!😃"},
        {"role": "assistant", "content": "Positive"},
        {"role": "user", "content": "Arrived damaged, very disappointed."},
        {"role": "assistant", "content": "Negative"},
        {"role": "user", "content": review}
    ]

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    sentiment_result = ""
    for chunk in response:
        content = chunk.choices[0].delta.content or ""
        sentiment_result += re.sub(r"\s+", "", content)  # Clean up the content

    return sentiment_result

# API endpoint to analyze the sentiment of uploaded reviews
@app.route('/analyze', methods=['POST'])
def analyze_reviews():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400
        
        file = request.files['file']
        reviews = read_reviews(file)

        # Initialize sentiment counters
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

        for review in reviews:
            sentiment = get_sentiment(review)
            print(f"Review: '{review}' | Sentiment: {sentiment}")

            if "Positive" in sentiment:
                sentiment_counts["positive"] += 1
            elif "Negative" in sentiment:
                sentiment_counts["negative"] += 1
            elif "Neutral" in sentiment:
                sentiment_counts["neutral"] += 1
        
        return jsonify(sentiment_counts)
    
    except ValueError as ve:
        print(f"Value Error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Unexpected Error: {e}")
        traceback.print_exc()
        return jsonify({"error": "An error occurred during processing."}), 500

# Error handler for 404 errors
@app.errorhandler(404)
def handle_not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)