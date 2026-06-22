import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    """Converts text to lowercase and removes punctuation for a fair comparison."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def check_plagiarism(text_a, text_b):
    """Calculates the similarity percentage between two texts."""
    clean_a = preprocess_text(text_a)
    clean_b = preprocess_text(text_b)
    
    # Convert texts into TF-IDF mathematical vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([clean_a, clean_b])
    
    # Calculate the similarity score (value between 0 and 1)
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    # Convert to a clean percentage string
    percentage = similarity_score * 100
    return percentage

# --- Simple Console Interface to Test ---
print("=" * 50)
print("       CODEALPHA AI PLAGIARISM CHECKER       ")
print("=" * 50)

# Input original reference text
print("\nEnter the Original Text:")
original_text = input("> ")

# Input text to check
print("\nEnter the Text to Check for Plagiarism:")
checked_text = input("> ")

print("\nAnalyzing texts...")
print("-" * 50)

# Run the check
similarity_result = check_plagiarism(original_text, checked_text)

print(f"Match Similarity: {similarity_result:.2f}%")

# Give a verdict based on standard thresholds
if similarity_result > 70:
    print("Verdict: ALERT! High risk of direct plagiarism.")
elif similarity_result > 30:
    print("Verdict: CAUTION! Moderate similarity detected (possible heavy paraphrasing).")
else:
    print("Verdict: CLEAN! Low similarity detected.")
print("-" * 50)