from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def extract_negative_keywords(text):
    words = text.split()
    negative_words = []

    for word in words:
        score = sia.polarity_scores(word)['compound']
        if score < -0.3:
            negative_words.append(word)

    overall_sentiment = sia.polarity_scores(text)['compound']
    return {
        'overall_sentiment': overall_sentiment,
        'negative_keywords': negative_words
    } # dict



# student_input = "I am really lost on integrals"
# result = extract_negative_keywords(student_input)

# print("Overall Sentiment Score:", result['overall_sentiment'])
# print("Negative Keywords:", result['negative_keywords'])
# print(len(result['negative_keywords']))
