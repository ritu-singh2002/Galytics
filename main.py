import gensim
from gensim.models import KeyedVectors


class WordEmbeddingProcessor:
    def __init__(self, vectors_file):
        self.vectors_file = vectors_file
        self.word_vectors = None

    def load_word_vectors(self):
        if not self.word_vectors:
            self.word_vectors = KeyedVectors.load_word2vec_format(self.vectors_file, binary=True, limit=1000000)

    def process_phrases(self, phrases):
        # Initialize word vectors if not loaded
        if not self.word_vectors:
            self.load_word_vectors()

        phrase_embeddings = []
        for phrase in phrases:
            words = phrase.split()
            # Calculate the phrase vector as the normalized sum of individual word vectors
            phrase_vector = sum(self.word_vectors.get_vector(word) for word in words)
            phrase_vector /= len(words)  # Normalize by the number of words
            phrase_embeddings.append(phrase_vector)

        return phrase_embeddings

    def calculate_similarity(self, phrase1, phrase2):
        # Calculate the cosine similarity between two phrase vectors
        if not self.word_vectors:
            self.load_word_vectors()

        vec1 = sum(self.word_vectors.get_vector(word) for word in phrase1.split())
        vec2 = sum(self.word_vectors.get_vector(word) for word in phrase2.split())
        similarity = self.word_vectors.cosine_similarities(vec1, [vec2])[0]

        return similarity


# Example usage
if __name__ == "__main__":
    vectors_file = "GoogleNews-vectors-negative300.bin"
    processor = WordEmbeddingProcessor(vectors_file)

    # Task I: Load word vectors and save as a flat file
    processor.load_word_vectors()
    processor.word_vectors.save_word2vec_format('vectors.csv')

    # Task II: Process data and calculate similarity
    phrases = ["how company compares to its peers?", "How does the forecasted insurance premium penetration in country trend compare to its peers?", "what is company general information?"]
    embeddings = processor.process_phrases(phrases)
    similarity = processor.calculate_similarity("how company compares to its peers?", "How does the forecasted insurance premium penetration in country trend compare to its peers?")

    # print(embeddings)
    print("Similarity between 'Phrase 1' and 'Phrase 2':", similarity)
