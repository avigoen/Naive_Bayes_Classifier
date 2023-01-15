import math
import re

STOPWORDS=['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 
        'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours',
        'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
        'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 
        'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 
        'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 
        'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why',
        'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 
        'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 
        'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

MOVIE_DATA_REGEX=r"(?P<rating>\d{1})\|(?P<movie_id>\d*)\|(?P<comment>.*)"
WORD_STEMMING_REGEX=r'less|ship|ing|les|ly|es|s'
STRING_WITHOUT_PUNCTUATION_REGEX=r'[^\w\s]'

class Bayes_Classifier:

    def __init__(self):
        self.total_dataset_count = 0
        self.rating_words_count = {'positive': 0, 'negative': 0}
        self.bag_of_words_overall = {}

    def cleanup_comment(self, comment):
        words_in_comments = [word.lower() for word in comment.split(" ")]
        words_in_comments = [re.sub(STRING_WITHOUT_PUNCTUATION_REGEX, '', word) for word in words_in_comments] # remove punction from word
        words_in_comments = [re.sub(WORD_STEMMING_REGEX, '', word) for word in words_in_comments if word not in STOPWORDS] # remove stopwords and update words with stemmming
        return words_in_comments

    def process_entry_value(self, key, value):
        val = value
        if key != "comment":
            val = int(value)
        if key == 'rating':
            val = "positive" if val == 5 else "negative"
        if key == 'comment':
            val = self.cleanup_comment(value)
        return val

    def read_data(self, lines):
        lines_processed = []
        for line in lines:
            grouped_match = re.match(MOVIE_DATA_REGEX, line)
            processed_data_types = {}
            for key, value in grouped_match.groupdict().items():
                value = self.process_entry_value(key, value)
                processed_data_types[key] = value
            lines_processed.append(processed_data_types)
        return lines_processed

    def separate_by_rating(self, dataset):
        ratings_segregated = dict()
        for user_rating in dataset:
            rating = user_rating['rating']
            if rating not in ratings_segregated:
                ratings_segregated[rating] = list()
            ratings_segregated[rating].append(user_rating)
        return ratings_segregated

    def rating_wise_bag_of_words(self, rating_entries):
        bow = {}
        for entry in rating_entries:
            for word in entry['comment']:
                if word not in bow:
                    bow[word] = 0
                bow[word] = bow[word] + 1
        return bow

    def create_bag_of_words(self, rating_segregated):
        rating_bow_dict = dict()
        for rating in rating_segregated:
            rating_bow = self.rating_wise_bag_of_words(rating_segregated[rating])
            rating_bow_dict[rating] = rating_bow
        return rating_bow_dict    

    def overall_bow(self, rating_segregated_bow):
        positive_bow = rating_segregated_bow['positive']
        negative_bow = rating_segregated_bow['negative']

        for word in positive_bow:
            self.rating_words_count['positive'] = self.rating_words_count['positive'] + positive_bow[word] + 1
            if word not in self.bag_of_words_overall:
                self.bag_of_words_overall[word] = {'positive': 1, 'negative': 1}
            self.bag_of_words_overall[word]['positive'] = positive_bow[word] + 1

        for word in negative_bow:
            self.rating_words_count['negative'] = self.rating_words_count['negative'] + negative_bow[word] + 1
            if word in self.bag_of_words_overall:
                self.bag_of_words_overall[word]['negative'] = negative_bow[word] + 1
                continue
            self.bag_of_words_overall[word] = {'positive': 1, 'negative': negative_bow[word] + 1}


    def train(self, lines):
        processed_train_data = self.read_data(lines)
        self.total_dataset_count = len(processed_train_data)

        rating_segregated = self.separate_by_rating(processed_train_data)
        rating_segregated_bow = self.create_bag_of_words(rating_segregated)

        self.overall_bow(rating_segregated_bow)

    def calculate_sentance_probability(self, word, rating, prev_probability):
        def calculate_prob(word, rating):
            numerator = self.bag_of_words_overall[word][rating] + 1
            denominator = len(self.bag_of_words_overall) + self.rating_words_count[rating]
            return math.log(numerator/denominator)

        def calculate_prob_false(rating):
            numerator = 1
            denominator = len(self.bag_of_words_overall) + self.rating_words_count[rating]
            return math.log(numerator/denominator)

        if word in self.bag_of_words_overall and self.bag_of_words_overall[word][rating]:
            return prev_probability + calculate_prob(word, rating)
        return prev_probability + calculate_prob_false(rating)


    def classify(self, lines):
        proccessed_test_data = self.read_data(lines)

        ratings = []
        for review in proccessed_test_data:
            pos_review_prob = 0
            neg_review_prob = 0
            for word in review['comment']:
                pos_review_prob = self.calculate_sentance_probability(word, 'positive', pos_review_prob)
                neg_review_prob = self.calculate_sentance_probability(word, 'negative', neg_review_prob)
            
            if pos_review_prob > neg_review_prob:
                ratings.append('5')
            else:
                ratings.append('1')

        return ratings