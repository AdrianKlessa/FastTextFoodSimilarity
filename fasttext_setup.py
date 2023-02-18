import gensim.models.fasttext
from gensim.models import FastText
import pandas as pd
from numpy.linalg import norm
import numpy as np
import fasttext
import re
import compress_fasttext
from gensim.models.fasttext import load_facebook_model


def cosine_similarity(A, B):
    all_zeros = not (np.any(A) and np.any(B))
    if all_zeros:
        return 0.0
    return (np.dot(A, B) / (norm(A) * norm(B)))


# Stopwords found by looking for not-so-useful words in the dataset ranked by document frequency
# One word per line
stopwords = []
with open("stopwords.txt", 'r', encoding="utf-8") as stopwords_file:
    stopwords = [line.rstrip() for line in stopwords_file]


# Strips stopwords from the line
def process_line(text_line):
    return re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in stopwords),
                  "", text_line)


def create_model_gensim():
    dataset1 = pd.read_csv("dataset1.csv")
    dataset2 = pd.read_csv("dataset2.csv")
    dataset3 = pd.read_csv("dataset3.csv")

    dataset1['dish_description'].fillna('', inplace=True)
    dataset2['dish_description'].fillna('', inplace=True)
    dataset3['dish_description'].fillna('', inplace=True)

    cols = ['dish_name', 'dish_description']
    dataset1['combined'] = dataset1[cols].apply(lambda row: '|||'.join(row.values.astype(str)), axis=1)
    dataset2['combined'] = dataset2[cols].apply(lambda row: '|||'.join(row.values.astype(str)), axis=1)
    dataset3['combined'] = dataset3[cols].apply(lambda row: '|||'.join(row.values.astype(str)), axis=1)

    print(dataset2.head())

    dataset1['combined'] = dataset1['combined'].apply(lambda x: x[0:].split())
    dataset2['combined'] = dataset2['combined'].apply(lambda x: x[0:].split())
    dataset3['combined'] = dataset3['combined'].apply(lambda x: x[0:].split())

    print(dataset2.head())

    list1 = dataset1['combined'].tolist()
    list2 = dataset2['combined'].tolist()
    list3 = dataset3['combined'].tolist()

    final_list = list1
    final_list.extend(list3)
    final_list.extend(list2)
    print(final_list[:10])
    print("Starting training...")
    model = FastText(vector_size=200, window=5, min_count=1)
    model.build_vocab(corpus_iterable=final_list)
    model.train(final_list, total_examples=len(final_list),
                epochs=15)  # We have a relatively small dataset so we'll train longer than by default
    model.save("fasttext_trained")


def create_model_facebook():  # Separate description and title with |||
    dataset1 = pd.read_csv("dataset1.csv")
    dataset2 = pd.read_csv("dataset2.csv")
    dataset3 = pd.read_csv("dataset3.csv")

    dataset1['dish_description'].fillna('', inplace=True)
    dataset2['dish_description'].fillna('', inplace=True)
    dataset3['dish_description'].fillna('', inplace=True)
    cols = ['dish_name', 'dish_description']
    dataset1['combined'] = dataset1[cols].apply(lambda row: ' ||| '.join(row.values.astype(str)), axis=1)
    dataset2['combined'] = dataset2[cols].apply(lambda row: ' ||| '.join(row.values.astype(str)), axis=1)
    dataset3['combined'] = dataset3[cols].apply(lambda row: ' ||| '.join(row.values.astype(str)), axis=1)
    list1 = dataset1['combined'].tolist()
    list2 = dataset2['combined'].tolist()
    list3 = dataset3['combined'].tolist()
    final_list = list1
    final_list.extend(list3)
    final_list.extend(list2)
    training_file = open("training_file.txt", 'w', encoding="utf-8")
    training_file.write('\n'.join(final_list))
    model = fasttext.train_unsupervised('training_file.txt', model='skipgram')
    model.save_model("facebook_model.bin")


def create_model_facebook_v2():  # Separate description and title with |||, added some preprocessing to the dataset
    dataset1 = pd.read_csv("dataset1.csv")
    dataset2 = pd.read_csv("dataset2.csv")
    dataset3 = pd.read_csv("dataset3.csv")
    dataset1['dish_description'].fillna('', inplace=True)
    dataset2['dish_description'].fillna('', inplace=True)
    dataset3['dish_description'].fillna('', inplace=True)

    cols = ['dish_name', 'dish_description']
    dataset1['combined'] = dataset1[cols].apply(lambda row: ' ||| '.join(row.values.astype(str)), axis=1)
    dataset2['combined'] = dataset2[cols].apply(lambda row: ' ||| '.join(row.values.astype(str)), axis=1)
    dataset3['combined'] = dataset3[cols].apply(lambda row: ' ||| '.join(row.values.astype(str)), axis=1)

    dataset1.replace(r'\n', ' ', regex=True, inplace=True)
    dataset2.replace(r'\n', ' ', regex=True, inplace=True)
    dataset3.replace(r'\n', ' ', regex=True, inplace=True)
    dataset1['combined'] = dataset1['combined'].replace(('|'.join(r'\b%s\b' % re.escape(s) for s in stopwords)), "")
    dataset2['combined'] = dataset2['combined'].replace(('|'.join(r'\b%s\b' % re.escape(s) for s in stopwords)), "")
    dataset3['combined'] = dataset3['combined'].replace(('|'.join(r'\b%s\b' % re.escape(s) for s in stopwords)), "")

    list1 = dataset1['combined'].tolist()
    list2 = dataset2['combined'].tolist()
    list3 = dataset3['combined'].tolist()
    final_list = list1
    final_list.extend(list3)
    final_list.extend(list2)

    training_file = open("training_file.txt", 'w', encoding="utf-8")
    training_file.write('\n'.join(final_list))
    model = fasttext.train_unsupervised('training_file.txt', model='skipgram')
    model.save_model("facebook_model.bin")


def load_small_model():
    return compress_fasttext.models.CompressedFastTextKeyedVectors.load('small_model_v2')


def compress_model():
    big_model = load_facebook_model('facebook_model.bin').wv
    small_model = compress_fasttext.prune_ft_freq(big_model, pq=False)
    small_model.save('small_model_v2')


model = load_small_model()


def get_dish_similarity(dish1, dish2):
    dish1_stripped = process_line(dish1).split()
    dish2_stripped = process_line(dish2).split()
    return cosine_similarity(model.get_sentence_vector(dish1_stripped), model.get_sentence_vector(dish2_stripped))


def get_human_dish_similarity(dish1, dish2):
    similarity = get_dish_similarity(dish1, dish2)
    print("Similarity of: ", dish1, " AND ", dish2, "==== ", similarity)


def test_model():
    dish1 = "Sashimi with wasabi ||| "
    dish2 = "Salmon sushi with soy sauce |||"
    dish3 = "Ramen ||| "
    dish4 = "Rosół drobiowy ||| "
    dish5 = "Chicken BBQ with ribs ||| "

    print()
    print(model.get_sentence_vector(dish2))
    print(model.get_sentence_vector(dish3))

    vector1 = model.get_sentence_vector(dish1)
    vector2 = model.get_sentence_vector(dish2)
    vector3 = model.get_sentence_vector(dish3)
    vector4 = model.get_sentence_vector(dish4)
    vector5 = model.get_sentence_vector(dish5)

    print(cosine_similarity(vector1, vector2))
    print(cosine_similarity(vector2, vector3))
    print(cosine_similarity(vector3, vector4))
    print(cosine_similarity(vector1, vector2))
    print(cosine_similarity(vector1, vector5))

    get_human_dish_similarity("Japanese sushi", "Ramen")
    get_human_dish_similarity("Sushi", "Onigiri")
    get_human_dish_similarity("Onigiri z łososiem", "Zupa pomidorowa")
    get_human_dish_similarity("Pizza Americana", "Pizza Margherita")
    get_human_dish_similarity("Pizza Americana", "Sushi z łososiem")
    get_human_dish_similarity("Pizza rzeźnicka", "Pizza Americana")
    get_human_dish_similarity("Pizza rzeźnicka", "Ramen")
