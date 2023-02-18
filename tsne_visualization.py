import main
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = main.load_small_model()
file = open("training_file.txt", "r", encoding="utf-8")
perplexities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
texts = []
categories = []
vectors = []
coords_x = []
coords_y = []

for line in file.readlines():
    category = ""
    line = line.lower()
    if "rib" in line or "żebe" in line:
        category = "ribs"
    elif "pizza" in line:
        category = "pizza"
    elif "soup" in line or "stew" in line or "zupa" in line:
        category = "soup"
    elif "onigiri" in line or "sashimi" in line or "sushi" in line:
        category = "sushi"
    elif "korea" in line:
        category = "korean"
    elif "sałatka" in line or "salad" in line:
        category = "salad"
    elif "pad thai" in line:
        category = "pad thai"
    elif "wrap" in line:
        category = "wrap"
    elif "mochi" in line:
        category = "mochi"
    elif "pancake" in line or "naleśn" in line:
        category = "pancake"
    elif "tofu" in line:
        category = "tofu"

    if category != "":
        vector = model.get_sentence_vector(line.split())
        if (np.isnan(vector).any()):
            print("Contained nan")
        else:
            texts.append(line)
            categories.append(category)
            vectors.append(vector)
file.close()
vectors_numpy = np.array(vectors)
print("Length of the test dataset: ", len(texts))
color_dict = {'ribs': 'red', 'pizza': 'blue', 'soup': 'black', 'sushi': 'green', 'korean': 'purple', 'salad': 'silver',
              'pad thai': 'peru', 'wrap': 'darkslategrey', 'mochi': 'yellow', 'pancake': 'fuchsia',
              'tofu': 'firebrick'}
for perp in perplexities:
    tsne = TSNE(perplexity=perp)
    vectors_tsne = tsne.fit_transform(vectors_numpy)
    coords_x = vectors_tsne[:, 0]
    coords_y = vectors_tsne[:, 1]
    fig = plt.figure(figsize=(16, 12), dpi=200)
    plt.scatter(coords_x, coords_y, color=[color_dict[i] for i in categories], s=4.0, alpha=0.5)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)
    plt.title("Distribution of restaurant food as seen by Fasttext, perplexity: "+str(perp))
    filename = "tsne (perplexity " + str(perp) + ").jpg"
    fig.savefig(filename, dpi=200)
plt.show()
