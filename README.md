### Measuring how similar the dishes in different restaurants are based on their names and textual descriptions

### Used in the MealQR engineering project

* fasttext_setup.py creates a text embedding model trained on a locally-stored CSV dataset. The dataset consisted of dishes offered by restaurants in different cities (one city per CSV file). The information used for training consisted solely of the dish name and description, extracted from the CSVs into a single .txt file.<br>
The script can be used to create a model using the original Facebook-made fasttext library, or the Gensim implementation of it. For our project, we used the Facebook model and then compressed it using the compress-fasttext library. <br>

* Fasttext_Flask.py is a Flask server that can be queried for the similarities between dishes, provided their name and descriptions. It automatically strips non-relevant stopwords (calculated by frequency in the dataset and then manually picked from that) and converts the rest of the text into the format previously used for training by the model. <br>

* tsne_visualization.py is a script used to load the training file created by fasttext_setup.py and visualize how the different dishes are distributed in a high-dimensional vector space. If the model works as it should, different food types should generally form clusters of different colors (since FastText is designed so that similar texts are close together after embedding).<br>
t-SNE is used as the algorithm to transform the high-dimensional data to 2D points before plotting on a graph. Due to the nature of the method one should created multiple plots using different parameter values before coming to conclusions about the data. Read [this article](https://distill.pub/2016/misread-tsne/) for more information on this.<br>

To test the model after training it was applied to a portion of the data (labelled by the general type of food it was), and then graphed using t-SNE and matplotlib. The resulting graph can be seen below:

![t-SNE graph](https://github.com/AdrianKlessa/FastTextFoodSimilarity/blob/fd009f3a2241cff7c88ddbbf0fc921fb4c34c436/tsne.jpg)


It can be seen that in general similar food appears to be clustered together, showing that the model did learn useful representation during the learning.

Cosine similarity is used as a similarity measure after embedding - this measure is the standard for NLP purposes due to preventing issues related to comparing texts of different length.

Combined together, this can be used to recommend food to users by finding the dishes that are similar (close together after embedding) to the food that a user previously bought, liked or showed interest in. <br>
