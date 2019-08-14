import numpy as np
import pandas as pd
from os import path
import PIL
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

df = pd.read_csv("data/winemag-data-130k-v2.csv", index_col=0)
df.head()
print("There are {} observations and {} features in this dataset. \n".format(
    df.shape[0], df.shape[1]))
print("There are {} types of wine in this dataset such as {}... \n".format(len(df.variety.unique()),
                                                                           ", ".join(df.variety.unique()[0:5])))
print("There are {} countries producing wine in this dataset such as {}... \n".format(
    len(df.country.unique()),  ", ".join(df.country.unique()[0:5])))

df[["country", "description","points"]].head()
# Groupby by country
country = df.groupby("country")

# Summary statistic of all countries
country.describe().head()

country.mean().sort_values(by="points",ascending=False).head()
plt.figure(figsize=(15,10))
country.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Number of Wines")
# plt.show()

plt.figure(figsize=(15,10))
country.max().sort_values(by="points",ascending=False)["points"].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Highest point of Wines")
# plt.show()

# Start with one review:
text = df.description[0]
print("===========================================================")
print(text)
# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# plt.show()

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.show()

wordcloud.to_file("img/first_review.png")

text = " ".join(review for review in df.description)
print ("There are {} words in the combination of all review.".format(len(text)))

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# plt.show()


# Generate a word cloud image
mask = np.array(PIL.Image.open("./img/gorilla.png"))
wordcloud_usa = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA", max_words=1000, mask=mask).generate(text)

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[7,7])
plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# store to file
plt.savefig("img/gorilla_word_cloud.png", format="png")

plt.show()

# alice_coloring = np.array(PIL.Image.open("./img/goku.png"))

# image_colors = ImageColorGenerator(alice_coloring)

# # wc = WordCloud(background_color="white", max_words=1000, mask=alice_coloring,
# #                stopwords=stopwords, contour_width=3, contour_color='firebrick')
# wc = WordCloud(background_color="white", max_words=2000, mask=alice_coloring,
#                stopwords=stopwords, max_font_size=40, random_state=42)
# # Generate a wordcloud
# wc.generate(text)
# # show
# fig, axes = plt.subplots(1, 3)
# axes[0].imshow(wc, interpolation="bilinear")
# # recolor wordcloud and show
# # we could also give color_func=image_colors directly in the constructor
# axes[1].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# axes[2].imshow(alice_coloring, cmap=plt.cm.gray, interpolation="bilinear")
# for ax in axes:
#     ax.set_axis_off()
# plt.show()

# def transform_format(val):
#     print(val)
#     if val == 0:
#         return 255
#     else:
#         return val

# # Transform your mask into a new one that will work with the function:
# transformed_wine_mask = np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)

# for i in range(len(wine_mask)):
#     transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))

# # Create a word cloud image
# wc = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask,
#                stopwords=stopwords, contour_width=3, contour_color='firebrick')

# # Generate a wordcloud
# wc.generate(text)

# # store to file
# wc.to_file("./img/goku_word_cloud.png")

# # show
# plt.figure(figsize=[20,10])
# plt.imshow(wc, interpolation='bilinear')
# plt.axis("off")
# plt.show()