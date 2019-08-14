

df = importData("data/Sentiment/data_latih.json")
df = cleaningDate(df, "tanggal")

print(df.head())
print("There are {} observations and {} features in this dataset. \n".format(
    df.shape[0], df.shape[1]))
print("There are {} users twitter in this dataset such as {} \n".format(len(df.id_user.unique()),
                                                                           [i for i in df.id_user.unique()[0:5]]))
print("There are {} clasess this dataset \n".format(
    len(df.sentimen.unique())))

user = df.groupby("id_user")
print(user.describe().head())
sentimen = df.groupby("sentimen")
print(sentimen.describe().head())
days =  df.groupby("tgl")
print(days.describe().head())

# drawPlot("Tweet Pengguna", user, "user id", "n tweet")
# drawPlot("Sentimen", sentimen, "kelas", "n tweet")
# drawPlot("Hari", days, "tanggal", "n tweet")
df_clean = removeUnicode(df)
df_clean = removeMention(df_clean)
df_clean = removeLink(df_clean)
df_clean = removeNumberAndSymbol(df_clean)
paragraph = bundlingTweet(df_clean)
# print(paragraph)
formalized = formalize(paragraph)
removed = removeStopwords(formalized)
print(removed)

stopword_list = openStopFile("./data/Sentiment/stopword_list_TALA.txt")

wordcloud = WordCloud(stopwords=stopword_list, background_color="white").generate(removed)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

