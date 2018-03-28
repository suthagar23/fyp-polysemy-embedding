from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

text=(open("polysemy.txt").read()).split("\n")
vetorize =TfidfVectorizer(stop_words='english')
X=vetorize.fit_transform(text)

model=KMeans(n_clusters=10)
model=model.fit(X)


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vetorize.get_feature_names()
for i in range(10):
    print ("Cluster %d:" % i),
    for ind in order_centroids[i]:
        print (' %s' % terms[ind]),
    print()
