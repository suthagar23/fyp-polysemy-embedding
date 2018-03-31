from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans,DBSCAN
import collections
import nltk
import numpy as np
from sklearn.preprocessing import StandardScaler
from nltk.cluster import KMeansClusterer
import plotly.offline as py
import plotly.graph_objs as go




# sentences=[]
# # st=""
# # stline=True
# lines=open("result_text_address_after_iteration_2").read()
# i=1
# for line in lines.split(".\n"):
# 	words=line.split(" ")
# 	for num in range(0,len(words)):
# 		if (words[num].startswith("address_")):
# # 			print(words[num])
# # 			words[num]="address"+"_"+str(i)
# 			i+=1
# # 	st+=(" ".join(words))
# # 	st+=".\n"
# 	sentences.append(words)
# # f=open("result_text_address_before_iteration_1","w")
# # f.write(st)
# # f.close()
# model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
# print(i)
# model.save('model3_add_tri.bin')
model=Word2Vec.load('model1_add_tri.bin')
#
#
# # words = list(model.wv.vocab)
# #
# #
# # X = model[model.wv.vocab]
# # pca = PCA(n_components=2)government_NN
# # result = pca.fit_transform(X)
# # pyplot.scatter(result[:, 0], result[:, 1])
# # for i, word in enumerate(words):
# # 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# # pyplot.savefig("gensim.png")
# # pyplot.show()
# # print(model["government_NN_2"])
vecs=[]
words=[]
for word in model.wv.vocab:
	if "address_" in word:
		words.append(word)
		vecs.append(model[word])
# # pca = PCA(n_components=2)
# # result = pca.fit_transform(vecs)
# # plt.scatter(result[:, 0], result[:, 1])
# # plt.savefig("gensim-2000.png")
# # plt.close()
#
#
# # kmeans = KMeans(n_clusters=1)
# # kmeans.fit(result)
# # y=kmeans.predict(result)
# # print (y)
# #
# #
# # centroids = kmeans.cluster_centers_
# # centers = kmeans.cluster_centers_
# # labels = kmeans.labels_
# # plt.scatter(X[:,0],X[:,1])
# # print(centroids)
# # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
# # print(labels)
# # plt.savefig("gensim-100.png")
# # plt.show()
# #
#
#
# # NUM_CLUSTERS=3
# # kclusterer = KMeans(NUM_CLUSTERS)
# # assigned_clusters = kclusterer.fit(X)
# # print (assigned_clusters)
#
#
#
# # word2vec_dict = {}
# # words = model.wv.index2word  # order from model.wv.syn0
# #
# # for i in words:
# #     word2vec_dict[i] = model[i]
# #
# # X = np.array([word2vec_dict[i].T for i in words])
# #
# # kmeans = KMeans(10)
# # kmeans.fit_predict(X)
# # labels = kmeans.labels_
# # vocab = list(model.wv.vocab)
# # centroids=kmeans.cluster_centers_
# # print(centroids)
# # clusters = [list(a) for a in zip(vocab, labels)]
# #
# # pca = PCA(n_components=2)
# # result = pca.fit_transform(X)
# # plt.scatter(result[:, 0], result[:, 1])
# # plt.scatter(centroids[:, 0], centroids[:, 1], c='black');
# # plt.show()
#
#
# words=[]
# X=[]
# for t in model.wv.vocab:
#     if "government_NN" in t:
#         words.append(t)
#         X.append(model[t])
# NUM_CLUSTERS=1
# print(2)
# kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
# assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
# print(3)
# print (assigned_clusters)
# print(4)
# aftermodel1={}
# strg=""
# for i, word in enumerate(words):
#     aftermodel1[word]= str(assigned_clusters[i])
#     strg+=(word + ":" + str(assigned_clusters[i])+"\n")
# print(5)
# f=open("afteriteration7.txt","w")
# f.write(strg)
#
#
#
# text=open("polysemy_after_iterationN+1.txt").read();
# strg=""
# for sentence in text.split(".\n"):
#     for word in sentence.split(" "):
#         if("c" in word):
#             if(word in aftermodel1.keys()):
#                 print(word,"   ======>      "),
#                 word="government_NN_"+aftermodel1[word]
#                 print(word)
#         strg+=" "+word
#     strg+=".\n"
# f=open("polysemy_after_iteration7.txt","w")
# f.write(strg)
# f.close()
#
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
#
#
# # plt.scatter(result[:, 0], result[:, 1])
# # plt.savefig("iteration7.png")
# # plt.show()
#
#
# print("==============================================================")
# print(model.similar_by_word("government_NN_0"))



def matplotlib_to_plotly(cmap, pl_entries):
	h = 1.0 / (pl_entries - 1)
	pl_colorscale = []

	for k in range(pl_entries):
		Ctemp = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
		C = list(map(int, Ctemp))
		pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

	return pl_colorscale

new_vectors = StandardScaler().fit_transform(vecs)
for eps in np.arange(1,2,1):
  print (eps)
  new_db = DBSCAN(eps=7.5, min_samples=1).fit(new_vectors)
  labels = new_db.labels_
  # print (labels)
  core_samples_mask = np.zeros_like(new_db.labels_, dtype=bool)
  core_samples_mask[new_db.core_sample_indices_] = True
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  counter=collections.Counter(labels)
  print (counter)
  print (len(counter))
  unique_labels = set(labels)
  if(len(unique_labels)>1):
	  colors = matplotlib_to_plotly(plt.cm.Spectral, len(unique_labels))
	  data = []

	  for k, col in zip(unique_labels, colors):

		  if k == -1:
			  # Black used for noise.
			  col = 'black'
		  else:
			  col = col[1]

		  class_member_mask = (labels == k)

		  xy = new_vectors[class_member_mask & core_samples_mask]
		  trace1 = go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='markers',
							  marker=dict(color=col, size=14,
										  line=dict(color='black', width=1)))

		  xy = new_vectors[class_member_mask & ~core_samples_mask]
		  trace2 = go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='markers',
							  marker=dict(color=col, size=14,
										  line=dict(color='black', width=1)))
		  data.append(trace1)
		  data.append(trace2)

	  layout = go.Layout(showlegend=False,
						 title='Estimated number of clusters: %d' % n_clusters_,
						 xaxis=dict(showgrid=False, zeroline=False),
						 yaxis=dict(showgrid=False, zeroline=False))
	  fig = go.Figure(data=data, layout=layout)

	  py.plot(fig)
#
result={}

for j in range(0,len(labels)):
	result["address_"+str(j)]=labels[j]

lines=open("result_text_address_after_iteration_2").read().split(".\n")
st=""
k=1
stline=True
for line in lines:
	for word in line.split(" "):
		if word.startswith("address"):
			if(word in result.keys()):
				word="address_"+str(result[word])
			else:
				word="address_"+str(len(unique_labels)+k)
				k+=1
		if(stline):
			st+=word
			stline=False
		else:
			st+=(" "+word)
	st+=",\n"
	stline=True

file=open("result_text_address_after_iteration_3","w")
file.write(st)
file.close()


pca = PCA(n_components=2)
result = pca.fit_transform(vecs)
plt.scatter(result[:, 0], result[:, 1])
plt.savefig("gensim-2000.png")
plt.close()


kmeans = KMeans(n_clusters=1)
kmeans.fit(result)
y=kmeans.predict(result)
print (y)


centroids = kmeans.cluster_centers_
centers = kmeans.cluster_centers_
labels = kmeans.labels_