# -*- coding: UTF-8 -*-
import feedparser, logging
from tgrocery import Grocery
import opencc
import json
import sys, os, io
import xml.etree.ElementTree as ET
import codecs
from types import *
import multiprocessing as mp
from gensim import corpora, models
from gensim.similarities.docsim import Similarity
import jieba
import jieba.analyse
import jieba.posseg as pseg
from scipy import stats
from snownlp import sentiment
from snownlp import SnowNLP
import numpy as np
import matplotlib.pyplot as plt

# Logging from gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

########## INITIATION ##########
allowCategory = [ u'社會', u'娛樂', u'政治', u'生活']
publisherList = [ 'LTN', 'APD', 'CTS']

# jieba add_word
jieba.add_word('黄世铭')
jieba.add_word('特侦组')
jieba.add_word('黄色小鸭')
jieba.add_word('大统')
jieba.add_word('太阳花')
jieba.add_word('服贸协定')
jieba.add_word('服贸')
jieba.add_word('波卡')
jieba.add_word('台商')
jieba.add_word('北捷')
jieba.add_word('郑捷')
jieba.add_word('瓦斯')
jieba.add_word('气爆')

########## CLASS ##########
class EBCawler(object):
	def __init__(self):
		self.url = "http://news.ebc.net.tw/rss/"
		self.data = None
	def getCurrentXML(self):
		self.data = feedparser.parse("./download.xml")
	def getCount(self):
		return len(self.data['entries'])
	def getNewsTitleAt(self ,index):
		return self.data['entries'][index]['title']
	def getNewsContentAt(self, index):
		return self.data['entries'][index]['description']
	def getNewsCategortAt(self, index):
		return self.data['entries'][index]['subcategory']
	def getTranningData(self):
		list = []
		for data in self.data['entries']:
			if data['subcategory'] in allowCategory:
				list.append((allowCategory.index(data['subcategory']),simplify(data['title']+data['description'])))
		return list

class OriginDocs(object):
	def __init__(self, dirname, outputDir):
		self.dirname = dirname
		self.outputDir = outputDir
	def getOutputDir(self):
		return self.outputDir

	def simplifyAllDoc(self):
		if not os.path.exists(self.outputDir):
			os.makedirs(self.outputDir)
		
		pool = mp.Pool(processes=3)#mp.cpu_count()
		for x in os.listdir(self.dirname):
			if x == ".DS_Store":
				continue
			pool.apply_async(parseFile, args=(x,self.dirname,self.outputDir, ))
		pool.close()
		pool.join()

class CorporaMgr(object):
	def __init__(self, docsURL):
		self.docsURL = docsURL
		self.docs = Docs(docsURL)
		self.corporaDict = []
		self.index = []
	def generateCategoryAndPublisher(self):
		
		for x in xrange(0,len(allowCategory)*len(publisherList)):
			self.corporaDict.append(corpora.Dictionary())

		# for category, publisher, content in self.docs:
		# 	self.corporaDict[int(category)*len(publisherList)+publisherToNum(publisher)].add_documents([content])
		
		count = 0
		cate_count = 0
		publisher_count = 0

		for dictionary in self.corporaDict:
			# once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq <= 5]
			# dictionary.filter_tokens(once_ids)
			# dictionary.compactify()
			# dictionary.save('./'+ str(cate_count)+ "_" + str(publisher_count) +'.dict')
			self.corporaDict[count] = corpora.Dictionary.load('./'+ str(cate_count)+ "_" + str(publisher_count) +'.dict')
			
			# dc = DocCorpus(Docs(self.docsURL,cate_count,publisher_count), dictionary)
			# tfidf = models.TfidfModel(dc)
			# tfidf.save('./'+ str(cate_count)+ "_" + str(publisher_count) +'.tfidf_model')
			
			# index = Similarity(corpus=tfidf[dc], num_features=tfidf.num_nnz, output_prefix=(str(cate_count)+ "_" + str(publisher_count)),num_best=50)
			# self.index.append(index)
			# index.save('./'+ str(cate_count)+ "_" + str(publisher_count) +'.sim')
			self.index.append(Similarity.load('./'+ str(cate_count)+ "_" + str(publisher_count) +'.sim'))

			publisher_count+=1
			count+=1
			if publisher_count == 3:
				publisher_count = 0
				cate_count+=1
	def getClassifiyPublisherName(self, queryDoc, category, sentimentResult):
		words = pseg.cut(simplify(removeCharacter(queryDoc)))
		terms = []
		allowFlag = ['v','vd','vn','vshi','vyou','vf','vx','vi','vl','vg','a','ad','an','ag','al','d']
		for word, flag in words:
			if flag in allowFlag and len(word) > 1:
				terms.append(word)
				# print word, 
			# print('%s %s' % (word, flag))

		# totalDocs = 0
		# for x in xrange(0, len(self.index)) :
		# 	if x >= category*len(publisherList) and x < category*len(publisherList)+ len(publisherList):
		# 		totalDocs += len(self.index[x])

		# normalizer = []
		# for x in xrange(0, len(self.index)) :
		# 	if x >= category*len(publisherList) and x < category*len(publisherList)+ len(publisherList):
		# 		normalizer.append(totalDocs/float(len(self.index[x])))
		# print normalizer


		semti = 0
		count = 0
		for term in terms:
			s = SnowNLP(term)
			if s.sentiments == 0.5:
				continue
			semti += s.sentiments
			count += 1

		sentimentOfQuery = 0
		
		if count == 0:
			sentimentOfQuery = 0
		else:
			sentimentOfQuery = semti/float(count)
		

		resultsSemi = []
		resultsLike = []
		count = 0
		for x in xrange(0, len(self.corporaDict)) :
			# print self.corporaDict[x].doc2bow(terms)
			if x >= category*len(publisherList) and x < category*len(publisherList)+ len(publisherList):

				resultsSemi.append( 1 + sentimentResult[x].getPercentageOfValue(sentimentOfQuery))


				resultLikeFromFile = self.index[x][self.corporaDict[x].doc2bow(terms)]
				
				# === ALL RESULT ===
				# list = []
				# results.append(list)
				# for rank in result:
					# list.append(rank[1])

				
				# ==== MIDDLE ===
				# if len(result) == 0:
				# 	results.append(0)
				# else:
				# 	results.append(result[len(result)/2][1])# * normalizer[count]
				

				# ==== FIRST VALUE ===
				# if len(result) == 0:
				# 	resultsLike.append(0)
				# else:
				# 	resultsLike.append(result[0][1])

				# ==== AVG ===
				avg_score = []
				for rank in resultLikeFromFile:
					avg_score.append(rank[1])
				if len(avg_score) == 0:
					resultsLike.append(0)
				else:
					resultsLike.append(sum(avg_score) / float(len(avg_score)))
				# count+=1

		# minLength = 50
		# for result in results:
		# 	if len(result) == 0:
		# 		continue
		# 	if len(result) < minLength:
		# 		minLength = len(result)

		# finalOutput = []
		# count = 0
		# for result in results:
		# 	if len(result) == 0:
		# 		finalOutput.append(0)
		# 		count += 1
		# 		continue
		# 	else:
		# 		tmp = 0
		# 		for x in xrange(0, minLength):
		# 			tmp += result[x]
		# 		finalOutput.append(tmp)
		# 		count += 1
		
		# count = 0
		# for x in xrange(0, len(finalOutput)):
		# 	finalOutput[x] *= normalizer[x]

		# pValue = []
		# for x in xrange(0,len(results)):

		# 	if len(results[x]) == 0:
		# 		pValue.append(100)
		# 		continue

		# 	upper = x+1
		# 	if upper >= 3:
		# 		upper = 0

		# 	lower = x-1
		# 	if lower < 0:
		# 		lower = 2
		# 	pValue.append(stats.ttest_ind(results[x],results[upper])[1] + stats.ttest_ind(results[x],results[lower])[1])

		resultOfAll = []
		for x in xrange(0,len(resultsSemi)):
			resultOfAll.append( resultsSemi[x] * resultsLike[x] )

		print resultOfAll
		return numToPublisher(resultOfAll.index(max(resultOfAll)))

class DocCorpus(object):
	def __init__(self, texts, dict):
		self.texts = texts
		self.dict = dict

	def __iter__(self):
		for line in self.texts:
			yield self.dict.doc2bow(line[2])

class Docs(object):
	def __init__(self, dirname, categoryFilter = None, publisherFilter = None):
		self.dirname = dirname
		self.docList = os.listdir(self.dirname)
		self.categoryFilter = categoryFilter
		self.publisherFilter = publisherFilter

	def __iter__(self):
		for fname in self.docList:
			if fname == ".DS_Store":
				continue
			with open(os.path.join(self.dirname, fname)) as f:
				data = f.read().splitlines()
				if len(data) == 3:
					if self.categoryFilter == None and self.publisherFilter == None:
						category = data[0]
						# pop is to remove last , which cause the last character always empty
						content = data[1].split(',')[:-1]
						publisher = data[2]
						yield [category, publisher, content]
					else:
						category = data[0]
						publisher = data[2]
						if self.categoryFilter != int(category) or self.publisherFilter != publisherToNum(publisher):
							continue
						else:
							# pop is to remove last , which cause the last character always empty
							content = data[1].split(',')[:-1]
							publisher = data[2]
							yield [category, publisher, content]
				else:
					continue

class SentimentMgr(object):
	def __init__(self):
		# self.min = 1
		# self.max = 0
		# self.count = 0
		# self.sentiValue = 0
		self.values = []
	def addValue(self, sentiValue):
		self.values.append(sentiValue)
		# print self.values
		
		# if sentiValue > self.max:
		# 	self.max = sentiValue

		# if sentiValue < self.min:
		# 	self.min = sentiValue

		# self.count += 1
		# self.sentiValue += sentiValue
	def max(self):
		return np.amax(self.values)
	def min(self):
		return np.amin(self.values)
	def median(self):
		return np.median(self.values)
	def average(self):
		return np.average(self.values)
	def std(self):
		return np.std(self.values)
	def saveHistogram(self, fname):
		plt.clf()
		plt.hist(self.values, 100,range=[0, 1])
		plt.savefig(fname)
	def getPercentageOfValue(self, value):
		valuesInArray = np.array(self.values)
		return np.compress(((round(value,1)-0.1 <= valuesInArray) & (valuesInArray <= round(value,1)+0.1)),valuesInArray).size / float(len(self.values))


########## FUNCTIONS ##########

def removeCharacter(string):
	# Noted that there is no 。 due to split by this punc.
	puncs = ['，', '?', '@', '!', '$', '%', '『', '』', '「', '」', '＼', '｜', '？', ' ', '*', '(', ')', '~', '.', '[', ']', '\n','1','2','3','4','5','6','7','8','9','0']
	for punc in puncs:
		string = string.replace(punc,'')
	return string

def parseFile(filename, dirname, outputDir):	
	tree = ET.parse(os.path.join(dirname, filename))
	root = tree.getroot()

	title_Text = root[0].attrib['title']
	title_Text = title_Text.encode('utf-8')
	title_Text = removeCharacter(title_Text)
	# # remove useless puncs
	# for punc in puncs:
	# 	title_Text = title_Text.replace(punc,'')
	# Title
	title_Text = simplify(title_Text)

	content_text = None
	# Content
	if root[0][0].text != None:
		content_text = root[0][0].text
		content_text = content_text.encode('utf-8')

		for punc in puncs:
			content_text = content_text.replace(punc,'')

		content_text = simplify(content_text)
			
	# Writing Segment files
	with codecs.open(outputDir + "/" + os.path.basename(filename)[:-3] + "txt", "w", "utf-8") as f:
		f.write(title_Text)
		# Content : if there is no content then skip.
		if root[0][0].text != None:
			f.write(content_text)
		f.closed

def categoryToNum(argument):
    switcher = {
        u'社會':0,
        u'娛樂':1,
        u'政治':2,
        u'生活':3,
    }
    return switcher.get(argument)

def numToCategory(argument):
    switcher = {
        0:u'社會',
        1:u'娛樂',
        2:u'政治',
        3:u'生活',
    }
    return switcher.get(argument)

def publisherToNum(argument):
    switcher = {
        'LTN':0,
        'APD':1,
        'CTS':2,
    }
    return switcher.get(argument)

def numToPublisher(argument):
    switcher = {
        0:'LTN',
        1:'APD',
        2:'CTS',
    }
    return switcher.get(argument)

def simplify(text):
 return opencc.convert(text, config='t2s.json')

def traditionalize(text):
 return opencc.convert(text, config='zhs2zht.ini').encode('utf-8')

if __name__ == '__main__':
	
	if len(sys.argv) == 6:
		# ======= Trainning News Category =======
		# Depends on current news classifier
		ebCawler = EBCawler()
		# Warning!!! - Need web connection
		ebCawler.getCurrentXML()

		# for data in ebCawler.getTranningData():
		# print len(ebCawler.getTranningData())
		grocery = Grocery('sample')
		grocery.train(ebCawler.getTranningData())

		# ======= Simplify All Original Docs =======
		originDocs_Dir = sys.argv[1]
		outputDocs_Dir = sys.argv[2]
		outputWithCateDocs_Dir = sys.argv[3]
		tranningCSV = sys.argv[4]
		queryDir = sys.argv[5]
		originDocs = OriginDocs(originDocs_Dir, outputDocs_Dir)
		# originDocs.simplifyAllDoc();

		# ======= Category Original Docs and save to json file =======
		# for x in os.listdir(outputDocs_Dir):
		# 	if x == ".DS_Store":
		# 		continue
		# 	content = None
		# 	with open(outputDocs_Dir + "/" + x) as f:
		# 		content = f.readline()
		# 	list = []
		# 	content_sentences = content.split('。')
		# 	for sentence in content_sentences:
		# 		if len(sentence) != 0:
		# 			data = []
		# 			allowFlag = ['v','vd','vn','vshi','vyou','vf','vx','vi','vl','vg','a','ad','an','ag','al','d']
		# 			words = pseg.cut(sentence)
		# 			for word, flag in words:
		# 				if flag in allowFlag and len(word) > 1:
		# 					list.append(word)
		# 			# list.extend(jieba.cut(sentence, cut_all=True))
		# 	result = grocery.predict(content)
		# 	print x
		# 	with codecs.open(outputWithCateDocs_Dir + "/" + x, "w", "utf-8") as f:
		# 		f.write(str(result))
		# 		f.write("\n")
		# 		for term in list:
		# 			# if the term word is less than 2, which is a single word, then ignore it
		# 			if len(term) < 2:
		# 				continue
		# 			else:
		# 				f.write(term)
		# 				f.write(',')
		
		# ======= Add Publisher to All docs =======
		# TODO : Should combine with the previous step which will decrease computation time.
		# with open(tranningCSV) as f:
		# 	next(f)
		# 	for line in f:
		# 		filename, publisher = line.rstrip().split(',')
		# 		with open( outputWithCateDocs_Dir+ "/" + filename +".txt", "a") as appendfile:
		# 		    appendfile.write("\n")
		# 		    appendfile.write(publisher)
		# 		print filename


		# ======= Train 4 Category with 3 Publisher(LTN,CTS,APD) =======
		corporaMgr = CorporaMgr(outputWithCateDocs_Dir)
		corporaMgr.generateCategoryAndPublisher()


		# sentiments_FromSet = []
		# SENTIMENT TRAINNING FROM NTUSD
		# sentiment.train('neg.txt', 'pos.txt')
		# sentiment.save('sentiment.marshal')

		sentimentResult = []

		# ====== Sentiments ======
		for cate in xrange(0, len(allowCategory)):
			print "cate", cate
			for publisher in xrange(0, len(publisherList)):
				sentiMgr = SentimentMgr()
				sentimentResult.append(sentiMgr)
				print "publisher", publisher
				
				for doc in Docs(outputWithCateDocs_Dir,cate,publisher):
					count = 0
					semti = 0.0	
					if len(doc[2]) == 0:
						continue
					for term in doc[2]:
						s = SnowNLP(unicode(term, 'utf-8'))
						if s.sentiments == 0.5:
							continue
						semti += s.sentiments
						count += 1
					if count == 0:
						continue
					else:
						sentiMgr.addValue(semti/float(count))
						# print "median: ",sentiMgr.median()
						# , "min: ", sentiMgr.min, "avg:", sentiMgr.avgValue(), "-", semti/float(count)

					# if count_inDoc == 0:
						# continue
					# else:
						# print senti_inDoc,count_inDoc
						# semti += float(senti_inDoc)/float(count_inDoc)
						# count += 1
				print "================="
				print "===max: ",sentiMgr.max(), "min: ", sentiMgr.min(), "avg:", sentiMgr.average()
				sentiMgr.saveHistogram('current.png')
				# break
				# sentiments_FromSet.append(sentiMgr.avgValue())

		publisher_count = 0
		count = 0
		cate_count = 0

		for result in sentimentResult:
			result.saveHistogram(str(cate_count) + "_" + str(publisher_count) + ".png")
			publisher_count+=1
			count+=1
			if publisher_count == 3:
				publisher_count = 0
				cate_count+=1
			print result.max(), result.min(), result.median(), result.average(), result.std()
			# 0.935333489596 0.00875296241739 0.319083118846 0.326583518388 0.0610376482466
			# 0.784000052087 0.0014028422927 0.321060214695 0.328754298799 0.0609401440388
			# 0.878522504892 0.0615563217165 0.324475004439 0.330887137374 0.0555331749459
			# 
			# 0.935333489596 0.0106472864747 0.3697758035 0.379513786538 0.098874425074
			# 0.812518360162 0.0300867666187 0.375512839139 0.379377350102 0.0912036821854
			# 0.669268342313 0.147583789222 0.367846773649 0.373351085225 0.0677476942484

			# 0.915597328212 0.0622895622896 0.313632581676 0.321264363939 0.0557029996915
			# 0.666666666667 0.0582134937822 0.320980445391 0.327295822889 0.057736354674
			# 0.568727372665 0.168654524941 0.320292885567 0.326198792206 0.0507511380966

			# 0.75 0.0568408790917 0.347004991076 0.354779423899 0.0674763535299
			# 0.792439372976 0.0588539620499 0.353527820966 0.360996346848 0.0713863655732
			# 0.823173454013 0.15818249969 0.344966478066 0.353326906681 0.0615560359763
			
		# print sentiments_FromSet

		# allowCategory = [ u'社會', u'娛樂', u'政治', u'生活']
		# publisherList = [ 'LTN', 'APD', 'CTS']
		# 0.319, 0.3178, 0.327
		# 0.367, 0.38
		# [0.32658351838803207, 0.3287542987987323,  0.33088713737381503,
		#  0.37951378653834134, 0.37937735010195844, 0.3733510852251421,
		#  
		#  
		#  0.3212643639392646,  0.32729582288912906, 0.32619879220636183,
		#  0.39268
		#  
		#  0.35477942389873096, 0.36099634684839993, 0.35332690668123784]
		# sentiMgr = SentimentMgr()
		# for fname in os.listdir(queryDir):
		# 	if fname == ".DS_Store":
		# 		continue
		# 	with open(os.path.join(queryDir, fname)) as f:
		# 		data = f.read().splitlines()
		# 		nameOfdata = data[0]
		# 		# print nameOfdata
		# 		content = ""
		# 		for x in xrange(2,len(data)):
		# 			# print data[x]
		# 			content += data[x]
				
		# 		s_total = 0
		# 		count = 0
		# 		for term in jieba.lcut(content):
		# 			s = SnowNLP(term)
		# 			if s.sentiments == 0.5:
		# 				continue
		# 			# print term,s.sentiments
		# 		if grocery.predict(simplify(content)) == 2:
		# 			s_total += s.sentiments
		# 			count += 1
		# 			sentiMgr.addValue(s_total/float(count))
				# print simplify(content)
				# print nameOfdata,s.sentiments


		# ======= Predict And result.csv =======
		with open( "result.csv", "w") as outputfile:
			outputfile.write("NewsId,Agency")
			outputfile.write("\n")
			for fname in os.listdir(queryDir):
				if fname == ".DS_Store":
					continue
				with open(os.path.join(queryDir, fname)) as f:
					data = f.read().splitlines()
					nameOfdata = data[0]
					print nameOfdata
					content = ""
					for x in xrange(2,len(data)):
						# print data[x]
						content += data[x]
					# print content
					result = 0
					# if grocery.predict(simplify(content)) == 2:
						
					# else:
					result = corporaMgr.getClassifiyPublisherName(content, grocery.predict(simplify(content)),sentimentResult)
					outputfile.write(nameOfdata)
					outputfile.write(",")
					outputfile.write(result)
					outputfile.write("\n")

	else:
		print "Please use like 'python sim.py [originDocs_Dir] [outputDocs_Dir] [WithCatagoryAndPublisher] [train.csv]'"
	# python categorydocs.py news_story_dataset/ preprocessingData/simplify/ preprocessingData/withCategory/ train.csv p2data/phase2_test_dataset/