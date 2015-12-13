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

# Logging from gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
			# print dictionary
			# dc = DocCorpus(Docs(self.docsURL,cate_count,publisher_count), dictionary)
			# tfidf = models.TfidfModel(dc)
			# tfidf.save('./'+ str(cate_count)+ "_" + str(publisher_count) +'.tfidf_model')
			
			# index = Similarity(corpus=tfidf[dc], num_features=tfidf.num_nnz, output_prefix="str(cate_count)+ "_" + str(publisher_count)",num_best=50)
			# self.index.append(index)
			# index.save('./'+ str(cate_count)+ "_" + str(publisher_count) +'.sim')
			
			self.index.append(Similarity.load('./'+ str(cate_count)+ "_" + str(publisher_count) +'.sim'))

			publisher_count+=1
			count+=1
			if publisher_count == 3:
				publisher_count = 0
				cate_count+=1
	def getClassifiyPublisherName(self, queryDoc, category):
		words = pseg.cut(simplify(removeCharacter(queryDoc)))
		terms = []
		allowFlag = ['v','vd','vn','vshi','vyou','vf','vx','vi','vl','vg','a','ad','an','ag','al','d']
		for word, flag in words:
			if flag in allowFlag and len(word) > 1:
				terms.append(word)
				# print word, 
			# print('%s %s' % (word, flag))

		# terms = jieba.lcut(simplify(removeCharacter(queryDoc)), cut_all=True)
		# termsToRemove = []
		# for term in terms:
		# 	if len(term) < 2:
		# 		termsToRemove.append(term)
		# for term in termsToRemove:
		# 	terms.remove(term)
		# print terms

		results = []
		# print len(self.corporaDict)
		# print self.corporaDict[0].doc2bow([u'\u6e38\u4e50\u56ed'])
		for x in xrange(0, len(self.corporaDict)) :
			# print self.corporaDict[x].doc2bow(terms)
			if x >= category*len(publisherList) and x < category*len(publisherList)+ len(publisherList):
				result = self.index[x][self.corporaDict[x].doc2bow(terms)]
				if len(result) == 0:
					results.append(0)
				else:
					results.append(result[len(result)/2][1])
				# avg_score = []
				# for rank in result:
				# 	avg_score.append(rank[1])
				# if len(avg_score) == 0:
				# 	results.append(0)
				# else:
				# 	results.append(sum(avg_score) / float(len(avg_score)))
		return results

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
	
	# # Noted that there is no 。 due to split by this punc.
	# puncs = ['，', '?', '@', '!', '$', '%', '『', '』', '「', '」', '＼', '｜', '？', ' ', '*', '(', ')', '~', '.', '[', ']', '\n','1','2','3','4','5','6','7','8','9','0']

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
		# 			list.extend(jieba.cut(sentence, cut_all=True))
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
					results = corporaMgr.getClassifiyPublisherName(content, grocery.predict(content))
					outputfile.write(nameOfdata)
					outputfile.write(",")
					outputfile.write(numToPublisher(results.index(max(results))))
					outputfile.write("\n")

	else:
		print "Please use like 'python sim.py [originDocs_Dir] [outputDocs_Dir] [WithCatagoryAndPublisher] [train.csv]'"