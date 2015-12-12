# -*- coding: UTF-8 -*-
import numpy, os, logging
from gensim import similarities, corpora, models
from gensim.similarities.docsim import Similarity
import jieba
import jieba.analyse
import xml.etree.ElementTree as ET
import opencc
import sys
import codecs
import multiprocessing as mp
from types import *
from datetime import date
from dateutil.relativedelta import relativedelta

# Logging from gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

########## CLASSES ##########
class DocCorpus(object):
	def __init__(self, texts, dict):
		self.texts = texts
		self.dict = dict

	def __iter__(self):
		for line in self.texts:
			yield self.dict.doc2bow(line)

class OriginDocs(object):
	def __init__(self, dirname, outputDir):
		self.dirname = dirname
		self.outputDir = outputDir
	def getOutputDir(self):
		return self.outputDir

	def simplifyAllDoc(self):
		if not os.path.exists(self.outputDir):
			os.makedirs(self.outputDir)
		
		pool = mp.Pool(processes=mp.cpu_count())
		for x in os.listdir(self.dirname):
			if x == ".DS_Store":
				continue
			pool.apply_async(parseFile, args=(x,self.dirname,self.outputDir))
		pool.close()
		pool.join()
		

# Docs 
class Docs(object):
	def __init__(self, dirname, querys=None):
		self.dirname = dirname
		self.querys = querys
		self.docList = os.listdir(self.dirname)
		# self.docList.remove(".DS_Store")

	def __iter__(self):
		for fname in self.docList:
			with open(os.path.join(self.dirname, fname)) as f:
				s = f.read()
				yield s.split(',')
		if self.querys != None:
			for query in self.querys:
				yield query
	def getDocCount(self):
		return len(self.docList)
	def getDocNameByID(self, docId):
		return self.docList[docId]

class QueryList(object):
	# """docstring for QueryList"""
	def __init__(self, url):
		self.url = url
	def getQuerys(self):
		tree = ET.parse(self.url)
		root = tree.getroot()

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

		querys = []
		for query in root:
			# print "ok"
			queryDoc = []
			querys.append(queryDoc)
			for doc in query:
				if type(doc) is not ListType:
					for term in jieba.cut(simplify(doc.attrib['title'].rstrip('\n')), cut_all=True):
						queryDoc.append(term)
					if doc[0].text != None:
						for term in jieba.cut(simplify(doc[0].text.rstrip('\n')), cut_all=True):
							queryDoc.append(term)
		return querys
########## FUNCTIONS ##########
def simplify(text):
 return opencc.convert(text, config='t2s.json')

def traditionalize(text):
 return opencc.convert(text, config='zhs2zht.ini').encode('utf-8')

def parseFile(filename, dirname, outputDir):

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
		
		tree = ET.parse(os.path.join(dirname, filename))
		root = tree.getroot()
		
		# Noted that there is no 。 due to split by this punc.
		puncs = ['，', '?', '@', '!', '$', '%', '『', '』', '「', '」', '＼', '｜', '？', ' ', '*', '(', ')', '~', '.', '[', ']', '\n','1','2','3','4','5','6','7','8','9','0']

		title_Text = root[0].attrib['title']
		title_Text = title_Text.encode('utf-8')
		# remove useless puncs
		for punc in puncs:
			title_Text = title_Text.replace(punc,'')

		title_Text_sentences = title_Text.split('。')

		# Title
		title_cuts = []
		for sentence in title_Text_sentences:
			title_cuts.extend(jieba.cut(simplify(sentence), cut_all=True))	
		
		# Content
		seg_list = None
		if root[0][0].text != None:
			content_text = root[0][0].text
			content_text = content_text.encode('utf-8')

			for punc in puncs:
				content_text = content_text.replace(punc,'')
			content_text_sentences = content_text.split('。')
			seg_list = []
			for sentence in content_text_sentences:
				seg_list.extend(jieba.cut(simplify(sentence), cut_all=True))
				
		# Writing Segment files
		with codecs.open(outputDir + "/" + os.path.basename(filename), "w", "utf-8") as f:
			# Get news title.
			for title in title_cuts:
				if title != '':
					f.write(title)
					f.write(',')
			# Content : if there is no content then skip.
			if root[0][0].text != None:
				for seg in seg_list:
					if seg !='':
						f.write(seg)
						f.write(',')
			f.closed

class TimeManager(object):
	def __init__(self):
		self.time = []
		self.count = []
	def addTime(self, time):
		timeFormat = time[:-3]
		if timeFormat in self.time:
			self.count[self.time.index(timeFormat)]+=1
		else :
			self.time.append(timeFormat)
			self.count.append(0)
	def maxAmountMonths(self):
		
		returnMonths = []
		chooseMonth = self.time[self.count.index(max(self.count))]
		date_format = date(int(chooseMonth.split('-')[0]),int(chooseMonth.split('-')[1]),1)
		
		for x in xrange(-2,6):
			date_tmp = date_format + relativedelta(months=x)
			returnMonths.append(date_tmp.isoformat()[:-3])
		return returnMonths
		
if __name__ == '__main__':

	if len(sys.argv) == 5:
		originDocs_Dir = sys.argv[1]
		outputDocs_Dir = sys.argv[2]
		queryFile = sys.argv[3]
		resultFileName = sys.argv[4]
		
		print "->>Processing Origin Files"
		# Origin Docs
		originDocs = OriginDocs(originDocs_Dir, outputDocs_Dir)
		originDocs.simplifyAllDoc();
		print "->>Load QueryFile"
		# Load QueryFile
		queryObject = QueryList(queryFile)
		querys = queryObject.getQuerys()

		# # Load Merge(Docs, QueryFile)
		documents = Docs(outputDocs_Dir,querys)
		
		# # Build Dict
		dictionary = corpora.Dictionary(documents)
		once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq <= 20]
		dictionary.filter_tokens(once_ids)
		dictionary.compactify()
		# Save or not depends.
		dictionary.save('./dict.dict')
		# Use this if you saved before
		# dictionary = corpora.Dictionary.load('./dict.dict')

	    # TF-IDF calculation
		dc = DocCorpus(documents, dictionary)
		tfidf = models.TfidfModel(dc)

		# Build DocSimilarityMatrix  num_nnz = total number of non-zeroes in the BOW matrix
		index = Similarity(corpus=tfidf[dc], num_features=tfidf.num_nnz, output_prefix="shard",num_best=300)
		index.save('./sim.sim')
		# Use this if you saved before
		# index = Similarity.load('./sim.sim')

		# Writing down result of query
		with open(resultFileName, 'w+') as f:
			f.write("run,id,rel\n")
			# Run 1
			queryid = 1
			for query in querys:
				result = index[dictionary.doc2bow(query)]
				f.write("1,"+ str(queryid) +",")
				count = 0
				for rank in result:
					if int(rank[0]) < documents.getDocCount() and count < 100:
						docName = documents.getDocNameByID(rank[0])
						f.write(docName.split('.')[0] + " ")
						count+=1
				f.write("\n")
				queryid+=1
			# Run 2
			queryid = 1
			for query in querys:
				result = index[dictionary.doc2bow(query)]

				# filter by time.
				timeMgr = TimeManager()
				for rank in result[:50]:
					if int(rank[0]) < documents.getDocCount():
						docName = documents.getDocNameByID(rank[0])
						
						tree = ET.parse(os.path.join(originDocs_Dir, docName))
						root = tree.getroot()
						timeMgr.addTime(root[0].attrib['date'])

				# f.write("run,id,rel\n")
				f.write("2,"+ str(queryid) +",")
				count = 0

				maxMonths = timeMgr.maxAmountMonths()
				# print maxMonths
				for rank in result:
					if int(rank[0]) < documents.getDocCount() and count < 100:
						docName = documents.getDocNameByID(rank[0])
						tree = ET.parse(os.path.join(originDocs_Dir, docName))
						root = tree.getroot()
						dateOfDoc = root[0].attrib['date']

						if dateOfDoc[:-3] in maxMonths:
							docName = documents.getDocNameByID(rank[0])
							f.write(docName.split('.')[0] + " ")
							count+=1
				f.write("\n")
				queryid+=1
				
	else:
		print "Please use like 'python sim.py [originDocs_Dir] [outputDocs_Dir] [queryFile] [resultFileName]'"