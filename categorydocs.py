# -*- coding: UTF-8 -*-
import feedparser
from tgrocery import Grocery
import opencc
import json
import sys, os, io
import xml.etree.ElementTree as ET
import codecs
from types import *
import multiprocessing as mp
import collections
import jieba
import jieba.analyse

allowCategory = [ u'社會', u'娛樂', u'政治', u'生活']

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
		# self.grocery = grocery
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

########## FUNCTIONS ##########

def parseFile(filename, dirname, outputDir):	
	tree = ET.parse(os.path.join(dirname, filename))
	root = tree.getroot()
	
	# Noted that there is no 。 due to split by this punc.
	puncs = ['，', '?', '@', '!', '$', '%', '『', '』', '「', '」', '＼', '｜', '？', ' ', '*', '(', ')', '~', '.', '[', ']', '\n','1','2','3','4','5','6','7','8','9','0']

	title_Text = root[0].attrib['title']
	title_Text = title_Text.encode('utf-8')
	# remove useless puncs
	for punc in puncs:
		title_Text = title_Text.replace(punc,'')
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


def simplify(text):
 return opencc.convert(text, config='t2s.json')

def traditionalize(text):
 return opencc.convert(text, config='zhs2zht.ini').encode('utf-8')

if __name__ == '__main__':
	
	if len(sys.argv) == 4:
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
		originDocs = OriginDocs(originDocs_Dir, outputDocs_Dir)
		# originDocs.simplifyAllDoc();

		# ======= Category Original Docs and save to json file =======
		for x in os.listdir(outputDocs_Dir):
			if x == ".DS_Store":
				continue
			content = None
			with open(outputDocs_Dir + "/" + x) as f:
				content = f.readline()
			list = []
			content_sentences = content.split('。')
			for sentence in content_sentences:
				if len(sentence) != 0:
					list.extend(jieba.cut(sentence, cut_all=True))
			result = grocery.predict(content)
			print x
			with codecs.open(outputWithCateDocs_Dir + "/" + x, "w", "utf-8") as f:
				f.write(str(result))
				f.write("\n")
				for term in list:
					# if the term word is less than 2, which is a single word, then ignore it
					if len(term) < 2:
						continue
					else:
						f.write(term)
						f.write(',')
				
		# ======= Train 4 Category with 3 Publisher(LTN,CTS,APD) =======
		

	else:
		print "Please use like 'python sim.py [originDocs_Dir] [outputDocs_Dir] [With]'"