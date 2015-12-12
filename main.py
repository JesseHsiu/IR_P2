# -*- coding: UTF-8 -*-
import requests
import opencc
import multiprocessing as mp
import io, json, sys, os
import xml.etree.ElementTree as ET
import codecs
from types import *

class OriginDocs(object):
	def __init__(self, dirname, outputDir, querySaveDir):
		self.dirname = dirname
		self.outputDir = outputDir
		self.querySaveDir = querySaveDir
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
	def queryFilesFromNLPIR(self):
		if not os.path.exists(self.querySaveDir):
			os.makedirs(self.querySaveDir)
		
		pool = mp.Pool(processes=100)
		for x in os.listdir(self.outputDir):
			if x == ".DS_Store":
				continue
			content = None
			with open(self.outputDir + "/" + x) as f:
				content = f.readline()
			print x
			pool.apply_async(requestFromPost, args=(x,content, ), callback=log_result)
		pool.close()
		pool.join()

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

########## FUNCTIONS ##########
def simplify(text):
 return opencc.convert(text, config='t2s.json')

def traditionalize(text):
 return opencc.convert(text, config='zhs2zht.ini').encode('utf-8')


def requestFromPost(filename, content):
	url = 'http://ictclas.nlpir.org/nlpir/index/getAllContentNew.do'
	params = {
	    'type': 'all',
	    'content': content,
	}
	headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36',
	           'X-Requested-With': 'XMLHttpRequest',
	           'Host': 'ictclas.nlpir.org',
	           'Referer': 'http://ictclas.nlpir.org/nlpir/index/getAllContentNew.do'}

	response = requests.get(url, params=params, headers=headers)

	fixtures = response.json()
	return [filename, fixtures]

def log_result(result):
	print "done",result[0]
	with io.open("./preprocessingData/resultNLPIR/" + result[0], 'w', encoding='utf-8') as f:
	  f.write(unicode(json.dumps(result[1], ensure_ascii=False)))

if __name__ == '__main__':

	if len(sys.argv) == 4:
		originDocs_Dir = sys.argv[1]
		outputDocs_Dir = sys.argv[2]
		querySave_Dir = sys.argv[3]

		originDocs = OriginDocs(originDocs_Dir, outputDocs_Dir, querySave_Dir)
		# originDocs.simplifyAllDoc();

		# Warning!!! connection needed and it will need about 3 hours querying.
		originDocs.queryFilesFromNLPIR();
	else:
		print "Please use like 'python sim.py [originDocs_Dir] [outputDocs_Dir] [querySaveDir]'"