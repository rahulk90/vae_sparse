""" Download text from wikipedia """
import urllib2,re
def getResponse(url):
    response = urllib2.urlopen(url)
    html = response.read()
    return html

def parseWiki(word):
    URL = ('https://en.wikipedia.org/w/index.php?title=%s&action=raw'%(word))
    result = getResponse(URL)
    sentence = re.sub('[^A-Za-z ]+', ' ', result)
    result = ' '.join([word.strip().lower() for word in sentence.split(' ') if len(word.strip())>=1])
    return result
#result = parseWiki('bird')
#import ipdb;ipdb.set_trace()

""" List of polysemous words, and context words """
polysemous_words = {}

"""fire"""
polysemous_words['fires']    = {} 
polysemous_words['fires']['control'] = 'bottle'
polysemous_words['fires']['c1'] = 'burn'
polysemous_words['fires']['c2'] = 'layoff'

"""park"""
polysemous_words['park']     = {} 
polysemous_words['park']['control']  = 'bottle'
polysemous_words['park']['c1']  = 'car'
polysemous_words['park']['c2']  = 'forest'

"""bank"""
polysemous_words['bank']     = {} 
polysemous_words['bank']['control']  = 'bottle'
polysemous_words['bank']['c1']  = 'river'
polysemous_words['bank']['c2']  = 'money' 

"""bar"""
polysemous_words['bar']      = {} 
polysemous_words['bar']['control']  = 'bottle'
polysemous_words['bar']['c1']   = 'pub'
polysemous_words['bar']['c2']   = 'lawyer'

"""court"""
polysemous_words['court']    = {}
polysemous_words['court']['control'] = 'bottle'
polysemous_words['court']['c1'] = 'sports'
polysemous_words['court']['c2'] = 'lawyer'

"""crane"""
polysemous_words['crane']    = {}
polysemous_words['crane']['control'] = 'bottle'
polysemous_words['crane']['c1'] = 'construction'
polysemous_words['crane']['c2'] = 'bird'


from wiki import parseWiki
for word in polysemous_words:
    w0 = polysemous_words[word]['c1']
    w1 = polysemous_words[word]['c2']
    polysemous_words[word]['c1_doc'] = parseWiki(w0)
    polysemous_words[word]['c2_doc'] = parseWiki(w1)
    print 'Downloading wikipedia entry for: ',w0,' ',w1
import os
fname = 'wordinfo.pkl'
from utils.misc import savePickle  
if os.path.exists(fname):
    os.remove(fname)
savePickle([polysemous_words],fname)
print 'Saved: ',fname
