from textblob import TextBlob
from pyspark import SparkConf, SparkContext
import re



def abb_en(line):
   abbreviation_en = {
    'u': 'you',
    'thr': 'there',
    'asap': 'as soon as possible',
    'lv' : 'love',    
    'c' : 'see'
   }
   
   abbrev = ' '.join (abbreviation_en.get(word, word) for word in line.split())
   return (abbrev)

def remove_features(data_str):
   
    url_re = re.compile(r'https?://(www.)?\w+\.\w+(/\w+)*/?')    
    mention_re = re.compile(r'@|#(\w+)')  
    RT_re = re.compile(r'RT(\s+)')
    num_re = re.compile(r'(\d+)')
    
    data_str = str(data_str)
    data_str = RT_re.sub(' ', data_str)  
    data_str = data_str.lower()  
    data_str = url_re.sub(' ', data_str)   
    data_str = mention_re.sub(' ', data_str)  
    data_str = num_re.sub(' ', data_str)
    return data_str

def polarity_check(polarity):
    if polarity > 0.0:
        polarityvalue = 'Positive'
    elif polarity < 0.0:
        polarityvalue = 'Negative'
    else:
        polarityvalue = 'Neutral'
    return polarityvalue


   
  
   
#Write your main function here
def main(sc, filename):
    
    data = sc.textFile(filename)\
    .map(lambda x: x.split(","))\
    .filter(lambda x: len(x) == 8)\
    .filter(lambda x: len(x[0])>1)
    
    #print(data.take(10))
    
    tweets = data.map(lambda x:x[7])\
    .map(lambda x:x.lower())\
    .map(lambda x: remove_features(x))\
    .map(lambda x: abb_en(x))
    
    #print(tweets.take(5))
                                    
    sentiment = tweets.map(lambda x:TextBlob(x)\
    .sentiment.polarity)\
    .map(lambda x: polarity_check(x))\
    
    #print(sentiment.take(5))  
    
    section1 = data.map(lambda x:x[0])#created_at
    section2 = data.map(lambda x:x[1])#screen_name
    section3 = data.map(lambda x:x[2])#text
    section5 = data.map(lambda x:x[3])#source
    section6 = data.map(lambda x:x[4])#location
    section7 = data.map(lambda x:x[5])#followers_count
    section8 = data.map(lambda x:x[6])#friends_count
    section9 = data.map(lambda x:x[7])#language
    section4 = sentiment#sentiment_polarity
     
    print(section1.take(5))
    print(section2.take(5))
    print(section3.take(5))
    print(section4.take(5))
    print(section5.take(5))
    print(section6.take(5))
    print(section7.take(5))
    print(section8.take(5))
    print(section9.take(5))
    
        
    output = section1\
    .zip(section2)\
    .zip(section3)\
    .zip(section4)\
    .zip(section5)\
    .zip(section6)\
    .zip(section7)\
    .zip(section8)\
    .zip(section9)\
    .map(lambda x: str(x).replace("'",''))\
    .map(lambda x: str(x).replace('"',''))\
    .map(lambda x: str(x).replace('(',''))\
    .map(lambda x: str(x).replace(')',''))\
    .saveAsTextFile("output1")
                                     
    print(output)  
    
    
    

if __name__ == "__main__":
    
    conf = SparkConf().setMaster("local[1]").setAppName("Sentiment Analysis")
    sc = SparkContext(conf=conf)
    
    filename = "starbucks.csv"
    
    main(sc, filename)

    sc.stop()

