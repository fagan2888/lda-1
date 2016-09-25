class LDASampler(object):
    def __init__(self, d=4211, w=5000):
        '''
            alpha: topic distribution over documents
            beta: word distribution over topicWordSum
           self.K: topic number
            docTopic: document-topic-matrix
            topicWord: topic-word-matrix
            topicWordSum: topic count
            documents: document-word-matrix
        '''
        self.alpha = 0.1
        self.beta = 0.1
        self.K = 20
        self.d = d
        self.w = w
        self.docTopic = np.zeros((self.K, d))
        self.topicWord = np.zeros((self.K, w))
        self.topicWordSum = np.zeros((self.K))
        self.docTopicSum = np.zeros((self.K))
        #dfTrain = dfItemReview[:2400]
        #dfTest = dfItemReview[2400:]
        self.documents = df_to_matrix(dfItemReview[-100:])
        #self.documentsTest = df_to_matrix(dfTest)
        self.topicAsgn = {}
        #self.chain =[[],[]]
        #self.zScore = [[],[]]#[[phi],[theta]]
        self.phi = np.full((self.K, w), 1/w)
        self.theta = np.full((self.K, d), 1/self.K)
        self.phiTrace = [None] * self.w
        self.thetaTrace = [None] * self.d
        #10 words to descripe one topic
        self.bestWords = np.zeros((self.K, 10))    
        self.bestTopic = np.zeros((self.K, d))  
        self.startSample = False
        
                
    def _initialize(self):
        #keep track on the topic assignment on each word 
        wordTrack = 1
        for index, occurence in np.ndenumerate(self.documents):
            if occurence == 0:
                continue
            docIndex = index[0]
            wordIndex = index[1]
            for _ in xrange(occurence):
                topicIndex = np.random.randint(0,self.K)
                self.topicAsgn[(docIndex, wordTrack)] = topicIndex
                self.docTopic[topicIndex, docIndex] += 1
                self.topicWord[topicIndex, wordIndex] += 1
                self.topicWordSum[topicIndex] += 1
                self.docTopicSum[topicIndex] += 1
                wordTrack += 1
        #create w vectors for phiTrace
        #and d vectors for thetaTrace
        for wordIndex in xrange(self.w):
            self.phiTrace[wordIndex] = np.zeros((1, self.K))
        for docIndex in xrange(self.d):
            self.thetaTrace[docIndex] = np.zeros((1, self.K))
    
    def _cut_first_layer(self):
        for wordIndex in xrange(self.w):
            self.phiTrace[wordIndex] = self.phiTrace[wordIndex][1:]
        for docIndex in xrange(self.d):
            self.thetaTrace[docIndex] = self.thetaTrace[docIndex][1:]

    def _condition_distribution(self, docIndex, wordIndex):
        '''
       
        '''
        proTopicWord = (self.topicWord[:, wordIndex] + self.beta) / (self.topicWordSum + self.beta)
        proTopicWord /= np.sum(proTopicWord)
        proTopicDoc = (self.docTopic[:, docIndex] + self.alpha) / (self.docTopicSum + self.alpha)
        proTopicDoc /= np.sum(proTopicDoc)
        prob = proTopicWord * proTopicDoc
        #renormalize to eusure sum(prob) == 1
        prob /= np.sum(prob)
        #return a vertical vector
        return prob
        
            
    def _cal_convergence(self, chain, first=.3, last=.5, interval=2):
        if np.ndim(chain) > 1:
            if min(first, last) * len(chain) < 2:
                return
            else:
                return [self._cal_convergence(y) for y in np.transpose(chain)]
        zScore = []
        end = len(chain) - 1
        starts = np.arange(0, end // 2, step=int(end / 2) / (interval - 1))
        for start in starts:
            lenOfStart = int(first * (end - start))
            firstSlice = chain[start: start + lenOfStart]
            #print 'first',firstSlice, 'start:', start, 'to:', lenOfStart
            lenOfEnd = int(last * (end - start))
            lastSlice = chain[end - lenOfEnd:]
            
            z = (firstSlice.mean() - lastSlice.mean())
            z /= np.sqrt(firstSlice.std() ** 2 + lastSlice.std() ** 2)

            zScore.append(z)
        return zScore
    
    def _is_convergent(self, zScore):
        if np.all(zScore) <= 1 and np.all(zScore) >= -1:
            return True
    
    def _check_convergent(self, traceList):
        #:param an array of vector
        for param in traceList:
            #print 'param:', param
            zScore = self._cal_convergence(param)
            print 'zscore:', zScore
            #print 'phi:', self.phiTrace
            #print 'theta:', self.thetaTrace
            if self._is_convergent(zScore):
                self.startSample = True
            else:
                self.startSample = False
                break
                
    def _sampling(self, matrix):
        wordTrack = 1
        for index, occurence in np.ndenumerate(matrix):
            if occurence == 0:
                continue
            #index: (docIndex, wordIndex)
            docIndex = index[0] 
            wordIndex = index[1]
            for i in xrange(occurence):
                topicIndex = self.topicAsgn[(docIndex, wordTrack)]
                self.docTopic[topicIndex, docIndex] -= 1
                self.topicWord[topicIndex, wordIndex] -= 1
                self.topicWordSum[topicIndex] -= 1
                self.docTopicSum[topicIndex] -= 1
                prob = self._condition_distribution(docIndex=docIndex, wordIndex=wordIndex)
                newTopicIndex = np.random.multinomial(1, prob).argmax()
                self.topicAsgn[(docIndex, wordTrack)] = newTopicIndex
                self.docTopic[newTopicIndex, docIndex] += 1
                self.topicWord[newTopicIndex, wordIndex] += 1
                self.topicWordSum[newTopicIndex] += 1
                self.docTopicSum[newTopicIndex] += 1
                wordTrack += 1
            

    #measure the topic distribution
    def _cal_phi_and_theta(self):
        '''
            update phi and theta
            add into phiTrace and thetaTrace
        '''
        for wordIndex in xrange(self.w):
            self.phi[:, wordIndex] = (self.topicWord[:, wordIndex] + self.beta) / (self.topicWordSum + self.beta)
            self.phi[:, wordIndex] /= sum(self.phi[:, wordIndex])
            self.phiTrace[wordIndex] = np.vstack((self.phiTrace[wordIndex], np.transpose(self.phi[:, wordIndex])))
        for docIndex in xrange(self.d):
            self.theta[:, docIndex] = (self.docTopic[:, docIndex] + self.alpha) / (self.docTopicSum + self.alpha)
            self.theta[:, docIndex] /= sum(self.theta[:, docIndex]) 
            self.thetaTrace[docIndex] = np.vstack((self.thetaTrace[docIndex], np.transpose(self.theta[:, docIndex])))
           
    def run(self, maxIteration=50, burnin=3, lag=5, cut=False):
        self._initialize()  
        for i in range(maxIteration):
            print i
            self._sampling(self.documents)
            if i > burnin:
                self._cal_phi_and_theta()

                if cut == False:
                    self._cut_first_layer()
                    cut = True
                for traceList in [self.phiTrace, self.thetaTrace]:           

                    print 'traceList:', traceList
                    self._check_convergent(traceList)
                
                if self.startSample:
                    lag -= 1
                
                if lag == 0:
                    break
        
        #measure posterior phi and theta        
        for wordIndex in xrange(self.w):
            self.phi[:, wordIndex] = np.mean(self.phiTrace[wordIndex][-lag:, 0:])
            self.phi[:, wordIndex] /= sum(self.phi[:, wordIndex])
        for docIndex in xrange(self.d):
            self.theta[:, docIndex] = np.mean(self.thetaTrace[docIndex][-lag:, 0:])
            self.theta[:, docIndex] /= sum(self.theta[:, docIndex])
            
        self.bestWords = np.argsort(self.phi)
        self.bestTopic = np.argsort(self.theta)
        
        print (self.bestWords)
        print (self.bestTopic)