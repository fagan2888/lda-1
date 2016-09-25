from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords, brown
import itertools
import string
import numpy as np
from scipy.special import gamma
from scipy.special import digamma

class LDA_VI():
    def __init__(self, docword, k=5, eta=0.01, alpha=0.01):
        '''
        m: document number
        n: word number
        alpha: Dirichlt prior for topic
        eta: Dirichlet prior for word
        beta: word distribution
        lambda_: variational parameter governs eta
        phi: variational parameter governs topic asgnment
        gamma_: variational parameter governs theta
        '''
        self.k = k
        self.docword = docword
        self.m, self.v = np.shape(docword)
        self.eta = eta
        self.alpha = alpha
        self.phi = np.full((self.m, self.v, self.k), 1/self.k)
        self.gamma_ = np.full((self.m, self.k), self.alpha+self.v/self.k) 
        
    def lgamma(self, var):
        return np.log(np.absolute(gamma(var)))
    
    def updat_lambda_(self, eta, phi, docword, num_doc, num_word, num_top):
        # shape(k, v)
        lambda_ = np.full((num_top, num_word), eta)
        for doc_idx in range(num_doc):
            lambda_ += phi[doc_idx].T * docword[doc_idx, :]
        return lambda_

    def updat_gamma_in_d(self, alpha, phi_in_d):
        # shape(1, k)
        # prior of theta
        gamma_in_d = alpha + np.sum(phi_in_d, axis=0)
        return gamma_in_d

    def updat_phi_in_d(self, gamma_in_d, lambda_, v):
        t1 = digamma(np.tile(gamma_in_d, (v, 1)))
        t2 = digamma(lambda_).T
        t3 = digamma(np.tile(np.sum(lambda_, axis=1), (v, 1)))
        phi_in_d = np.exp(t1+t2-t3)
        phi_in_d /= np.sum(phi_in_d, axis=1).reshape(v, 1)
        return phi_in_d
                 
    def llh_d(self, gamma_in_d, alpha, k, document, phi_in_d, eta, lambda_):
        t1 = self.lgamma(alpha*k)\
            - self.lgamma(alpha) * k \
            + (alpha - 1) * np.sum(digamma(gamma_in_d) - digamma(np.sum(gamma_in_d)))
        t21 = digamma(gamma_in_d) - digamma(np.sum(gamma_in_d))
        t2 = np.sum(np.dot(phi_in_d, t21.reshape(k, 1)))
        t3 = np.sum(phi_in_d*document.reshape(len(document), 1)*lambda_.T)
        t4 = self.lgamma(np.sum(gamma_in_d))\
            + np.sum(self.lgamma(gamma_in_d))\
            - np.sum((gamma_in_d-1)*(digamma(gamma_in_d)-digamma(np.sum(gamma_in_d))))
        t5 = np.sum(phi_in_d*np.log(phi_in_d))
        t6 = self.lgamma(eta*k)\
            - self.lgamma(eta) * k\
            + (eta - 1) * np.sum(digamma(lambda_) - digamma(np.sum(lambda_, axis=1)).reshape(k, 1))
        t7 = self.lgamma(np.sum(lambda_))\
             + np.sum(self.lgamma(lambda_))\
             - np.sum((lambda_-1)*(digamma(lambda_)-digamma(np.sum(lambda_, axis=0))))
        '''
        print ('t1:', t1)
        print ('t2:', t2)
        print ('t3:', t3)
        print ('t4:', t4)
        print ('t5:', t5)
        print ('t6:', t6)
        print ('t7:', t7)
        '''
        llh = t1 + t2 + t3 + t6 - t4 - t5 - t7
        return llh
    
    def get_topic_words(self, lambda_, voc, k, num_word=10):
        for top_idx in range(k):
            topic_word_id = lambda_[top_idx, :].argsort()[-num_word:][::-1]
            print ('Topic ', top_idx)
            for id_ in topic_word_id:
                print (voc[id_], ' ')

    def run(self, voc, tol=0.01, max_iter=1000):
        llh_old = 0
        for _ in range(max_iter):
            llh_new = 0
            self.lambda_ = self.updat_lambda_(self.eta, self.phi, self.docword, self.m, self.v, self.k)
            for doc_idx in range(docword.shape[0]):
                self.gamma_[doc_idx, :] = self.updat_gamma_in_d(self.alpha, self.phi[doc_idx, :, :])
                self.phi[doc_idx, :, :] = self.updat_phi_in_d(self.gamma_[doc_idx, :], self.lambda_, self.v)
                llh_new += self.llh_d(self.gamma_[doc_idx, :], self.alpha, self.k, self.docword[doc_idx, :], self.phi[doc_idx, :, :], self.eta, self.lambda_)
            if llh_new - llh_old < tol and llh_new > llh_old:
                return
            else:
                llh_old = llh_new

            self.get_topic_words(self.lambda_, voc, self.k)

