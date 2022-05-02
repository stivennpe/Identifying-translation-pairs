# Identifying-translation-pairs

Implementation of the Character n-grams approach for detecting cross-lingual biomedical term translations from [Hakami and Bollegala, 2015](https://cgi.csc.liv.ac.uk/~huda/papers/biotrans.pdf)
Although authors propose three methods, namely, character n-grams, context  and hybrid models. This implementation makes use of only the concatenation of feature vectors from character n-grams (2-5).

An example from the article:

  ``English-French training instance (antibody, anticorps)``
  
  ``English n-grams features (Sn) an, nt, ib, bo, od, dy, ant, nti, tib, ibo, bod, ody``
  
  ``French n-grams features (Tn) an, nt, ti, ic, co, or, rp, ps, ant, nti, tic, cor, orp, rps``
  
  ``Feature vector [EN+an, EN+nt, EN+ib, . . ., FR+cor, FR+orp, FR+rps]``

Authors propose a model for translation detection as a binary classification problem employing character (2-5) n-grams. If a pair of tokens are translation equivalents, this system should return 1. The authors found that applying a Random Forest classifier over logistic regression with the concatenation of source and target vector spaces to generate a single vector space yielded the best results. 

Dataset
-----
The dataset used in this implementation is the CRAFT dataset for English, and using Google's translator I obtained the Spanish equivalent, inspired in the back translation tecnique used in Machine Translation. 

Results
-----

* The Logistic Regression obtained 0.6051 accuracy
* The Random Forest obtained 0.8735 accuracy
* The RidgeClassifier obtained 0.1886 accuracy

Training
-----

For training see models.py which also includes examples of inference.
