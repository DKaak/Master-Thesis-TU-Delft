# Discovering fraud related e-mails using Bayesian statistical techniques
## Master Thesis TU Delft - abstract

During a digital fraud investigation the search for relevant information in mailboxes of custodians is like finding a needle in a haystack. This time consuming task can, on various levels, be improved and made more efficient. Technology Assisted Review (TAR) is already one of the available machine learning algorithms that helps speeding up the process of finding relevant information. In Technology Assisted Review a model is trained based on the classification of e-mails by expert review. During the review process TAR continuously gives back the (potentially) most relevant e-mails that still need to be given a classification. The downside of this algorithm is that a manual expert review is still needed before TAR can give recommendations. In this thesis a introductory research is done on models that give an initial sorting before the expert review is done. The hypothesis that will be used is that this sorting (or classification) can be done in a similar manner as spam e-mails are removed to the junk folder in a mailbox. Three different features have been used (word frequencies, word occurrences and length of an e-mail) on four different models for each feature (A generative and discriminative model, each with maximum likelihood estimation or Bayesian estimation). Each of these 12 different implementations have been tested on three different datasets (TREC, ENRON and a confidential dataset).  Based on 5-fold cross validation the Bayesian generative model based on word frequencies has been shown to perform best on the confidential dataset. This model shows that a classification at the start of a digital fraud investigation can be done. Combining different models, and finding the best parameters for practical usage of the model is left for further research.

Keywords: classification, fraud, generative model, discriminative model, Naive Bayes, logistic regression, TAR


## Documented code
In the attached files the used and document python files can be found.

The setup files are used in to create the input based on the available text and .msg files for each dataset.

MLGM_Training is used to train the MLGM model (and MLGM1000W the corresponding training of the model based on 1000 words as parameters), MLGM is used to test the model based on the dataset that is used as input.

BEGM_Training is used to train the BEGM model (and BEGM1000W the corresponding training of the model based on 1000 words as parameters), BEGM is used to test the model based on the dataset that is used as input.

MLDM is used to train and  test the corresponding model based on the dataset that is used as input.

BEDM is used to train and test the corresponding model based on the dataset that is used as input.

AdaBoost is used to train and test the corresponding model based on the dataset that is used as input.
