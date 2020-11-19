'''
Naive Bayes Theroem:
    P(A|B) = (P(B|A) * P(A)) / P(B)
P(A|B) => Postirior Probability
P(A) = Prior Probability
P(B|A) = Likelihood
P(B) = Marginal Probability or Evidence

Types of Naive Bayes Algorithm:
    1. Bernoulli Naive Bayes:
        - Whenever there is a binomial values that is 0/1, yes/no, etc. then Bernoulli Naive Bayes is used.
        - Distribution: Bernoulli Distribution
            P(Success) = p
            P(Failure) = q = 1 - p
            Let random value be X. If X is 1 then Success, and if 0 then Failure.
            X has Bernoulli Distribution.
            P(X = x) = p^x * (1 - p)^(1-x)
        - The decision rule for Bernoulli naive Bayes is based on :
            P(xi|y) = P(i|y)xi + (1 - P(i|y))(1 - xi)

    2. Multinomial Naive Bayes:
        - Whenever there is a situation such as we are intrested in the freuqency of occurence of the word in a text document then we use Multinomial Naive Bayes.
        - Distribution: Mutlinomial Distribution
        - Formula:
            P(X1=x1, X2=x2, .... , Xk=xk) = (n! * (P1^x1 .... Pk^xK)) / (x1!.....xk!)

    3. Gaussian Naive Bayes:
        - If our data is Discrete and we want to focus on Discrete values then not to use GNB.
        - If data is of Continuous nature then use GNB.
        - Distribution: Gaussian Distribution
        - Formula:
            P(xi|y) = (exp(-(xi - mean(y))^2 / (2 * variance(y)))) / sqrt(2 * pi * variance(x)
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

X,y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

GNB = GaussianNB()
y_pred = GNB.fit(X_train,y_train).predict(X_test)
print(accuracy_score(y_test, y_pred))