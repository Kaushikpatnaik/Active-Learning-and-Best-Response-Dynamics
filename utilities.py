from numpy import sqrt, abs, array, dot, sign, float64, inf, ones, zeros
from numpy import cos, sin, pi, linspace, where
from scipy.stats import norm as normal
from scipy.linalg import norm, lstsq
import matplotlib.pyplot as plt

from cvxopt import matrix


def matrix_norm(M, a, b):
    a_norms = (abs(M)**a).sum(axis = 1)**(1.0 / a)
    return norm(a_norms, b)


def rank(M):
    M = array(matrix(M))
    return lstsq(M, ones(M.shape[0]))[2]


def display_distribution(U, ids, w_star = None, num_points = 10000):
    
    X = []
    Y = []
    P = []
    Q = []

    for i in range(num_points):
        if ids[i] == 1:
            x,y = U[i][1:]
            X.append(x)
            Y.append(y)
        else:
            p,q = U[i][1:]
            P.append(p)
            Q.append(q)

    plt.clf()

    '''
    plot_data(array(X), array(Y))
    if w_star is not None:
        plot_line(2*w_star, color = 'k')
    plt.axis('equal')
    plt.show()
    '''
    w_star /= norm(w_star)
    print w_star
    plt.plot(X,Y,'bo')
    plt.plot(P,Q,'g+')
    plt.plot([w_star[1]+w_star[0], -w_star[1]+w_star[0]], [-w_star[2]+w_star[0], w_star[2]+w_star[0]],'r-')
    #plt.show()

def display_inner_iterations(U, ids, separators , num_points):
    
    X = []
    Y = []
    P = []
    Q = []

    for i in range(num_points):
        if ids[i] == 1:
            x,y = U[i][1:]
            X.append(x)
            Y.append(y)
        else:
            p,q = U[i][1:]
            P.append(p)
            Q.append(q)

    plt.clf()

    plt.plot(X,Y,'bo')
    plt.plot(P,Q,'g+')
    
    color = ['r-','c-','m-','y-','k-']

    for j in range(len(separators)):
        #w_star = zeros((1,3))
    	w_star = separators[j]/norm(separators[j])
    	print 'weight vector in iteration ' + str(j)
        print w_star
    	plt.plot([w_star[1]+w_star[0], -w_star[1]+w_star[0]], [-w_star[2]+w_star[0], w_star[2]+w_star[0]],color[j])
        plt.annotate('Iteration ' + str(j), xy=(w_star[1]+w_star[0], -w_star[1]+w_star[0])) 
    plt.show()



def compute_margin(dist, w, p, q, num_points = 10000):
    
    min_margin = inf
    
    for i in range(num_points):
        try:
            x, y = dist.next()
        except StopIteration:
            break
        
        margin = abs(dot(w, x)) / (norm(w, q) * norm(x, p))
        
        if margin < min_margin:
            min_margin = margin
    
    return min_margin


def compute_data_norm(dist, p, m, num_trials):
    norms = []
    for trial in range(num_trials):
        data_matrix = array([dist.next()[0] for i in range(m)]).T
        data_norm = matrix_norm(data_matrix, 2, p)
        norms.append(data_norm)
    return sum(norms) / num_trials


### Other plotting functions
def plot_circle(color = 'k'):
    theta = linspace(0, 2*pi, 100)
    plt.plot(cos(theta), sin(theta), color)
    
def plot_data(labeled, labels, unlabeled = None):
    if unlabeled != None:
        unlabeled_x = unlabeled[:, 0]
        unlabeled_y = unlabeled[:, 1]
    
    pos = labeled[where(labels > 0)[0], :]
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    
    neg = labeled[where(labels < 0)[0], :]
    neg_x = neg[:, 0]
    neg_y = neg[:, 1]
    
    if unlabeled != None:
        plt.plot(unlabeled_x, unlabeled_y, 'ko')
    plt.plot(pos_x, pos_y, 'ro')
    plt.plot(neg_x, neg_y, 'bo')

def plot_line(w, color = 'k'):
    plt.plot([-w[1], w[1]], [w[0], -w[0]], color=color)
    
def plot_state(labeled, labels, unlabeled, w_star, w, pdf):
    plt.clf()
    plot_circle()
    plot_line(w_star, 'k')
    plot_data(labeled, labels, unlabeled)
    plot_line(w, 'g')
    
    plt.axis('equal')
    pdf.savefig()
