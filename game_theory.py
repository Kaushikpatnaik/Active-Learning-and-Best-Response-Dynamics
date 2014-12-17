from numpy import sqrt, array, ones, zeros, dot, sign, float64, hstack, vstack, random, std
from numpy.random import uniform as unf 
from numpy.random import random_integers, rand, randint, shuffle
from numpy.linalg import norm
import sys
from collections import defaultdict, Counter
from experiments import *
from classrooms import *
from active_learners import *
from kernels import *
from itertools import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from math import *

class Create_Noisy_Labels(object):

    def __init__(self):
        self.noisy = 0
        self.total = 0

    def randomclassnoise(self, alpha, y):
        # receives a parameter alpha and an array of labels, which is randomly flips based on alpha
        temp_y = zeros(y.shape[0])
        self.total += y.shape[0]
        for i in range(y.shape[0]):
            if rand() < alpha:
                self.noisy += 1
                temp_y[i] = -y[i]
            else:
                temp_y[i] = y[i]

        return temp_y

    def linearclassnoise(self, intercept, slope, w_star, x, y):
        # receives intercept (noise at the separator), slope (how noise changes with distance from the separator)
        # w_star (optimal separator) datapoints x and array of labels y
        temp_y = zeros(y.shape[0])
        self.total += y.shape[0]
        for i in range(y.shape[0]):
            if rand() < slope * pow(abs(dot(w_star, x[i])),-1) + intercept:
                self.noisy += 1
                temp_y[i] = -y[i]
            else:
                temp_y[i] = y[i]

        return temp_y

    def randomballnoise(self, beta, y, dict_index):
        temp_y = zeros(y.shape[0])
        noisy = 0
        total = y.shape[0]
        while noisy <= beta*total:
            k = randint(0,y.shape[0])
            for v in iter(dict_index[k]):
                temp_y[v] =  -y[v]
            noisy += len(dict_index[k])

        for j in range(y.shape[0]):
            if temp_y[j] == 0:
                temp_y[j] = y[j]
        return temp_y

    def noiserate(self):
        if self.total == 0:
            return 0.0
        else:
            return self.noisy / float(self.total)


class Consensus_Dynamics(object):

    def __init__(self, opt_resp_iter, opt_update, opt_num_points):
        self.opt_update = opt_update
        self.opt_num_points = opt_num_points
        self.opt_resp_iter = opt_resp_iter

    def run_majority(self, inf_y, dict_index):
        # run a majority algorithm
        temp_inf_y = zeros(self.opt_num_points)
        if self.opt_update == 1:
            
            for i in range(self.opt_resp_iter):
                for k in dict_index.iterkeys():
                    sign = 0
                    for v in iter(dict_index[k]):
                        sign += inf_y[v]
                    if sign > 0:
                        temp_inf_y[k] = 1
                    else:
                        temp_inf_y[k] = -1
                    if temp_inf_y[k] == 0:
                        print "error"
                    	print sign, temp_inf_y[k]

                inf_y = temp_inf_y


        else:
            for i in range(self.opt_resp_iter):
                k = randint(0,self.opt_num_points)
                sign = 0
                for v in iter(dict_index[k]):
                    sign += inf_y[v]
                if sign > 0:
                    inf_y[k] = 1
                else:
                    inf_y[k] = -1

        return inf_y

    def run_weighted_majority(self, inf_y, dict_index, dict_weight):
        # run a weighted majority algorithm (based on distance)
        # the distance represents the weights that it reads from
        temp_inf_y = zeros(self.opt_num_points)
        if self.opt_update == 1:
            for i in range(self.opt_resp_iter):
                for k in dict_index.iterkeys():
                    sign = 0
                    for v in izip(iter(dict_index[k]), iter(dict_weight[k])):
                        sign += inf_y[v[0]] * v[1]
                    if sum(dict_weight[k]) != 0:
                        sign /= sum(dict_weight[k])
                    if sign > 0:
                        temp_inf_y[k] = 1
                    else:
                        temp_inf_y[k] = -1

                inf_y = temp_inf_y
        else:
            for i in range(self.opt_resp_iter):
                k = randint(0, self.opt_num_points)
                sign = 0
                for v in izip(iter(dict_index[k]), iter(dict_weight[k])):
                    sign += inf_y[v[0]] * v[1]
                    if sum(dict_weight[k]) != 0:
                        sign /= sum(dict_weight[k])
                    if sign > 0:
                        inf_y[k] = 1
                    else:
                        inf_y[k] = -1

        return inf_y

    def run_prob_knn(self, inf_y, dict_index):
        # run a probabilistic version of k-NN
        # look at neighbors, calculate ratio of maximum class and probabilistically pick it up

        temp_inf_y = zeros(self.opt_num_points)
        if self.opt_update == 1:
            for i in range(self.opt_resp_iter):
                for k in dict_index.iterkeys():
                    temp = []
                    for v in iter(dict_index[k]):
                        temp.append(inf_y[v])
                    c = Counter(temp)
                    temp_label, count_most  = c.most_common(1)[0]
                    flip_prob = count_most/len(dict_index[k])
                    # with flip prob choose the inf_label
                    if flip_prob < rand(1):
                        temp_inf_y[k] = temp_label
                inf_y = temp_inf_y
        else:
            for i in range(self.opt_resp_iter):
                k = randint(0, self.opt_num_points)
                temp = []
                for v in iter(dict_index[k]):
                    temp.append(inf_y[v])
                c = Counter(temp)
                temp_label, count_most  = c.most_common(1)[0]
                flip_prob = count_most/len(dict_index[k])
                # with flip prob choose the inf_label
                if flip_prob < rand(1):
                    inf_y[k] = temp_label

        return inf_y


class create_nn_graph(object):
    '''
    Parameterize how sensors/points communicate their information to each other including the cost associated
    The consensus game has a payout matrix where disagreement is costly and agreement is cheap
    Basically a 2x2 matrix at the start that pushes for similar labels with a probability
    the inclusion of probability adds to the noise of the dataset
    '''

    def __init__(self, dist, opt_num_points):
        self.opt_num_points = opt_num_points
        self.dict_index = defaultdict(list)
        self.dict_weight = defaultdict(list)
        self.dist = dist


    def radius_nn_graph(self, radius):
        # Case 1 - Sensors communicate their info only within a particular radius of themselves
        tree_obj = KDTree(self.dist)
        pair_points = tree_obj.query_pairs(radius,p=2.0,eps=0)
        pair_points_list = list(pair_points)
        for i in range(len(pair_points)):
            pta, ptb = pair_points_list[i]
            dist_pts = norm(self.dist[pta] - self.dist[ptb],2)
            self.dict_index[pta].append(ptb)
            self.dict_index[ptb].append(pta)
            self.dict_weight[pta].append(dist_pts)
            self.dict_weight[ptb].append(dist_pts)



    def knn_nn_graph(self, des_nbrs):
        # Case 3 - K - nearest neighbors instead of distance
        nbrs = NearestNeighbors(n_neighbors=des_nbrs, algorithm='auto').fit(self.dist)
        distances, indices = nbrs.kneighbors(self.dist)
        distances = distances.tolist()
        indices = indices.tolist()
        for i in range(self.opt_num_points):
            for j in range(des_nbrs):
                self.dict_index[i].append(indices[i][j])
                self.dict_weight[i].append(distances[i][j])

    def gaussian_nn_graph(self, radius):
        # Case 4 - A gaussian communication based on distance, (probabilistic)
        tree_obj = KDTree(self.dist)
        pair_points = tree_obj.query_pairs(radius,p=2.0,eps=0)
        pair_points_list = list(pair_points)
        for i in range(len(pair_points)):
            pta, ptb = pair_points_list[i]
            dist_pts = norm(self.dist[pta] - self.dist[ptb],2)
            if dist_pts < rand(1):
                self.dict_index[pta].append(ptb)
                self.dict_index[ptb].append(pta)
                self.dict_weight[pta].append(dist_pts)
                self.dict_weight[ptb].append(dist_pts)



def train_classifiers_gen_error(dist, inf_y, opt_w, label_budget, internal_iters =6, num_trials=50):

    '''
    Learn based on the active learning algorithm
    dist is the distribution on which learning will take place
    d is the number of dimensions
    inf_y is the current label set for the distribution
    opt_w is the optimal separator for calculating the generalization error
    label_budget can be list or a single number
    internal_iters is the number of iterations the active algorithm will take
    num_trials is the number of times the distribution will be shuffled and algorithm tested
    '''

    num_points = y.shape[0]
    dist_pre_consensus = UniformTestDataSet(d, dist, inf_y, num_points)

    avg_svm_angle = zeros(len(label_budget))
    avg_algo_angle = zeros(len(label_budget))
    avg_sm_angle = zeros(len(label_budget))

    svm_angle_std = zeros(len(label_budget))
    algo_angle_std = zeros(len(label_budget))
    sm_angle_std = zeros(len(label_budget))

    learners = []
    learners.append(lambda: MarginBasedTheoreticalParams(d,internal_iters,1,1,1))
    #learners.append(lambda: SimpleMarginSoftSVMBatch(d, 10))

    for r in range(len(label_budget)):

        accuracy = libsvm_angle_compare_test(learners, dist_pre_consensus, num_trials, num_points, label_budget[r], opt_w, display = False)

        # Get final direction
        # Compute the generalization error and print
        for i in range(len(accuracy)):
            avg_svm_angle[r] += accuracy[i][0]
            avg_algo_angle[r] += accuracy[i][1]
            #avg_sm_angle[r] += accuracy[i][2]

        svm_angle_std[r] = std(accuracy[:][0])
        algo_angle_std[r] = std(accuracy[:][1])
        sm_angle_std[r] = std(accuracy[:][2])

        avg_svm_angle[r] /= len(accuracy)
        avg_algo_angle[r] /= len(accuracy)
        #avg_sm_angle[r] /= len(accuracy)

        # Check accuracy or distance from optimal separator based on angle
        # print "For Label budget of " +str(label_budget[p])
        
    #print avg_algo_angle, avg_svm_angle

    #print "Error Labels are "
    #print algo_angle_std, svm_angle_std, sm_angle_std

    return avg_algo_angle, avg_svm_angle, avg_sm_angle, svm_angle_std, algo_angle_std, sm_angle_std

def train_classifiers_class_error(dist, inf_y, label_budget, internal_iters =6, num_trials=50):

    '''
    This differs from the previous method and calculates the classification error in a non-realizable case
    Learn based on the active learning algorithm
    dist is the distribution on which learning will take place
    d is the number of dimensions
    inf_y is the current label set for the distribution
    label_budget can be list or a single number
    internal_iters is the number of iterations the active algorithm will take
    num_trials is the number of times the distribution will be shuffled and algorithm tested
    '''

    num_points = y.shape[0]
    dist_used = UniformTestDataSet(d, dist, inf_y, num_points)
    test_size = 1000
    unlabeled_points = num_points - test_size
    avg_svm_angle = zeros(len(label_budget))
    avg_algo_angle = zeros(len(label_budget))
    avg_sm_angle = zeros(len(label_budget))
    p = GaussianKernel(0.1)

    learners = []
    learners.append(lambda: KernelMarginBasedTheoreticalParams(d, p, internal_iters))

    for r in range(len(label_budget)):

        accuracy = libsvm_compare_learners(learners, dist_used, num_trials, unlabeled_points, label_budget[r], test_size, display = False)

        # Get final direction
        # Compute the classification error
        # get algo error and then svm error
        avg_algo_angle[r] = accuracy[0][0]
        avg_svm_angle[r] = accuracy[0][1]

        #print avg_algo_angle[r], avg_svm_angle[r]

    return avg_algo_angle, avg_svm_angle, avg_sm_angle

def calculate_noisy(y, inf_y):
    # Calculate the final noise based on the best response dynamics
    count_noisy = 0
    num_points = y.shape[0]
    for i in range(num_points):
        if y[i] != inf_y[i]:
            count_noisy += 1

    noise_rate = (float(count_noisy)/num_points) * 100

    #print "Final noise after consensus calculations"
    #print count_noisy, noise_rate
    return count_noisy, noise_rate

if __name__ == '__main__':

    # Get variables for options
    opt_init_label_method = int(sys.argv[1])
    opt_comm = int(sys.argv[2])
    opt_dynamic = int(sys.argv[3])
    opt_update = int(sys.argv[4])
    opt_num_points = int(sys.argv[5])

    # So many parameters
    # At what radius do you want your nearest neighbors to be ?
    radius = 0.1
    # How many iterations do you want to do consensus updates
    opt_resp_iter = 100
    # how many nearest_neighbors do you want if you choose the nearest neighbors for connectivity
    des_nbrs = 20
    # how many random variations dataset, w_star combinations that you want to try ?
    super_iter = 50
    # How many dimensions ?
    d = 2

    # random noise rate
    alpha = 0.35

    # linear noise rate
    intercept = 0.06
    slope = 0.07

    # ball noise rate
    beta = 0.15

    # Label budget you want to test
    label_budget = [30, 40, 50, 60, 70, 80, 90, 100]
    #label_budget = [40]

    inf_y = zeros(opt_num_points)

    '''
    # prefix noisy points
    dist_noisy = []
    num_noisy = int(opt_num_points*alpha)
    while len(dist_noisy) < num_noisy:
        temp = unf(-1,1,(1,d))
        if norm(temp,2) <=1:
            dist_noisy.append(temp[0])
    '''

    for x in range(1):

        # list to calculate noise per super iter and thus calculate average noise
        iter_final_count_noisy = []
        iter_init_count_noisy = []
        pre_denoise_iter_algo_error = zeros((super_iter, len(label_budget)))
        pre_denoise_iter_svm_error = zeros((super_iter, len(label_budget)))
        pre_denoise_iter_sm_error = zeros((super_iter, len(label_budget)))

        post_denoise_iter_algo_error = zeros((super_iter, len(label_budget)))
        post_denoise_iter_svm_error = zeros((super_iter, len(label_budget)))
        post_denoise_iter_sm_error = zeros((super_iter, len(label_budget)))

        pre_denoise_iter_algo_std = zeros((super_iter, len(label_budget)))
        pre_denoise_iter_svm_std = zeros((super_iter, len(label_budget)))
        pre_denoise_iter_sm_std = zeros((super_iter, len(label_budget)))

        post_denoise_iter_algo_std = zeros((super_iter, len(label_budget)))
        post_denoise_iter_svm_std = zeros((super_iter, len(label_budget)))
        post_denoise_iter_sm_std = zeros((super_iter, len(label_budget)))


        for p in range(super_iter):

            print "In Iteration # " + str(p)

            '''
            Setup a two dimensional environment with points, which should act like sensors
            Topology of points can vary
            Creating a random uniform distribution first
            '''

            
            # Choose randomly an optimal separator
            w_star = ones(2)
            w_star /= norm(w_star,2)
            

            #print w_star

            # generating a uniform ball instead of a square
            dist = []
           
            # change based on pre selecting noisy points or not
            #num_clean = opt_num_points - num_noisy
            while len(dist) < opt_num_points:
                temp = unf(-1,1,(1,d))
                if norm(temp,2) <= 1:
                    dist.append(temp[0])
                    

            # Select or unselect based on fixing the noisy points in advance
            #for i in range(num_noisy):
            #    dist.append(dist_noisy[i])

            dist = array(dist)

            # Change based on realizable or non-realizable case
            y = zeros(opt_num_points)
            for i in range(opt_num_points):

                # Realizable case
                y[i] = sign(dot(w_star, dist[i]))

                # Non- Realizable case with a sine curve
                #if dist[i,1] > sin(dist[i,0]*pi)/3:
                #    y[i] = 1
                #else:
                #    y[i] = -1

            dict_index = defaultdict(list)
            dict_weight = defaultdict(list)
            nn_object = create_nn_graph(dist,opt_num_points)
            # Accumulate information
            # create the Neighborhood dictionary
            if opt_comm == 1:
                nn_object.radius_nn_graph(radius)
                dict_index = nn_object.dict_index
                dict_weight = nn_object.dict_weight

            elif opt_comm == 2:
                nn_object.radius_nn_graph(radius)
                dict_index = nn_object.dict_index
                dict_weight = nn_object.dict_weight

            else:
                nn_object.knn_nn_graph(des_nbrs)
                dict_index = nn_object.dict_index
                dict_weight = nn_object.dict_weight


            noisy = Create_Noisy_Labels()

            # Introducing noise
            # Random
            # Distance based
            # Pocket
            count_noisy = 0
            if opt_init_label_method == 1:

                # random noise
                inf_y = noisy.randomclassnoise(alpha, y)

                '''
                # Scheme for pre-choosing noisy points
                for i in range(num_clean):
                    inf_y[i] = y[i]
                for j in range(num_noisy):
                    inf_y[j+num_clean] = -y[j + num_clean]

                # creating own shuffling scheme
                z = hstack((dist, inf_y.reshape(opt_num_points,1)))
                u = hstack((z, y.reshape(opt_num_points,1)))
                shuffle(u)
                dist = u[:,0:2]
                inf_y = u[:,2]
                y = u[:,3]
                '''

            elif opt_init_label_method == 2:
                # choose based on w_star and distance (a random line in the unit ball)
                inf_y = noisy.linearclassnoise(intercept,slope,w_star,dist,y)
            else:
                # create random balls of noise
                inf_y = noisy.randomballnoise(beta, y, dict_index)

            temp1, temp2 = calculate_noisy(y, inf_y)
            iter_init_count_noisy.append(temp1)

            '''
            Learn based on the active learning algorithm
            Switch between train_classifier_gen_error
            and train_classifier_class_error as necessary
            '''
            
            # Running before denoising
            temp3, temp4, temp5, temp6, temp7, temp8 = train_classifiers_gen_error(dist, inf_y, w_star, label_budget)
            pre_denoise_iter_algo_error[p] = temp3
            pre_denoise_iter_svm_error[p] = temp4
            pre_denoise_iter_sm_error[p] = temp5
            pre_denoise_iter_algo_std[p] = temp6
            pre_denoise_iter_svm_std[p] = temp7
            pre_denoise_iter_sm_std[p] = temp8
            #print iter_algo_error[p], iter_svm_error[p]
            #print "\n"
            
            '''
            run best response dynamics for desired number of iterations
            '''

            consensus_obj = Consensus_Dynamics(opt_resp_iter,opt_update, opt_num_points)
            if opt_dynamic == 1:
                # run a majority algorithm
                inf_y = consensus_obj.run_majority(inf_y, dict_index)

            elif opt_dynamic == 2:
                # run a weighted majority algorithm (based on distance)
                # the distance represents the weights that it reads from
                inf_y = consensus_obj.run_weighted_majority(inf_y, dict_index, dict_weight)

            else:
                # run a probabilistic version of k-NN
                # look at neighbors, calculate ratio of maximum class and probabilistically pick it up
                inf_y = consensus_obj.run_prob_knn(inf_y, dict_index)

            temp6, temp7 = calculate_noisy(y, inf_y)
            # print "In iteration " + str(p) + " noise rate is " + str(temp7)
            iter_final_count_noisy.append(temp6)
            

            '''
            Learn based on the active learning algorithm
            Query appropriate points to determine optimal direction
            Remember you are querying the inferred labels rather than actual labels
            Theoretical margin based method with parameters for the uniform distribution
            Calculate cost of inferring results
            '''
            
            temp8, temp9, temp10, temp11, temp12, temp13 = train_classifiers_gen_error(dist, inf_y, w_star, label_budget)
            post_denoise_iter_algo_error[p] = temp8
            post_denoise_iter_svm_error[p] = temp9
            post_denoise_iter_sm_error[p] = temp10
            post_denoise_iter_algo_std[p] = temp11
            post_denoise_iter_svm_std[p] = temp12
            post_denoise_iter_sm_std[p] = temp13
            #print iter_algo_error[p], iter_svm_error[p]
            #print "\n"
            
       
        avg_init_noise_rate = sum(iter_init_count_noisy)/float(opt_num_points*super_iter)
        print "Average initial noise rate over " + str(super_iter) + " is " + str(avg_init_noise_rate)


        for q in range(len(label_budget)):
            avg_algo_error = sum(pre_denoise_iter_algo_error[:,q])/float(super_iter)
            avg_svm_error = sum(pre_denoise_iter_svm_error[:,q])/float(super_iter)
            avg_sm_error = sum(pre_denoise_iter_sm_error[:,q])/float(super_iter)

            avg_algo_std = std(pre_denoise_iter_algo_std[:,q])
            avg_svm_std = std(pre_denoise_iter_svm_std[:,q])
            avg_sm_std = std(pre_denoise_iter_sm_std[:,q])

            print "Averaged generalization error pre denoising for " + str(label_budget[q])
            print "Algo: SVM: SimpleMargin:  " + str(avg_algo_error) + " " + str(avg_svm_error) + " " + str(avg_sm_error)
            print "Algo: SVM: SimpleMargin:  " + str(avg_algo_std) + " " + str(avg_svm_std) + " " + str(avg_sm_std)
            print "\n"
        

        avg_final_noise_rate = sum(iter_final_count_noisy)/float(opt_num_points*super_iter)
        print "Average final noise rate over " + str(super_iter) + " is " + str(avg_final_noise_rate)
        print "\n"

        for q in range(len(label_budget)):
            avg_algo_error = sum(post_denoise_iter_algo_error[:,q])/float(super_iter)
            avg_svm_error = sum(post_denoise_iter_svm_error[:,q])/float(super_iter)
            avg_sm_error = sum(post_denoise_iter_sm_error[:,q])/float(super_iter)


            avg_algo_std = std(post_denoise_iter_algo_std[:,q])
            avg_svm_std = std(post_denoise_iter_svm_std[:,q])
            avg_sm_std = std(post_denoise_iter_sm_std[:,q])
            print "Averaged generalization error post denoising for " + str(label_budget[q])
            print "Algo: SVM: SimpleMargin:  " + str(avg_algo_error) + " " + str(avg_svm_error) + " " + str(avg_sm_error)
            print "Algo: SVM: SimpleMargin:  " + str(avg_algo_std) + " " + str(avg_svm_std) + " " + str(avg_sm_std)
            print "\n"


