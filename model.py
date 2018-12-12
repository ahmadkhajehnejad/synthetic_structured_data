import numpy as np
import pickle
import scipy.stats

def _make_tree_structure(n, num_parents):
    adjacency_matrix = np.zeros([n,n])
    parents = [None] * n
    parents[0] = np.array([], dtype=np.int)
    for i in range(1,n):
        parents[i] = np.array(np.random.choice(list(range(i)),np.minimum(num_parents, i), replace=False), dtype=np.int)
        adjacency_matrix[i,parents[i]] = 1
        adjacency_matrix[parents[i],i] = 1
        for j in parents[i]:
            for t in parents[i]:
                if j != t:
                    adjacency_matrix[j,t] = 1
                    adjacency_matrix[t,j] = 1
    return [adjacency_matrix, parents]


def _make_Bayesian_tree_parameters(parents):
    n = len(parents)        
    theta_select = []
    for i in range(n):
        k = len(parents[i])
        if k == 0:
            theta_select.append(np.array([]))
        else:
            w = np.random.sample(k)
            w = w / sum(w)
            w[-1] = 1 - np.sum(w[:-1])
            theta_select.append(w)
    return theta_select

class SyntheticStructuredDataGenerator:
    
    
    def _generate_root_sample(self):
        outcome = np.zeros(self.d)
        outcome[0] = np.random.sample()
        for i in range(1,self.d):
            ind = np.where(np.random.multinomial(1, self.theta_inner[i]) > 0)[0][0]
            mu = outcome[self.parents_inner[i][ind]]
            outcome[i] = np.random.sample() * self.sigma_inner + mu
        return outcome
    
    def generate_samples(self,num_samples):
        all_outcomes = np.zeros([num_samples, self.D*self.d])
        for sn in range(num_samples):
            all_outcomes[sn, :self.d] = self._generate_root_sample()
            for i in range(1,self.D):
                ind = np.where(np.random.multinomial(1, self.theta_outer[i]) > 0)[0][0]
                ind_mu = self.parents_outer[i][ind]
                mu = all_outcomes[sn, (ind_mu * self.d) : ((ind_mu+1) * self.d)]
                all_outcomes[sn, (i*self.d):((i+1)*self.d)] = np.random.sample(self.d) * self.sigma_outer + mu
        return all_outcomes
    

    def __init__(self, D=20, d=30, sigma_outer=0.2, sigma_inner=0.2, num_parents_outer=3, num_parents_inner=5):

        self.D, self.d = D, d
        self.sigma_outer, self.sigma_inner = sigma_outer, sigma_inner
        self.num_parents_outer, self.num_parents_inner = num_parents_outer, num_parents_inner
        self.adjacency_matrix_inner, self.parents_inner = _make_tree_structure(d, num_parents_inner)
        self.theta_inner = _make_Bayesian_tree_parameters(self.parents_inner)
        self.adjacency_matrix_outer, self.parents_outer = _make_tree_structure(D, num_parents_outer)
        self.theta_outer = _make_Bayesian_tree_parameters(self.parents_outer)
        
    def save_model(self, file_name):
        with open( file_name, 'wb') as fout:
            pickle.dump([self.D, self.d,\
                         self.sigma_outer, self.sigma_inner,\
                         self.num_parents_outer, self.num_parents_inner, \
                         self.adjacency_matrix_outer, self.adjacency_matrix_inner,\
                         self.parents_outer, self.parents_inner,\
                         self.theta_outer, self.theta_inner],fout)
            
    def load_model(self, file_name):
        with open( file_name, 'rb') as fin:
            [self.D, self.d,\
                         self.sigma_outer, self.sigma_inner,\
                         self.num_parents_outer, self.num_parents_inner, \
                         self.adjacency_matrix_outer, self.adjacency_matrix_inner,\
                         self.parents_outer, self.parents_inner,\
                         self.theta_outer, self.theta_inner] = pickle.load(fin)


    def _root_log_likelihood(self, x):
        std_normal = scipy.stats.norm()

        if len(x.shape) == 1:
            x = x.reshape([1,-1])

        n = x.shape[0]

        ll = np.zeros([n])

        for i in range(1, self.d):
            likelihood = np.zeros([n])
            parents = x[:, self.parents_inner[i]]
            for j in range(parents.shape[1]):
                likelihood += std_normal.pdf((x[:, i] - parents[:, j]) / self.sigma_inner) * self.theta_inner[i][j]
            ll += np.log(likelihood)
        return ll


    def log_likelihood(self, x):
        std_multivariate_normal = scipy.stats.multivariate_normal(np.zeros([self.d]), np.eye(self.d))

        if len(x.shape) == 1:
            x = x.reshape([1, -1])
        n = x.shape[0]

        ll = self._root_log_likelihood(x[:,:self.d])
        for i in range(1, self.D):
            likelihood = np.zeros(n)
            x_i = x[:, (i * self.d) : ((i + 1) * self.d) ]
            parents_ind = self.parents_outer[i]
            for j in range(len(parents_ind)):
                x_par_j = x[:, (parents_ind[j]*self.d) : ((parents_ind[j]+1)*self.d) ]
                likelihood += std_multivariate_normal.pdf((x_i - x_par_j) / self.sigma_outer) * self.theta_outer[i][j]
            ll += np.log(likelihood)
        return ll

