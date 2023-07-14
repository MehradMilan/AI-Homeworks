import pandas as pd
import numpy as np  
class BN(object):
    """
    Bayesian Network implementation with sampling methods as a class
    
    Attributes
    ----------
    n: int
        number of variables
        
    G: dict
        Network representation as a dictionary. 
        {variable:[[children],[parents]]} # You can represent the network in other ways. This is only a suggestion.

    topological order: list
        topological order of the nodes of the graph

    CPT: list
        CPT Table
    """
    
    def __init__(self , graph, CPT ) -> None:
        self.G = graph
        self.CPT = CPT
        self.topological_order = []
        self.set_topological_order()
        self.joint = self.create_joint_table()
        self.n = len(self.G)
        
        ############################################################
        # Initialzie Bayesian Network                              #
        # (1 Points)                                               #
        ############################################################
        
        #Your code
        
    
    def cpt(self, node) -> dict:
        return self.CPT[node]
        ############################################################
        # (3 Points)                                               #
        ############################################################
        
        #Your code
        pass
    
    def join_factors(self, factor1, factor2):
        c1 = factor1.columns
        c2 = factor2.columns
        c = list(set(c1) & set(c2))
        result = pd.merge(factor1, factor2, on = c)
        column_names1 = factor1.keys()
        column_names2 = factor2.keys()
        for k1 in column_names1:
            if k1.startswith('p'):
                break
        for k2 in column_names2:
            if k2.startswith('p'):
                break
        result['p' + k1[1:] + k2[1:]] = result[k1] * result[k2]
        result = result.drop([k1, k2], axis = 1)
        return result

    def create_joint_table(self):
        joint_table = self.cpt(self.topological_order[0])
        for c in self.topological_order:
            if c == self.topological_order[0]:
                continue
            joint_table = self.join_factors(joint_table, self.cpt(c))
        return joint_table

    def pmf(self, query, evidence) -> float:
        ############################################################
        # (3 Points)                                               #
        ############################################################
        
        temp_table = self.joint
        for var, value in evidence.items():
            temp_table = temp_table[temp_table[var] == value]

        temp_table = temp_table.groupby(list(query.keys())).sum()
        temp_table['pABECDF'] /= sum(temp_table['pABECDF'])
        temp_table.reset_index(inplace=True)
        for var, value in query.items():
            temp_table = temp_table[temp_table[var] == value]

        return float(temp_table['pABECDF'])
    
    def sampling(self, query, evidence, sampling_method, num_iter, num_burnin = 100) -> float:
        if(sampling_method == "Prior" )  :
            return self.prior_sample(query, evidence, num_iter)
        elif(sampling_method =="Rejection" ) :
            return self.rejection_sample(query, evidence, num_iter)
        elif (sampling_method =="Likelihood Weighting") :
            return self.likelihood_sample(query, evidence, num_iter)
        elif (sampling_method=="Gibbs" ) :
            return self.gibbs_sample(query, evidence, num_iter, num_burnin)
        else:
            print("Sampling method not found")
            return 0.0    

        ############################################################
        # (27 Points)                                              #
        #     Prior sampling (6 points)                            #
        #     Rejection sampling (6 points)                        #
        #     Likelihood weighting (7 points)                      #
        #     Gibbs sampling (8 points)                      #
        ############################################################
        
        #Your code
    def sample_with_prob(self ,prob):
        return np.random.choice([0,1], p=[1-prob, prob])

    def prior_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            prior samples
        """
        samples = pd.DataFrame(columns=list(self.G.keys()), dtype=int)
        for _ in range(num_iter):
            temp = {}
            for var in self.G.keys():
                parents = {}
                for p in self.G[var][1]:
                    parents[p] = temp[p]
                temp[var] = self.sample_with_prob(self.pmf({var: 0}, parents))
            samples = samples.append(temp, ignore_index=True)
        return self.find_infered_prob(samples, query, evidence)

    def sample_consistent_with_evidence(self, sample, evidence):
        """
            To check if a sample is consistent with evidence or not?

            Parameters
            ----------
            sample:
                a sample
            evidence:
                evidence set
            
            Returns
            -------
            True if the sample is consistent with evidence, False otherwise.
        """
        for var, value in evidence.items():
            if sample[var] != value:
                return False
        return True
    
    
    def sample_consistent_with_query(self, sample, query):
        """
            To check a sample is consistent with query or not?

            Parameters
            ----------
            sample:
                a sample
            evidence:
                query set
            
            Returns
            -------
            True if the sample is consistent with query, False otherwise.
        """
        for var, value in query.items():
            if sample[var] != value:
                return False
        return True

    def find_infered_prob(self, samples, query, evidence):
        for var, variable in evidence.items():
            samples = samples[samples[var] == variable]
        m = samples.shape[0]
        for var, variable in query.items():
            samples = samples[samples[var] == variable]
        k = samples.shape[0]
        return k/m  
        
    def get_prior_sample(self):
        """
            Returns
            -------
            Returns a set which is the prior sample. 
        """
        sample = {}
        for var in self.G.keys():
            parents = {}
            for p in self.G[var][1]:
                parents[p] = sample[p]
            sample[var] = self.sample_with_prob(self.pmf({var: 0}, parents))
        return sample
    
    
    def rejection_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            rejection samples
        """
        samples = pd.DataFrame(columns=list(self.G.keys()), dtype=int)
        for _ in range(num_iter):
            temp = {}
            q = True
            for var in self.G.keys():
                parents = {}
                for p in self.G[var][1]:
                    parents[p] = temp[p]
                temp[var] = self.sample_with_prob(self.pmf({var: 0}, parents))
                if var in evidence.keys() and temp[var] != evidence[var]:
                    q = False
            if q:
                samples = samples.append(temp, ignore_index=True)
        return self.find_infered_prob(samples, query, evidence)
        
    def likelihood_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            likelihood samples
        """
        samples = pd.DataFrame(columns=list(self.G.keys()), dtype=int)
        for _ in range(num_iter):
            temp = {}
            weight = 1
            for var in self.G.keys():
                parents = {}
                for p in self.G[var][1]:
                    parents[p] = temp[p]
                if var not in evidence.keys():
                    temp[var] = self.sample_with_prob(self.pmf({var: 0}, parents))
                else:
                    temp[var] = evidence[var]
                    weight *= self.pmf({var: evidence[var]}, parents)
            temp['weight'] = weight
            samples = samples.append(temp, ignore_index=True)

            weight_sum = samples['weight'].sum()

            for var, variable in query.items():
                samples = samples[samples[var] == variable]

            query_weight_sum = samples['weight'].sum()
            return query_weight_sum/weight_sum
 

    def gibbs_sample(self, query, evidence, num_iter, num_burnin):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            gibbs samples
        """
        state = {var: (np.random.random() < 0.5) * 1 for var in self.G.keys() if var not in evidence.keys()}
        for v in evidence.keys():
            state[v] = evidence[v]
        samples = pd.DataFrame(columns=list(state.keys()), dtype=int)

        changables = [var for var in self.G.keys() if var not in evidence.keys()]
        for i in range(num_iter + num_burnin):
            for var in changables:
                sub_dict = {key: state[key] for key in state.keys() if key != var}
                state[var] = self.sample_with_prob(self.pmf({var: 0}, sub_dict))
            if i >= num_burnin:
                samples = samples.append(state, ignore_index=True)

        n = samples.shape[0]
        for var, variable in query.items():
            samples = samples[samples[var] == variable]
        m = samples.shape[0]
        
        return m/n            

    
    def topological_sort_util(self, v, visited, stack):
        visited[v] = True
        for l in self.G[v][0]:
            if visited[l] == False:
                self.topological_sort_util(l, visited, stack)
        stack.insert(0,v)


    def topological_sort(self):
        """
            This function wants to make a topological sort of the graph and set the topological_order parameter of the class.

            Parameters
            ----------
            node:
                the list of nodes
            visited:
                the list of visited(1)/not visited(0) nodes

        """
        visited = {}
        for v in self.G :
            visited[v] = False
        stack = []
        for l in self.G:
            if visited[l] == False:
                self.topological_sort_util(l, visited, stack)
        return stack        
        

    def set_topological_order(self):
        """
            This function calls topological sort function and set the topological sort.
        """
        self.topological_order = self.topological_sort()
        pass   
            
    def all_parents_visited(self, node, visited) -> bool:
        """
            This function checks if all parents are visited or not?

            Parameters
            ----------
            node:
                the list of nodes
            visited:
                the list of visited(1)/not visited(0) nodes

            Return
            ----------
            return True if all parents of node are visited, False otherwise.
        """
        for p in self.G[node][1]:
            if visited[p] == False:
                return False
        return True