import numpy as np
from utils import get_individuals_from_graphs
from system_definition import onset_time, beta_data
import networkx as nx
from collections import deque
import random
import pickle
import os
import json
import csv
#from npy_append_array import NpyAppendArray
import numpy as np
from scipy.sparse import csc_matrix
#import time

#random.seed(1)


class DigitalContactTracing:
    '''
    A class that implements digital contact tracing on a real contact network.

    The class loads an existing network and simulates the spread of a virus on
    the network, based on the characteristics of an infectious disease.
    At the same time, a digital contact tracing policy is implemented to try to
    contain the spread of the virus by enforcing isolation and quarantine,
    depending on a policy specification.
    The class keeps track of a number of relevant quantities (mainly, tracing
    efficacy and histories of quarantined individuals).

    Attributes
    ----------
    I: dict
        details of infected people
    isolated: list
        isolated people
    quarantined: dict
        quarantined people
    symptomatic: list
        symptomatic people
    temporal_gap: float
        temporal gap between static networks
    memory_contacts: int
        tracing memory
    max_time_quar: float
        quarantine duration
    contacts: dict
        contacts of each node
    sympt: float
        fraction of symptomatic individuals
    test: float
        fraction of asymptomatics who are detected via random testing
    eps_I: float
        isolation effectivity
    graphs: list
        snapshots of the temporal graph


    Methods
    -------
    __init__(self, graphs, PARAMETERS, eps_I, use_rssi=True)
        Constructor.

    spread_infection(graph, node, new_infected, current_time)
        Propagates the infection from an infected node to its neighbors.

    is_isolated(current_time,node)
        Update the state of a symptomatic node and quarantine its contacts.

    initialize_contacts(graphs)
        Initialize the contacts from the temporal graph.

    initialize_time0()
        Initialize the status of the initial infected people and initial knowledge.

    simulate()
        Run the simulation.

    update_contacts(graph)
        Update the list of traced contacts.

    update_infected(current_time, graph, new_infected)
        Updates the state of the infected nodes.

    update_quarantined(current_time)
        Update the list of quarantined people.
    '''


    def __init__(self, path_to_store,graphs, PARAMETERS, eps_I, eps_T,realization):
        '''
        Constructor.

        The method defines the setup for the simulations.

        Parameters
        ----------
        graphs: list
            snapshots of the temporal graph
        PARAMETERS: dict
            parameters defining the simulation
        eps_I: float
            isolation effectivity
        '''



        self.I = dict()
        self.temporal_gap = PARAMETERS['temporal_gap']
        self.memory_contacts = int(PARAMETERS['memory_contacts'] * 24 * 3600 / self.temporal_gap) # lo trasformo in passi temporali
        self.max_time_quar = PARAMETERS['max_time_quar'] * 24 * 3600 #lo trasformo in secondi
        self.max_time_iso = PARAMETERS['max_time_iso'] # lo lascio in giorni perché poi lo devo confrontare con tau (che è in giorni)
        self.sympt = PARAMETERS['symptomatics']
        self.test = PARAMETERS['testing']
        self.eps_I = eps_I
        self.eps_T = eps_T
        self.graphs = graphs
        self.recov_time = PARAMETERS['recov_time']

        self.nb_clusters = PARAMETERS['nb_clusters']
        self.cluster_size = PARAMETERS['cluster_size']
        self.net = PARAMETERS['net']
        if self.net == 'DTU':
            self.nb_nodes = 675
        else:
            self.nb_nodes = self.nb_clusters*self.cluster_size

        self.nb_ideas = PARAMETERS['nb_ideas']
        self.know_old = np.full((self.nb_nodes, self.nb_ideas), 0)
        self.idea_count = np.full((self.nb_nodes, self.nb_ideas), 0)
        self.know_threshold = PARAMETERS['k_threshold']
        self.A_24 = [[0 for x in range(self.nb_nodes)] for y in range(self.nb_nodes)] # adjacency of last 24h for knowledge
        #self.sameK = all(el == self.know_threshold[0] for el in self.know_threshold) # True if all K are the same
        #self.all_ideas = np.full([self.nb_clusters,self.nb_ideas], 0)
        self.path_to_store = path_to_store
        self.realization = realization



        nodes_list = get_individuals_from_graphs(graphs)

        # get infected at time 0
        self.pat0 = random.choice(list(graphs[0].nodes())) # 1 initial random infected among existing nodes at time 0


        self.contacts = self.initialize_contacts(graphs, self.memory_contacts)

        self.initialize_time0()



    def initialize_time0(self):
        '''
        Initialize the status of the initial infected people and initial
        knowledge.
        self.isolated, self.symptomatic, and self.recovered elements are 0 if
        the corresponding node is not in that state and 1 vice versa.
        self.quarantined elements are 0 if the corresponding node is not
        quarantined and equal to the time at which quarantine began if the node
        is quarantined.
        '''
        self.isolated = np.full(self.nb_nodes,0)
        self.quarantined = np.full(self.nb_nodes,0)
        self.symptomatic = np.full(self.nb_nodes,0)
        self.recovered = np.full(self.nb_nodes,0)

        # infected:
        tau = np.random.uniform(0, 10)  # fra 0 e 10 giorni
        self.I[self.pat0] = {'tau': tau,
                        'to': onset_time(symptomatics=self.sympt, testing=self.test),
                        }

        # knowledge:
        if self.net == 'synth':
            if self.nb_ideas >= self.nb_clusters:
                idea_n = 0
                for n in range(self.nb_clusters):
                    for i in range(int(self.nb_ideas/self.nb_clusters)):
                        node0 = random.randrange(n*self.cluster_size, (n+1)*self.cluster_size) #node0 has info n
                        self.know_old[node0][idea_n] = 1
                        #self.all_ideas[int(node0/self.cluster_size)][idea_n] += 1

                        idea_n += 1
            elif self.nb_ideas < self.nb_clusters:
                for idea_n in range(self.nb_ideas):
                    cluster_n = idea_n # idea0 in cluster0, idea1 in cluster1, ...
                    node0 = random.randrange(cluster_n*self.cluster_size, (cluster_n+1)*self.cluster_size) #node0 has info idea_n
                    self.know_old[node0][idea_n] = 1
                    #self.all_ideas[int(node0/self.cluster_size)][idea_n] += 1


        elif self.net == 'DTU':
            part_d = np.load('partition_d_10clusters.npy',allow_pickle=True)[0]
            clusters = []
            for cluster_id in range(max(part_d.values()) + 1):
                clusters.append([k for k,v in part_d.items() if v == cluster_id])
            idea_n = 0
            for n in range(self.nb_clusters):
                for i in range(int(nb_ideas/nb_clusters)):
                    node0 = random.choice(clusters[n])
                    self.know_old[node0][idea_n] = 1
                    idea_n += 1


    #@staticmethod
    def initialize_contacts(self,graphs, memory_contacts):
        ''''
        Initialize the contacts from the temporal graph.

        It creates a dictionary where the values of key idx is the list of
        contacts of the node idx. Each list is a deque , so elements are pop-ed
        from the left if the length exceeds self.memory_contacts.

        Parameters
        ----------
        graphs: list
            list of static graphs
        memory_contacts: int
            tracing memory

        Returns
        ----------
        contacts: dict
            contacts of each node
        '''

        contacts = dict()
        for i in range(self.nb_nodes):
            contacts[i] = deque(maxlen=memory_contacts)
        return contacts


    def simulate(self):
        ''''
        Run the simulation.

        It runs the simulation on the temporal network.

        '''
        # Initialize the simulation time
        current_time = 0

        # Initialize new_infected
        new_infected = np.full(self.nb_clusters, 0) #array of dimension nb_clusters full of zeros
        new_infected[int(self.pat0/self.cluster_size)] += 1

        eff_edges_t = []
        act_inf_t = []
        isolate_t = []
        recover_t = []
        quarantine_t = []
        quar_inf_t = []
        nb_ideas_per_node = []
#        nb_nodes_per_idea = []
        # Loop over the temporal snapshots
        for time_idx, graph in enumerate(self.graphs):

            # Update the tracing contacts
            self.update_contacts(graph, time_idx)

            # Update the state of nodes that are currently in quarantine
            self.update_quarantined(current_time)

            # Update the state of the infected nodes
            self.update_infected(current_time, graph, new_infected)

            # Remove nodes from isolation when they have terminated their isolation
            self.update_isolated(current_time)

            # Update last 24h adjacency for knowledge
            eff_edgelist = list(graph.edges)
            for edge in graph.edges: #
                if self.quarantined[edge[0]]==0 and self.quarantined[edge[1]]==0 and self.isolated[edge[0]]==0 and self.isolated[edge[1]]==0:
                    self.A_24[edge[0]][edge[1]] += 1
                    self.A_24[edge[1]][edge[0]] += 1
                else:
                    eff_edgelist.remove(edge)
            #path = self.path_to_store + '/eff_edgelist/'
            #if not os.path.exists(path):
            #    os.makedirs(path)
            #save_on_npy(path + '%d_eff_edgelist_time_%d.npy'%(self.realization,time_idx),eff_edgelist)
            eff_edges_t.append(eff_edgelist)

            # Update of knowledge contagion every 24h
            if time_idx % (24*3600/self.temporal_gap) == 24*3600/self.temporal_gap-1:

                self.social_complex_contagion()

                self.A_24 = [[0 for x in range(self.nb_nodes)] for y in range(self.nb_nodes)] # reset A_24


            ''' SERVE SOLO SE VOGLIO USARE K DIVERSI PER IDEE DIVERSE
            if time_idx % (15*24*3600/self.temporal_gap) == 0: # ogni 15 giorni
                if self.net == 'synth':
                    if self.sameK == False:
                        #print(time_idx,self.all_ideas)
                        self.all_ideas_id_30.append(self.all_ideas.copy())
            '''

           # Update the histories of symptomatics, isolated, infected, ...
            act_inf = np.full(self.nb_nodes,0)
            for node in self.I:
                if self.quarantined[node]==0 and self.isolated[node]==0 and self.recovered[node]==0:
                    act_inf[node] = 1



            # update comparments in real time
            act_inf_t.append(act_inf)
            isolated_tmp = self.isolated.copy()
            isolate_t.append(isolated_tmp)
            recovered_tmp = self.recovered.copy()
            recover_t.append(recovered_tmp)
            if self.eps_T > 0:
                quar = np.full(self.nb_nodes,0)
                quar[self.quarantined > 0] = 1
                quar_inf = quar.copy()
                for node in range(self.nb_nodes):
                    if quar[node]==1 and node not in self.I:
                        quar_inf[node] = 0
                quarantine_t.append(quar)
                quar_inf_t.append(quar_inf)
            # nb of ideas that each node has acquired (array nb_nodes long)
            nb_ideas_per_node.append(np.sum(self.know_old,axis=1))
            # nb of nodes that each idea has reached (array nb_ideas long)
#            nb_nodes_per_idea.append(np.sum(self.know_old,axis=0))


            # save in real time
            #save_on_npy(self.path_to_store + '/%d_act_inf.npy'%self.realization,act_inf)
            #save_on_npy(self.path_to_store + '/%d_isolated.npy'%self.realization,self.isolated)
            #save_on_npy(self.path_to_store + '/%d_recovered.npy'%self.realization,self.recovered)
            #if self.eps_T > 0:
            #    save_on_npy(self.path_to_store + '/%d_quarantined.npy'%self.realization,quar)
            #nb_ideas_per_node = np.sum(self.know_old,axis=1) # nb of ideas that each node has acquired (array nb_nodes long)
            #nb_nodes_per_idea = np.sum(self.know_old,axis=0) # nb of nodes that each idea has reached (array nb_ideas long)

            #filename = self.path_to_store + '/%d_ideas_per_node.npy'%self.realization
            #with NpyAppendArray(filename) as npaa:
            #    npaa.append(nb_ideas_per_node)

            #filename = self.path_to_store + '/%d_nodes_per_idea.npy'%self.realization
            #with NpyAppendArray(filename) as npaa:
            #    npaa.append(nb_nodes_per_idea)

            # Advance the simulation time
            current_time = current_time + self.temporal_gap

        ## WRITE realization dynamics
        # # write effective_edgelist over time
        path = self.path_to_store + '/eff_edgelist/'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + '%d_eff_edgelist.npy'%(self.realization),
                np.asanyarray(eff_edges_t,dtype=object))
        # write compartments over time
        np.save(self.path_to_store + '/%d_act_inf.npy'%self.realization,csc_matrix(np.array(act_inf_t)))
        np.save(self.path_to_store + '/%d_isolated.npy'%self.realization,csc_matrix(np.array(isolate_t)))
        np.save(self.path_to_store + '/%d_recovered.npy'%self.realization,csc_matrix(np.array(recover_t)))
        if self.eps_T > 0:
            np.save(self.path_to_store + '/%d_quarantined.npy'%self.realization,csc_matrix(np.array(quarantine_t)))
            np.save(self.path_to_store + '/%d_quar_inf.npy'%self.realization,csc_matrix(np.array(quar_inf_t)))
        # write ideas over time
        np.save(self.path_to_store + '/%d_ideas_per_node.npy'%self.realization,csc_matrix(np.array(nb_ideas_per_node)))
#        np.save(self.path_to_store + '/%d_nodes_per_idea.npy'%self.realization,csc_matrix(np.array(nb_nodes_per_idea)))




    def update_infected(self, current_time, graph, new_infected):
        ''''
        Updates the state of the infected nodes.

        The method updates the state of each infected node by advancing in time
        its information and by checking if it has become symptomatic.
        Moreover, an infected node may be isolated according to the isolation
        efficiency: If it is not isolated, it spread the infection to its
        neighbors; If it is isolated, the tracing policy is enforced on its
        contacts. If recov_time days have passed since it has been infected it becomes
        recovered.

        Parameters
        ----------
        current_time: float
            the absolute time since the beginning of the simulation
        graph: networkx.classes.graph.Graph
            snapshots of the temporal graph
        new_infected: list
            nodes that are infected at the current time
        '''

        for node in list(self.I.keys()).copy():
            current_to = self.I[node]['to']
            self.I[node]['tau'] += self.temporal_gap / (3600 * 24)

            if self.quarantined[node]==0 and self.isolated[node]==0 and self.recovered[node]==0:
                if self.I[node]['tau'] >= self.recov_time:
                    self.recovered[node] = 1
                    #del self.I[node]

                else:
                    if self.symptomatic[node] == 1:
                        self.spread_infection(graph, node, new_infected, current_time)
                    elif self.symptomatic[node] == 0:
                        if current_to > current_time:
                            self.spread_infection(graph, node, new_infected, current_time)
                        elif current_to <= current_time:
                            r = np.random.uniform(0, 1)
                            if r <= self.eps_I:
                                #assert node not in self.isolated
                                self.is_isolated(current_time, node)
                            elif r > self.eps_I:
                                self.spread_infection(graph, node, new_infected, current_time)
                            self.symptomatic[node] = 1




    def is_isolated(self, current_time, node):
        ''''
        Isolate a symptomatic node and quarantine its contacts.

        The method updates the state of a node which is found infected, i.e.,
        it is isolated and its contacts are quarantined.
        First, the node is added to the list of isolated nodes and removed from
        the list of infected (if it is not quarantined) or from the list of
        quarantined (if it is quarantined).
        Second, if the node is adopting the app, the list of its past contacts
        which were 'at risk' is processed and each node is quarantined.

        Parameters
        ----------
        current_time: float
            the absolute time since the beginning of the simulation
        node: int
            a node in the snapshot
        '''

        self.isolated[node] = 1
        if self.quarantined[node] != 0:
            self.quarantined[node] = 0


        # metto in quarantena i contatti del nodo isolato:
        C = []
        C = [item for sublist in self.contacts[node] for item in sublist]
        C = np.unique(C)
        nb_elements_to_keep = int(np.round(len(C)*self.eps_T))
        C = set(random.sample(set(C), nb_elements_to_keep)) # keep only a fraction eps_T of the contacts

        for m in C:
            if self.quarantined[m]==0 and self.isolated[m]==0 and self.recovered[m]==0:
                self.quarantined[m] = current_time



    def spread_infection(self, graph, node, new_infected, current_time):
        ''''
        Propagates the infection from an infected node to its neighbors.

        The method loops over the neighbors of an infected node and selectively
        propagates the infection (i.e., add the neighbors to the list of
        infected nodes).
        To decide if the infection is propagated or not, the method checks the
        infection probability beta_data.

        Parameters
        ----------
        graph: networkx.classes.graph.Graph
            snapshots of the temporal graph
        node: int
            a node in the snapshot
        new_infected: list
            nodes that are infected at the current time
        current_time: float
            the absolute time since the beginning of the simulation
        '''

        if node in graph:
            for m in graph.neighbors(node):
                if m not in self.I and self.quarantined[m] == 0:
                    pp = beta_data(self.I[node]['tau'])  # probability of contagion node --> m

                    rr = np.random.uniform(0, 1)

                    if rr < pp:  # the contagion of m happens
                        to = onset_time(symptomatics=self.sympt, testing=self.test)
                        self.I[m] = {'tau': 0,
                                     'to': current_time + to,
                                     }
                        if self.net == 'synth':
                            new_infected[int(m/self.cluster_size)] += 1



    def update_contacts(self, graph, time_idx):
        ''''
        Update the list of traced contacts.

        The method uses the current snapshot graph to update the list contacts,
        which stores for each node a list of its contacts.
        For each node, the methods finds the neighbors which are 'at risk'
        according to the policy, and adds them to the list of contacts.
        Moreover, the past contacts which are older than the tracing memory are
        discarded.

        Parameters
        ----------
        graph: networkx.classes.graph.Graph
            snapshots of the temporal graph
        time_idx: int
            index of the current time in the list of time instants
        '''

        for node in self.contacts:
            if node in self.graphs[time_idx]:
                res = list(self.graphs[time_idx].neighbors(node))
            else:
                res = []
            #res = self.policy(time_idx, node)
            self.contacts[node].append(res) # This is a deque, so elements are
                                            # pop-ed from the left if the length
                                            # exceeds self.memory_contacts


    def update_quarantined(self, current_time):
        ''''
        Update the list of quarantined people.

        The method finds the nodes who have completed the quarantine time, and
        removes them from the list of quarantined nodes. Nodes who are infected
        at this stage are added to the list of infected nodes.
        The method also checks if a nodes becomes symptomatic while in
        quarantine, and in this case the tracing policy is enforced on its
        contacts.

        Parameters
        ----------
        current_time: float
            the absolute time since the beginning of the simulation
        '''

        # tolgo dalla quarantena chi l'ha finita
        for node in self.contacts.keys():
            if self.quarantined[node] != 0 and (current_time - self.quarantined[node]) >= self.max_time_quar:
                self.quarantined[node] = 0
                if node in self.I:
                    if self.I[node]['tau'] >= self.recov_time and self.recovered[node]==0:
                        self.recovered[node] = 1


        # passo all'isolamento chi inizia a presentare i sintomi
        quar_temp = np.nonzero(self.quarantined)[0]
        for node in quar_temp:
            if node in self.I:
                current_to = self.I[node]['to']
                if current_to < current_time:  # symptom onset
                    self.is_isolated(current_time, node)



    def update_isolated(self, current_time):
        ''''
        Update the list of isolated people.

        The method finds the nodes who have completed the isolation time, and
        removes them from the list of isolated nodes (they become recovered).

        '''

        for node in np.nonzero(self.isolated)[0]: # prendo solo i nodi isolati
            if self.I[node]['tau'] >= self.max_time_iso:
                self.isolated[node] = 0
                self.recovered[node] = 1









    def social_complex_contagion(self):

        ''''
        know_old is the current knowledge, know_new is a copy of it but will be
        updated: idea_count counts the nb of interactions of a node with nodes
        having a particular idea (different nodes or same node multiple times).
        When the count reaches a threshold the node gets the idea. This updates
        know_new and then it becomes know_old.
        '''

        #self.new_ideas = np.full([self.nb_clusters,self.nb_ideas], 0)# ELIMINARE?
        know_new = self.know_old.copy()

        for node in range(self.nb_nodes):
            if self.quarantined[node]==0 and self.isolated[node]==0:
                for idea_n in range(self.nb_ideas):
                    if not self.know_old[node][idea_n]:
                        # acquire knowledge
                        for neigh_node in [i for i, e in enumerate(self.A_24[node]) if e != 0]: # contacts of last 24h
                            if self.know_old[neigh_node][idea_n] and self.quarantined[neigh_node]==0 and self.isolated[neigh_node]==0:
                                self.idea_count[node][idea_n] += self.A_24[node][neigh_node] #add to count nb of interactions of last 24h with people with idea_n
                        if self.idea_count[node][idea_n] >= self.know_threshold[idea_n]:
                            know_new[node][idea_n] = 1

        self.know_old = know_new.copy()


def save_on_csv(filename, variable_list, writing_operation):
    with open(filename, writing_operation) as csvfile:
        writer = csv.writer(csvfile)
        try:
            [writer.writerow(s) for s in variable_list]
        except:
            writer.writerow(variable_list)

def save_on_npy(filename, vect):
    if not os.path.exists(filename):
        np.save(filename,[vect])
    else:
        old = list(np.load(filename,allow_pickle=True))
        old.append(vect)
        np.save(filename,old)
