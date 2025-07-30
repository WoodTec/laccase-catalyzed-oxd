def plot_label_dist(train_labels, seed, colormap, title):
    # plot distribution of positive and negative examples in train:
    fig = plt.figure(figsize=(4, 4))
    ax = sns.countplot(pd.DataFrame(train_labels), x='Oxd', hue='Oxd', stat="percent", palette=colormap, alpha=0.9, edgecolor='black', linewidth=0.25, legend=False)
    plt.suptitle(title)
    for i in ax.containers:
        ax.bar_label(i, fmt='{:,.2f}%')
        ax.set_ylim(0,100)
    plt.tight_layout()
    plt.show()
    #plt.close(fig)
    

    
def define_type(col):
    
    '''
    identify type of a descriptor (column)
    '''
    
    res = map(lambda x: x % 1 == 0, [min(col), np.percentile(col, 25), np.percentile(col, 50), np.percentile(col, 75), max(col), sum(col)])
    if all(list(res)):
        return np.int64
    else:
        return np.float64



def drop_col(data):
    
    '''
    drop uninformative descriptors (low variation)
    '''
    
    dropped = []
    for col in data.columns.to_list():
        rel_freqs = data[col].value_counts(normalize=True)

        if max(rel_freqs)>0.95:
            data = data.drop(col, axis=1)
            dropped.append(col)

    print(f'The following columns have been dropped:') 
    print(*dropped, sep=',')
    return data




def check_freq(data, col, thres=0.01):
    
    '''
    auxiliary function
    returns True if the feature requires binning, that is, if for at least one category the frequency is < threshold value
    '''
    
    if col.empty:
        raise ValueError("Column is empty.")
    if col.nunique() == 1:
        return False
    col = pd.Series(col)
    freq = data.groupby([col]).size().reset_index(name='counts')
    freq['rel_freq'] = freq['counts']/sum(freq['counts'])
    if sum(freq['rel_freq']<thres) > 0: 
        return True
    else:
        return False



def drop_dups(data):
    '''
    cast the substrate names to lower case
    drop duplicates if any and drop the name column  
    '''
    if 'Name' in data.columns:
        data['Name'] = data['Name'].str.lower()
        data = data.drop_duplicates().reset_index(drop=True)
        data.drop(['Name'], axis=1, inplace=True)
    return data



def set_coltypes(data):
    '''
    function that takes in a dataframe
    * replaces missing labels with 0
    * changes some dtypes (for Qindex, Wap and CENT) manually
    returns modified dataframe
    '''
    if 'Oxd' in data.columns:
        data['Oxd'].fillna(0, inplace=True) 
        data['Oxd'] = data['Oxd'].round().astype('int64')
    cols_tp = list(map(lambda x: data.iloc[:, x], np.arange(0, len(data.columns))))
    convert_dict = dict(zip(data.columns, list(map(define_type, cols_tp)))) 
    convert_dict.update(Qindex=np.float64, Wap=np.float64, CENT=np.float64) 
    data = data.astype(convert_dict)
    
    return data


def merge_bins(col, thres = 0.05):
    
    '''
    function that merges adjascent bins of a descriptor in case one of the bins contains too few observations
    therefore provides coarser binning with low information loss
    '''
    
    # initialize bins: each value in a separate bin
    bin_data, bin_edges = pd.cut(col, bins=sorted([-np.inf]+list(col.sort_values().unique())), retbins=True, include_lowest=True)
    # count instances in bins:
    freq = bin_data.value_counts().reset_index()
    freq.rename(columns={ freq.columns[0]: 'bin' }, inplace = True)
    freq['rel_freq'] = freq['count']/sum(freq['count'])
    freq = pd.DataFrame(freq).sort_values(by=['bin'])
    freq['edges'] = freq['bin'].values.categories.left # lower bounds of the bin intervals
    bins_ = list(bin_edges)
    
    if len(bins_)>0:
        # take the corresponding rel_freq and check the condition:
        while any(freq['rel_freq'].to_numpy() < thres):
            rmv =  max(freq[freq['rel_freq'] < thres]['edges']) # the eldest bucket with low rel_freq
            # adjust binning by excluding the above edge:
            try:  
                bins_.remove(rmv)
            except:   
                print('Something went wrong!')
                pass
            bin_data, bin_edges = pd.cut(col, bins=bins_, retbins=True, include_lowest=True)  
            
            # recompute frequencies with new bins: 
            freq = bin_data.value_counts().reset_index()
            freq.rename(columns={ freq.columns[0]: 'bin' }, inplace = True)
            freq['rel_freq'] = freq['count']/sum(freq['count'])
            freq = pd.DataFrame(freq).sort_values(by=['bin'])
            freq['edges'] = freq['bin'].values.categories.left
            
        # as we remove the left edge, if the very first bin is combined the very first edge is lost, after the binning is finished, 
        # the bin_edges need to be adjusted by replacing the very left edge by -inf
        bins_[0] = -np.inf 
        bins_[-1] = +np.inf 
        bin_data, bin_edges = pd.cut(col, bins=bins_, retbins=True, include_lowest=True)
        return bin_data      
    else: 
        raise RuntimeError(f"Failed to remove bin edge: {rmv}")
    
    
def bin_features(data_train):
    
    '''
    auxiliary function used for learning the Bayesian Network using bnlearn
    - bins the continuous variables before feeding them into the bnlearn algorithm
    - uses the following map that generally works well for the lac-data:
    a column has >=48 unique values --> 18 bins
    32 < unique values <= 48 --> 15 bins
    24 < unique values <= 32 --> 12 bins
    16 < unique values <= 24 --> 9 bins
    3 < unique values <= 16 --> 4 bins
    <= 3 unique values --> original structure is retained
    '''
    
    unq_ = data_train.nunique()
    
    c4 = unq_[(unq_ > 3) & (unq_ <= 16)].index.to_list()
    c9 = unq_[(unq_ > 16) & (unq_ <= 24)].index.to_list()
    c12 = unq_[(unq_ > 24) & (unq_ <= 32)].index.to_list()
    c15 = unq_[(unq_ > 32) & (unq_ <= 48)].index.to_list()
    c18 = unq_[(unq_ > 48)].index.to_list()
    data_dscr_train = data_train.copy()
    nbins = [4, 9, 12, 15, 18]
    ccols = [c4, c9, c12, c15, c18]
        
    for i in range(len(nbins)):
        dscr = KBinsDiscretizer(n_bins=nbins[i], encode='ordinal', strategy='quantile')
        dscr.fit(data_train)
        data_dscr_train[ccols[i]] = dscr.fit_transform(data_dscr_train[ccols[i]])
            
    return data_dscr_train


def plot_bn_graph(relevant_edges, seed, reduced, colormap):
    
    '''
    auxiliary function for plotting the whole BN graph or a Markov blanket if reduced=True
    groups of discriptors are depicted in different colors (hardcoded)
    
    args:
    relevant edges: either the whole graph or the reduced graph
    seed: used for naming the plot when it is saved in the directory
    reduced: if None, the whole network is plotted, else the Markov Blanket 
    
    outputs: a networkx plot
    '''
    
    # create a graph object and add edges
    G = nx.DiGraph()
    G.add_edges_from(relevant_edges)

    # color the nodes by groups
    node_colors = []
    target_node = 'Oxd'
    for node in G.nodes:
        if node == target_node:
            node_colors.append(colormap[0])  # tgt node
        elif node in ['nBM', 'nTB', 'nAB', 'nH', 'nC', 'nN', 'nO', 'nP', 'nS', 'nCL', 'nHM', 'nHet', 'nCsp3', 'nCsp2', 'nCsp', 'nCIC', 'nCIR']:
            node_colors.append(colormap[1]) 
        elif node in ['nR05', 'nR06', 'nR09', 'nR10', 'nR11', 'nR12', 'Psi_i_A', 'Psi_i_t', 'Psi_i_0d', 'Psi_i_1s']:
            node_colors.append(colormap[2]) 
        elif node in ['C%', 'N%', 'O%', 'X%', 'PW2', 'PW3', 'PW4', 'PW5']:
            node_colors.append(colormap[3]) 
        elif node in ['P_VSA_m_1', 'P_VSA_m_2', 'P_VSA_m_4', 'P_VSA_v_2', 'P_VSA_v_3', 'P_VSA_e_3', 'P_VSA_i_1', 'P_VSA_i_2', 'P_VSA_i_3', 'P_VSA_s_1', 'P_VSA_s_3', 'P_VSA_s_4', 'P_VSA_s_6']:
            node_colors.append(colormap[4]) 
        elif node in ['nCs', 'nCt', 'nCq', 'nCrs', 'nCrq', 'nCconj', 'nR=Cs', 'nR=Ct', 'nRCOOR', 'nRCO', 'nCONN', 'nRNH2', 'nRNR2', 'nRCN', 'nRNO', 'nC=N-N<', 'nROR', 'nRSR', 'nS(=O)2', 'nSO3', 'nSO2N', 'nCXr=', 'nCconjX', 'nTriazoles', 'nHDon']:
            node_colors.append(colormap[5]) 
        else:
            node_colors.append(colormap[6])  # all other nodes

    # draw the whole or reduced graph and save
    pos = nx.spiral_layout(G)  
    if reduced:
        fig = plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color=node_colors, font_size=8, arrowsize=10, alpha=0.85)
        plt.title("Bayesian Network ('Oxd' Markov Blanket)")
        #plt.savefig(f'plots/markov_blanket_{seed}.png', bbox_inches='tight') 
    else:
        fig = plt.figure(figsize=(18, 16))
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color=node_colors, font_size=10, arrowsize=10, alpha=0.85)
        plt.title("Bayesian Network")
        #plt.savefig(f'plots/bayes_net_{seed}.png', bbox_inches='tight') 
    #plt.close(fig) 
    plt.show()



def select_edges(best_bn):
    
    '''
    function for constructing the Markov Blanket from the learned Bayesian Network
    
    args:
    * best_bn: learned Bayesian Network (bnlearn object)
    
    outputs: edges constituting the Markov Blanket 
    '''
    
    # get the Markov blanket of the target node:
    markov_blanket = best_bn.get_markov_blanket('Oxd')

    # Identify relevant edges
    relevant_edges = []
    for edge in best_bn.edges():
        if edge[0] in markov_blanket or edge[1] in markov_blanket:
            relevant_edges.append(edge)
  
    return relevant_edges



    
def plot_bins(ord_features, seed, binned):
    
    '''
    auxiliary function for plotting the ordinals prior to binning and after: 
    allows to check how the binning function works
    args:
    ord_features: list of 
    '''
    
    l = len(ord_features.columns)//3
    fig, axn = plt.subplots(l, 3, figsize=(8, l*3), sharex=False, sharey=True)


    #cmap = colormap*(len(ord_features.columns)//7)
    n_ = len(ord_features.columns)
    cmap_ = mpl.colormaps['RdYlBu']
    # take colors at regular intervals spanning the colormap.
    cmap = cmap_(np.linspace(0, 1, n_))
    
    
    for i, ax in enumerate(axn.flat):
        labs = range(min(ord_features.iloc[:, i]), max(ord_features.iloc[:, i])+1, 1)
        g = sns.histplot(ord_features.iloc[:, i], discrete=True, stat='probability', color=cmap[i], edgecolor='darkslategray', ax=ax)  # Use the existing figure
        g.set_title(f'{ord_features.columns.tolist()[i]}', fontsize=9)
        g.set_xticks(range(len(labs))) 
        g.set_xticklabels(labs, fontsize=6) 
        g.set_xlabel(None)
        g.set_yticklabels(np.round(np.linspace(0, 1, 6),1), fontsize=6)
           
    fig.suptitle(f'Distribution of ordinal descriptors', fontsize=11)
    fig.tight_layout()
    #plt.savefig(f'plots/ord_binned_{binned}_{seed}.png', bbox_inches='tight')
    #plt.close(fig)
    plt.show()