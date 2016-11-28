import pandas as pd
import numpy as np
import networkx as nx
import community
import matplotlib.pyplot as plt


def q4():
    make_sililarity_files()
    cluster_sp()


def make_sililarity_files():
    data = get_data()
    pscarddf = data[['PS ID', 'CARD ID', 'TRANSACTION ID']].groupby(['PS ID', 'CARD ID']).count()
    psidslist = pscarddf.index.levels[0].values
    cardidslist = pscarddf.index.levels[1].values
    pscardarr = dataframe_to_matrix(pscarddf, 'TRANSACTION ID', len(psidslist), len(cardidslist), 1001, 10001)
    np.savetxt('datasets/pscardarr.csv', pscardarr, delimiter=',')
    similaritymatrix = compute_jaccard(pscardarr)
    np.savetxt('datasets/similaritymatrix.csv', similaritymatrix, delimiter=',')


def cluster_sp():
    similaritymatrix = get_matrix("datasets/similaritymatrix.csv")
    graph = add_graph(similaritymatrix, 1000)
    dendo = get_dendrogram(graph)
    bestpartition = best_partition(graph)
    result = dictionaries_to_dataframe(bestpartition, dendo[0], 'SP ID', 'State', 'City')
    result.to_csv('datasets/result.csv', sep=',', index=False)
    cities_graph = community.induced_graph(dendo[0], graph).edges(data='weight')
    list_to_csv(cities_graph, 'cities_graph.csv')
    states_graph = community.induced_graph(bestpartition, graph).edges(data='weight')
    list_to_csv(states_graph, 'states_graph.csv')


def get_data():
    return pd.read_csv('datasets/_4.csv')


def get_matrix(path):
    return np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)


def dataframe_to_matrix(dataframe, columnname, rowsnum, colsnum, psstartidx, cardstartidx):
    print('dataframe_to_matrix')
    resultarray = np.zeros((rowsnum, colsnum), dtype=np.int8)
    percent = 0
    for i in range(0, rowsnum):
        for j in range(0, colsnum):
            psindex = i + psstartidx
            cardindex = j + cardstartidx
            index = (psindex, cardindex)
            if index in dataframe.index:
                dataframe.get_value(index, columnname)
                resultarray[i][j] = 1
        if int((i * 100) / rowsnum) > percent:
            percent = int((i * 100) / rowsnum)
            print(percent, '%')
    print('100 %')
    return resultarray


def compute_jaccard(matrix):
    print('compute_jaccard')
    size = len(matrix)
    resultmatrix = np.zeros((size, size), dtype=np.float64)
    index = 1
    percent = 0
    for i in range(0, size):
        resultmatrix[i][i] = 1
        ilist = matrix[i]
        for j in range(index, size):
            jlist = matrix[j]
            intersection = sum(np.multiply(ilist, jlist))
            union = sum(ilist) + sum(jlist) - intersection
            if union != 0:
                similarity = intersection / union
                resultmatrix[i][j] = similarity
        index += 1
        if int((i * 100) / size) > percent:
            percent = int((i * 100) / size)
            print(percent, '%')
    print('100 %')
    return resultmatrix


def add_graph(matrix, scale):
    size = len(matrix)
    graph = nx.Graph()
    index = 1
    for i in range(0, size):
        graph.add_node(int(1001 + i))

    for i in range(0, size - 1):
        for j in range(index, size):
            if matrix[i][j] != 0.00:
                graph.add_edge(int(1001 + i), int(1001 + j), weight=int(scale * matrix[i][j]))
        index += 1

    return graph


def best_partition(graph):
    return community.best_partition(graph)


def draw_graph(graph, partition, draw_edges):
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(graph)
    count = 0.
    for com in set(partition.values()):
        count += 1.
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        nx.draw_networkx_nodes(graph, pos, list_nodes, node_size=20,
                               node_color=str(count / size))
    if draw_edges:
        nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.show()


def get_dendrogram(graph):
    return community.generate_dendrogram(graph)


def list_graph_dendrograms(graph):
    dendo = get_dendrogram(graph)
    print('dendogram_len: ', len(dendo))
    for level in range(0, len(dendo)):
        print("partition at level", level,
              "is", community.partition_at_level(dendo, level))


def add_list_to_dataframecol(dataframe, colname, mylist):
    dataframe[colname] = pd.Series(mylist, index=dataframe.index)
    return dataframe


def dictionary_to_dataframe(dict, col1, col2):
    return pd.DataFrame(list(sorted(dict.items())), columns=[col1, col2])


def dictionaries_to_dataframe(dict1, dict2, col1, col2, col3):
    df1 = dictionary_to_dataframe(dict1, col1, col2)
    col3list = list(dictionary_to_dataframe(dict2, col1, col3)[col3].values)
    return add_list_to_dataframecol(df1, col3, col3list)


def list_to_csv(my_list, file_name):
    import csv
    with open(file_name, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['i', 'j', 'k'])
        for row in my_list:
            csv_out.writerow(row)

q4()