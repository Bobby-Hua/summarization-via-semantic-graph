import json
import re
import unidecode
from tqdm import tqdm
from natsort import natsorted
import networkx as nx
import jsonlines
from networkx.algorithms.components.connected import connected_components
from collections import Counter
import argparse

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_text_file")
    argparser.add_argument("jamr_dir_out")
    argparser.add_argument("coref_output_file")
    argparser.add_argument("graph_output_file")

    args = argparser.parse_args()
    input_text_json = args.input_text_file
    jamr_output_dir = args.jamr_dir_out
    coref_output_file = args.coref_output_file
    graph_output_file = args.graph_output_file

    def in_range(i, j):
        start_i, end_i = i
        start_j, end_j = j
        if start_i >= start_j and end_i <= end_j:
            return True
        return False

    def find_node_by_jamr_format(index, jamr_nodes):
        node = []
        jamr_nodes = jamr_nodes.split("+")
        for i in jamr_nodes:
            node.append(jamr_format_to_node_arr[index][i])
        return node

    with open(input_text_json, encoding="utf-8") as file:
        dialogs = json.load(file)

    num_dialog = []
    dialog = []
    count = 0
    for i in dialogs:
        if len(i["dialogue"]) != 0:
            dialog.append(i["dialogue"].replace("\r\n", "\n"))
            if i["dialogue"].endswith("\n"):

                num_dialog.append(i["dialogue"].count("\n"))
            else:
                num_dialog.append(i["dialogue"].count("\n") + 1)

        else:
            print(count)
        count += 1

    speakers_id = []
    speakers_names = []
    for dialog in dialogs:
        dialog_list = dialog["dialogue"].split("\n")
        speakers = []
        for idx, val in enumerate(dialog_list):
            if len(val) != 0:
                speakers.append(re.search(r"^.+?:", val).group(0)[:-1].strip())
                lst = list(dict.fromkeys(speakers))
                dct = dict(zip(lst, range(len(lst))))
                speakers_a = []
                for i in speakers:
                    speakers_a.append(dct[i] + 1)
        speakers_id.append(speakers_a)
        speakers_names.append(speakers)

    start_index = 0
    end_index = len(dialogs)

    jamr_lines = []
    for index in range(start_index, end_index):
        with open(
            jamr_output_dir + "/jamr_output_" + str(index), encoding="utf-8"
        ) as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            jamr_lines.append(lines)

    jamr_toks = []
    jamr_alignments = []
    jamr_nodes = []
    jamr_edges = []
    count = 0
    for index, val in enumerate(jamr_lines):
        jamr_nodes_t = []
        jamr_edges_t = []
        jamr_toks.append(val[1])
        if len(val) > 2:
            jamr_alignments.append(val[2])
        else:
            print(index)
            count += 1
        for i in val:
            if i.startswith("# ::node"):
                jamr_nodes_t.append(i)
            if i.startswith("# ::edge"):
                jamr_edges_t.append(i)
        jamr_nodes.append(jamr_nodes_t)
        jamr_edges.append(jamr_edges_t)

    toks_arr = []
    snt_arr = []
    for i in jamr_toks:
        tok = i[8:]
        snt_arr.append(tok)
        toks_arr.append(tok.split())

    snt_to_node_arr = []
    for i in tqdm(jamr_alignments):
        alignment = i[15 : i[: i.rfind("::")].rfind("::")]
        snt_to_node = {}
        for i in alignment.split():
            t = i.split("|")
            snt_to_node[t[0]] = t[1]
        snt_to_node_arr.append(snt_to_node)

    snt_index_to_speaker_arr = []
    for index, val in enumerate(tqdm(speakers_id)):
        snt_index_to_speaker = {}
        speakers_names_copy = speakers_names[index].copy()
        s_i_copy = speakers_id[index].copy()
        for tok_index, tok in enumerate(toks_arr[index]):
            if tok_index + 1 < len(toks_arr[index]) and len(speakers_names_copy) != 0:
                if (
                    speakers_names_copy[0] == tok
                    and toks_arr[index][tok_index + 1] == ":"
                ):
                    speakers_names_copy.pop(0)
                    snt_index_to_speaker[tok_index] = s_i_copy.pop(0)
        snt_index_to_speaker_arr.append(snt_index_to_speaker)

    jamr_align_arr = []
    sort_order_arr = []
    for nodes in jamr_nodes:
        jamr_align = []
        for i in nodes:
            jamr_align.append(i.split("\t"))
        jamr_align_arr.append(jamr_align)
    for i in jamr_align_arr:
        sort_order = []
        for j in i:
            sort_order.append(j[1])
        sort_order_arr.append(sort_order)
    jamr_align_arr_sort_fixed = []
    for index, val in enumerate(jamr_align_arr):
        jamr_align_arr_sort_fixed.append(
            [x for _, x in natsorted(zip(sort_order_arr[index], jamr_align_arr[index]))]
        )

    dot_format_to_index_arr = []
    for index, val in enumerate(jamr_align_arr_sort_fixed):
        dot_format_to_index = {}
        for index2, val2 in enumerate(val):
            dot_format_to_index[val2[1]] = index2 + 1
        dot_format_to_index_arr.append(dot_format_to_index)

    jamr_edges_arr = []
    for index, edges in enumerate(jamr_edges):
        jamr_edges_t = []
        for i in edges:
            jamr_edges_dict = {}
            t = i.split("\t")
            jamr_edges_dict["label"] = t[2]
            jamr_edges_dict["from"] = dot_format_to_index_arr[index][t[4]]
            jamr_edges_dict["to"] = dot_format_to_index_arr[index][t[5]]
            jamr_edges_t.append(jamr_edges_dict)
        jamr_edges_arr.append(jamr_edges_t)

    concepts_list = []
    nodes_list = []

    for index, i in enumerate(jamr_align_arr_sort_fixed):
        concepts = []
        n = []
        e = []
        for j in i:
            jamr_nodes_dict = {}
            concepts.append(j[2])
            jamr_nodes_dict["label"] = j[2]
            jamr_nodes_dict["id"] = dot_format_to_index_arr[index][j[1]]
            n.append(jamr_nodes_dict)
        concepts_list.append(concepts)
        nodes_list.append(n)

    edges_list = jamr_edges_arr
    edges_dict_list = []
    for i in edges_list:
        edges = {}
        for j in i:
            f = j["from"]
            t = j["to"]
            edges[(f, t)] = "-:" + j["label"]
            edges[(t, f)] = "+:" + j["label"]
        edges_dict_list.append(edges)

    jamr_format_to_node_arr = []
    for index, simplified_nodes in enumerate(tqdm(nodes_list)):
        jamr_format_to_node = {}
        index_offset = 0
        for index2, i in enumerate(jamr_align_arr_sort_fixed[index]):
            src_arr = []
            src = jamr_align_arr_sort_fixed[index][index2][2]
            if src == "":
                print("??")
            src_arr.append(src)
            result = re.search(r"^(.+)-\d+$", src)
            if result:
                src = result.groups()[0]

            if src.isascii() == False:
                new_src = ""
                unidecoded = unidecode.unidecode(src)
                if unidecoded == "":
                    src_arr.append("??")
                    src_arr.append("?")
                else:
                    src_arr.append(unidecoded)
                    for char in src:
                        if char.isascii() == False:
                            new_src += "?"
                        else:
                            new_src += char
                    src_arr.append(new_src)

            src_arr.append(src)
            src_arr.append(src.replace("\\", "-"))
            for index3, j in enumerate(simplified_nodes[index_offset + index2 :]):
                tgt = j["label"]
                new_tgt = None
                if tgt.isascii() == False:
                    new_tgt = ""
                    for char in tgt:
                        if char.isascii() == False:
                            new_tgt += "?"
                        else:
                            new_tgt += char
                if "" in src_arr:
                    print("error")
                if j["label"] in src_arr or new_tgt in src_arr:
                    jamr_format_to_node[
                        jamr_align_arr_sort_fixed[index][index2][1]
                    ] = simplified_nodes[index_offset + index2]["id"]
                    break
                else:
                    index_offset += 1
        jamr_format_to_node_arr.append(jamr_format_to_node)

    for index, val in enumerate(jamr_align_arr):
        if len(val) != len(jamr_format_to_node_arr[index]):
            print(index)
            print("error")

    def create_graph(n, e):
        G = nx.Graph()
        for i in n:
            G.add_node(i["id"])
        for j in e:
            f = j["from"]
            t = j["to"]
            G.add_edge(f, t)
        return G

    def create_graph_vis(n, e):
        G = nx.DiGraph()
        for i in n:
            G.add_node(i["id"], label=i["label"])
        for j in e:
            f = j["from"]
            t = j["to"]
            G.add_edge(f, t, label=j["label"], color="black")
        return G

    G_list = []
    for idx, val in enumerate(nodes_list):
        n = nodes_list[idx]
        e = edges_list[idx]
        G_list.append(create_graph(n, e))
    G_list_og = []
    for idx, val in enumerate(nodes_list):
        n = nodes_list[idx]
        e = edges_list[idx]
        G_list_og.append(create_graph(n, e))
    G_list_vis = []
    for idx, val in enumerate(nodes_list):
        n = nodes_list[idx]
        e = edges_list[idx]
        G_list_vis.append(create_graph_vis(n, e))

    speaker_to_node_arr = []
    for index, val in enumerate(tqdm(G_list_vis)):
        speaker_to_node = {}
        my_map = nx.get_node_attributes(val, "label")
        inv_map = {v: k for k, v in my_map.items()}
        for j in range(max(speakers_id[index])):
            j = j + 1
            speaker_to_node[str(j)] = inv_map["speaker" + str(j)]
        speaker_to_node_arr.append(speaker_to_node)

    ## coreference resolution : connect nodes based on text coreference output
    with jsonlines.open(coref_output_file, "r") as jsonl_f:
        coref_lst = [obj for obj in jsonl_f]

    coref_arr = []
    for i in coref_lst:
        coref_arr.append(i["span_clusters"])

    max_span = 4
    coref_arr_limit_span = []
    for i in coref_arr:
        coref_arr_limit_span_tmp = []
        for j in i:
            coref_arr_limit_span_tmp_tmp = []
            for k in j:
                if k[1] - k[0] > max_span:
                    k[1] = k[0] + 3
                coref_arr_limit_span_tmp_tmp.append((k[0], k[1]))
            coref_arr_limit_span_tmp.append(coref_arr_limit_span_tmp_tmp)
        coref_arr_limit_span.append(coref_arr_limit_span_tmp)

    coref_nodes_arr = []
    for index, coref in enumerate(tqdm(coref_arr_limit_span)):
        coref_nodes = []
        coref_cluster = []
        # for each coref cluster
        for cluster in coref:
            coref_nodes = []
            # for each coref in coref cluster, turn it into node
            for i in cluster:
                coref_node = []
                # if the coref is speaker name, add speaker node
                if i[0] in snt_index_to_speaker_arr[index].keys():
                    coref_node.append(
                        speaker_to_node_arr[index][
                            str(snt_index_to_speaker_arr[index][i[0]])
                        ]
                    )

                # for other coref, if it is in range
                for j in snt_to_node_arr[index].keys():

                    j_range = j.split("-")
                    j_range = (int(j_range[0]), int(j_range[1]))
                    if in_range(j_range, i):
                        coref_node += find_node_by_jamr_format(
                            index, snt_to_node_arr[index][j]
                        )
                coref_nodes.append(coref_node)
            coref_cluster.append(coref_nodes)
        coref_nodes_arr.append(coref_cluster)

    def to_graph(l):
        G = nx.Graph()
        for part in l:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(l):
        """ 
            treat `l` as a Graph and returns it's edges 
            to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        """
        it = iter(l)
        last = next(it)

        for current in it:
            yield last, current
            last = current

    wrong_count = 0
    for index, coref_nodes in enumerate(tqdm(coref_nodes_arr)):
        test_cluster_flat = []
        for cluster in coref_nodes:
            cluster_flat = [item for sublist in cluster for item in sublist]
            if len(cluster_flat) > 0:
                test_cluster_flat.append(cluster_flat)

        G = to_graph(test_cluster_flat)
        cluster_flat = list(connected_components(G))
        cluster_flat_sorted = []
        for i in cluster_flat:
            i = sorted(list(i))
            cluster_flat_sorted.append(i)

        cluster_flat_sorted_filtered = []
        for i in cluster_flat_sorted:
            speaker_count = 0
            for j in i:
                if j in speaker_to_node_arr[index].values():
                    speaker_count += 1
            if not speaker_count > 1:
                cluster_flat_sorted_filtered.append(i)
            else:
                wrong_count += 1
        cluster_label_arr = []
        for i in cluster_flat_sorted_filtered:
            cluster_label = []
            for j in i:
                cluster_label.append(
                    nx.get_node_attributes(G_list_vis[index], "label")[j]
                )
            cluster_label_arr.append(cluster_label)

        new_concept = []
        black_list = ["this", "that", "it", "you", "i", "they"]
        for i in cluster_label_arr:
            new_concept_tmp = []
            flag = True
            for j in i:
                if j.startswith("speaker"):
                    flag = False
                    new_concept_tmp.append(j)
            if flag:
                copy_i = i.copy()
                for word in black_list:
                    copy_i = list(filter(lambda a: a != word, copy_i))

                if len(copy_i) > 0:
                    if max(Counter(copy_i).values()) == 1:
                        new_concept_word = copy_i[0]
                    else:
                        new_concept_word = max(set(copy_i), key=copy_i.count)
                else:
                    if max(Counter(i).values()) == 1:
                        new_concept_word = i[0]
                    else:
                        new_concept_word = max(set(i), key=i.count)
                new_concept_tmp.append(new_concept_word)
            new_concept.append(new_concept_tmp)
        concept_labels = nx.get_node_attributes(G_list_vis[index], "label")

        for idx, val in enumerate(cluster_flat_sorted_filtered):
            merge_to = val[0]
            if len(new_concept[idx]) != 1:
                print("error", new_concept[idx])
            concept_labels[merge_to] = new_concept[idx][0]

            for j in val[1:]:
                nx.contracted_nodes(
                    G_list[index], merge_to, j, copy=False, self_loops=False
                )
                nx.contracted_nodes(
                    G_list_vis[index], merge_to, j, copy=False, self_loops=False
                )

    ## Output final graph
    edges_dict_new_arr = []
    concepts_f_new_arr = []
    for AMR_ID in range(len(G_list_vis)):
        count = 1
        new_map = {}
        for i in G_list_vis[AMR_ID].nodes():
            new_map[i] = count
            count += 1
        new_g = nx.relabel_nodes(G_list_vis[AMR_ID], new_map, copy=True)
        edges_dict_new = {}
        for key, val in nx.get_edge_attributes(new_g, "label").items():
            s, e = key
            edges_dict_new[(s, e)] = "-:" + val
            edges_dict_new[(e, s)] = "+:" + val
        edges_dict_new_arr.append(edges_dict_new)
        concepts_f_new = []
        nodes_tmp = nx.get_node_attributes(new_g, "label")
        for i in range(len(nodes_tmp)):
            concepts_f_new.append(nodes_tmp[i + 1])
        concepts_f_new_arr.append(concepts_f_new)

    edges_dict_list = edges_dict_new_arr
    concepts_list = concepts_f_new_arr

    final_g_json_lst = []
    for ep_num in range(len(edges_dict_list)):
        relation = {}
        for k, v in edges_dict_list[ep_num]:
            if k not in relation.keys():
                relation[k] = {}
            if edges_dict_list[ep_num][(k, v)].startswith("-"):
                edge_type = edges_dict_list[ep_num][(k, v)][2:]
            else:
                edge_type = edges_dict_list[ep_num][(k, v)][2:] + "_reverse_"
            relation[k][v] = [{"edge": [edge_type]}]
        x = nx.shortest_path_length(G_list[ep_num], 1)
        depth = []
        for node, d in sorted(x.items(), key=lambda item: item[0]):
            depth.append(d)
        concepts_sense_removed = []
        for i in concepts_list[ep_num]:
            result = re.search(r"^(.+)-\d+$", i)
            if result:
                src = result.groups()[0]
            else:
                src = i
            concepts_sense_removed.append(src)
        final_g_json = {
            "concept": concepts_sense_removed,
            "relation": relation,
            "depth": depth,
        }
        final_g_json_lst.append(final_g_json)

    with open(graph_output_file, "w") as outfile:
        json.dump(final_g_json_lst, outfile)
