import json
import re
import nltk
import argparse

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_text_file")
    argparser.add_argument("jamr_dir_out")
    argparser.add_argument("coref_input_file")

    args = argparser.parse_args()
    input_text_file = args.input_text_file
    jamr_output_dir = args.jamr_dir_out
    coref_input_file = args.coref_input_file

    with open(input_text_file, encoding="utf-8") as file:
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

    char_lens_arr_arr = []

    for idx in range(len(dialog)):
        snt_a = (
            dialog[idx]
            .replace("!\n", "\n")
            .replace("?\n", "\n")
            .replace(".\n", "\n")
            .replace("\n", " .\n  ")
            .replace("``", '"')
            .replace("''", '"')
            .replace(",,", '"')
            .split("\n")
        )
        char_lens_arr = []
        for val in snt_a:
            char_lens_arr.append(len(val.replace(" ", "")))
        char_lens_arr_arr.append(char_lens_arr)

    for i in range(len(char_lens_arr_arr)):
        assert sum(char_lens_arr_arr[i]) == len(snt_arr[i].replace(" ", ""))

    char_lens_arr_acc_arr = []
    for idx in range(len(char_lens_arr_arr)):
        char_lens_arr_acc = []
        count_temp = 0
        for i in char_lens_arr_arr[idx]:
            char_lens_arr_acc.append(i + count_temp)
            count_temp += i
        char_lens_arr_acc_arr.append(char_lens_arr_acc)

    snt_splited_arr_arr = []
    for idx2 in range(len(char_lens_arr_arr)):
        char_count = 0
        snt_splited_arr = []
        snt_splited = ""
        snt_count = 0
        for idx, val in enumerate(snt_arr[idx2]):
            if val != " ":
                char_count += 1

            snt_splited += val
            if char_count == char_lens_arr_acc_arr[idx2][snt_count]:
                snt_splited_arr.append(snt_splited)
                snt_splited = ""
                snt_count += 1
        snt_splited_arr_arr.append(snt_splited_arr)

    snt_splited_arr_arr = []
    for idx2 in range(len(char_lens_arr_arr)):

        char_count = 0
        snt_splited_arr = []
        snt_splited = ""
        snt_count = 0
        for idx, val in enumerate(snt_arr[idx2]):
            if val != " ":
                char_count += 1

            snt_splited += val
            if char_count == char_lens_arr_acc_arr[idx2][snt_count]:
                snt_splited_arr.append(snt_splited)
                snt_splited = ""
                snt_count += 1
        snt_splited_arr_arr.append(snt_splited_arr)

    snts_arr = []
    speakers_arr = []
    for k in range(len(snt_splited_arr_arr)):
        csnt = snt_splited_arr_arr[k]

        speakers = []
        snts = []

        for idx, i in enumerate(csnt):
            char_count = 0
            speaker_found = ""
            speaker_known = speakers_names[k][idx]
            name_len = len(speaker_known.replace(" ", ""))
            if name_len != 0:
                for idx2, val2 in enumerate(i):
                    if val2 != " ":
                        char_count += 1
                    if val2 == '"':
                        char_count += 1
                    speaker_found += val2
                    if char_count == name_len:
                        break
            else:
                speaker_found = ""

            if speaker_found == "" and i[0] != " ":
                snt = i[len(speaker_found) + 2 :]
            else:
                snt = i[len(speaker_found) + 3 :]

            if idx != 0 and speaker_found != "":
                speaker_found = speaker_found[1:]

            if speaker_found == "":
                print(k)
                print(i)
                print(snt)
                print("*" + speaker_known + "*", "*" + speaker_found + "*")

            snts.append(snt)

            speakers.append(speaker_found)

        speakers_arr.append(speakers)
        snts_arr.append(snts)

    t = []
    for index, val in enumerate(snts_arr):
        tt = []
        for index2, val2 in enumerate(val):
            if speakers_arr[index][index2] == "":
                tt.append(speakers_arr[index][index2] + ": " + val2)
            else:
                tt.append(speakers_arr[index][index2] + " : " + val2)
        t.append(tt)

    coref_arr_in = []
    for scene_id in range(len(snt_arr)):
        text_t = snt_arr[scene_id]
        text = t[scene_id]
        text_arr = []
        name_arr = []
        for idx, val in enumerate(text):
            text_snt_split = nltk.sent_tokenize(val)
            text_arr += text_snt_split
            for _ in range(len(text_snt_split)):
                name_arr.append(speakers_names[scene_id][idx])
        tmp_count = 0
        snt_id = []
        speaker_tmp = []
        for idx, i in enumerate(text_arr):
            tmp_count += len(i.split())
            for _ in range(len(i.split())):
                snt_id.append(idx)
                speaker_tmp.append(name_arr[idx])
        doc_id = "doc_" + str(scene_id)
        test_dict = {"document_id": doc_id}
        test_dict["cased_words"] = text_t.split()
        test_dict["speaker"] = speaker_tmp
        test_dict["sent_id"] = snt_id
        coref_arr_in.append(test_dict)
        assert len(test_dict["cased_words"]) == len(test_dict["speaker"])
        assert len(test_dict["cased_words"]) == len(test_dict["sent_id"])
        assert len(test_dict["speaker"]) == len(test_dict["sent_id"])

    with open(coref_input_file, "w") as f:
        for d in coref_arr_in:
            json.dump(d, f)
            f.write("\n")
