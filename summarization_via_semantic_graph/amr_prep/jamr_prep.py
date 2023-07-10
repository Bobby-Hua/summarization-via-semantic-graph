import os
import re
import json
from tqdm import tqdm
import penman
import argparse
from penman.codec import PENMANCodec


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_amr_gs_file")
    argparser.add_argument("input_text_file")
    argparser.add_argument("jamr_dir")

    args = argparser.parse_args()
    input_amr_gs_file = args.input_amr_gs_file
    input_text_file = args.input_text_file
    jamr_input_dir = args.jamr_dir + "_in"
    jamr_output_dir = args.jamr_dir + "_out"

    c = PENMANCodec()
    with open(input_amr_gs_file, encoding="utf-8") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    with open(input_text_file, encoding="utf-8") as file:
        dialogs = json.load(file)

    start_index = 0
    end_index = len(dialogs)

    empty_example = []
    for idx, val in enumerate(dialogs):
        if len(val["dialogue"]) == 0:
            empty_example.append(idx)
    for i in sorted(empty_example, reverse=True):
        dialogs.pop(i)

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

    dialogs = dialogs[start_index:end_index]
    num_dialog = num_dialog[start_index:end_index]
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

    entry = []
    i = 0
    for line in lines:
        if line.strip().startswith("# ::id"):
            entry.append(i)
        i = i + 1

    start = []
    end = []
    for idx, val in enumerate(entry):
        start.append(val + 7)
        if idx != 0:
            end.append(val - 1)
    end.append(i - 1)

    amr = []
    for idx, val in enumerate(start):
        amr.append(lines[val : end[idx]])

    def add_dummy(amr):
        amr_txt = ["(c00 / multi-utter"]
        for i in amr:
            for j in i:
                amr_txt.append(j)
        amr_txt[-1] += ")"
        return amr_txt

    def preprocess_speaker(speakers_id, count_base):
        amr_speaker_node = []
        amr_speaker_node_utter = []
        count = count_base
        count_utter = 1
        for i in set(speakers_id):
            aux = []
            aux_utter = []
            for j in speakers_id:
                if i == j:
                    aux.append(amr[count])
                    aux_utter.append(count_utter)
                count += 1
                count_utter += 1
            count = count_base
            count_utter = 1
            amr_speaker_node.append(aux)
            amr_speaker_node_utter.append(aux_utter)
        return amr_speaker_node, amr_speaker_node_utter

    def add_speaker_node(amr, speaker_count):
        amr_txt = [
            ":participant"
            + str(speaker_count)
            + " (s"
            + str(speaker_count)
            + " / speaker"
            + str(speaker_count)
        ]
        for i in amr:
            for j in i:
                amr_txt.append(j)
        amr_txt[-1] += ")"
        return amr_txt

    def add_utter_v2(all_utter_speaker, utter_num):
        u = 1
        all_utter_speaker_new = []
        for idx, val in enumerate(all_utter_speaker):

            val = add_utter_node_v2(val, utter_num[idx])
            all_utter_speaker_new.append(val)
            u = u + 1
        return all_utter_speaker_new

    def add_utter_node_v2(amr, utter_num, need_num_node=False):
        if need_num_node:
            amr_txt = [
                ":utter"
                + str(utter_num)
                + " (u"
                + str(utter_num)
                + " / utter"
                + str(utter_num)
            ]
        else:
            amr_txt = [" (u" + str(utter_num) + " / utter"]
        count = 0
        multi_snt_flag = False
        for i in amr:
            if count == 0 and i.endswith("multi-sentence"):
                count += 1
                multi_snt_flag = True
            elif count == 0:
                i = ":snt1 " + i
                amr_txt.append(i)
            else:
                amr_txt.append(i)
            count += 1
        if not multi_snt_flag:
            amr_txt[-1] += ")"
        amr_txt_line = " ".join(amr_txt).replace("#", "")
        t = c.parse(amr_txt_line)
        t.reset_variables(fmt="u" + str(utter_num) + "c{i}")
        amr_txt = penman.format(t, indent=3).split("\n")
        amr_txt[0] = ":utter" + str(utter_num) + amr_txt[0]
        return amr_txt

    count = 0
    amr_txt = []
    amr_txt_utter = []
    for idx, val in enumerate(num_dialog):
        txt, utter = preprocess_speaker(speakers_id[idx], count)
        amr_txt.append(txt)
        amr_txt_utter.append(utter)
        count += val
    speaker_node_amr_final = []

    for index, i in enumerate(tqdm(amr_txt)):
        speaker_node_amr = []
        name = 1

        for index2, all_utter_speaker in enumerate(i):
            all_utter_speaker = add_utter_v2(
                all_utter_speaker, amr_txt_utter[index][index2]
            )
            speaker_node_amr.append(add_speaker_node(all_utter_speaker, name))
            name += 1
        speaker_node_amr_concated = []
        for k in speaker_node_amr:
            speaker_node_amr_concated += k
        speaker_node_amr_final.append(add_dummy([speaker_node_amr_concated]))

    def format_amr(amr_txt, index, count):
        string = ""
        for line in amr_txt:

            if line.count('"') > 0:
                p = re.compile('".*"')
                s = line
                r = p.search(s)
                new_s = (
                    s[0 : r.span()[0]]
                    + s[r.span()[0] : r.span()[1]]
                    .replace("/", "-")
                    .replace(":", "-")
                    .replace("(", "-")
                    .replace(")", "-")
                    + s[r.span()[1] :]
                )
                if new_s != s:
                    print(new_s)

                line = new_s

            if line.count(":") > 1:
                if line.count(":") != 2:
                    print("possible error: check", index)
                line = line[::-1].replace(":"[::-1], ""[::-1], 1)[::-1]
                count.add(index)
            if line.strip().startswith(":wiki"):
                x = re.search(r"wiki (\".*?\")?(-)?(\)*)", line)
                string += x.group(3)
            else:
                string += " "
                string += line.replace("\n", "").strip()

        string = string.strip()
        string = string.replace("<", "^<").replace(">", "^>")
        string = string.replace('"', "")
        string = string.replace("#", "")

        return string

    tmp = []
    count = set()
    for index, val in enumerate(speaker_node_amr_final):
        tmp.append(format_amr(val, index, count))

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

    out_strings = []
    for index, val in enumerate(tmp):
        text = (
            dialog[index]
            .replace("?\n", "?  ")
            .replace("!\n", "!  ")
            .replace(".\n", ".  ")
            .replace("\n", ".  ")
        )
        out_string = "# ::snt " + text + "\n"
        out_string += val
        out_strings.append(out_string)

    os.mkdir(jamr_input_dir)
    os.mkdir(jamr_output_dir)
    for index, val in enumerate(tqdm(out_strings)):
        with open(
            jamr_input_dir + "/jamr_input_" + str(index) + ".txt", "w", encoding="utf-8"
        ) as f:
            f.write(val)
