# from transformers import Pipeline
import string
import re
from tqdm import tqdm
import random


class Preprocess:
    def __init__(self, main_path, datasets, epochs=1):
        self.main_path = main_path

        self.DICTIONARY_DIR = main_path + "/dictionary/dictionary.txt"
        self.KEYBOARD_ERRORS_DIR = (
            main_path + "/dictionary/keyboard_realword_errors.txt"
        )
        self.SUBSTITUTION_ERRORS_DIR = (
            main_path + "/dictionary/substitution_realword_errors.txt"
        )
        self.HOMOPHONE_ERRORS_DIR = (
            main_path + "/dictionary/homophone_realword_errors.txt"
        )

        self.RAW_DATASET_DIR = main_path + "/raw datasets/total_rawdata.txt"

        self.PRUNE_DATA = main_path + "/pruned datasets/total_prunedata.txt"

        self.TRAIN_DATASET = main_path + "/train test datasets/train/"
        self.TRAIN_FINAL_DATASET = (
            main_path + "/train test datasets/train/final_dataset.txt"
        )
        self.CORRECT_SENTNCES = (
            main_path + "/train test datasets/train/correct_sentences.txt"
        )
        self.METADATA = main_path + "/train test datasets/train/metadata.txt"
        self.CORRUPTED_SENTENCES = (
            main_path + "/train test datasets/train/corrupted_sentences.txt"
        )
        self.MASKED_SENTENCES = (
            main_path + "/train test datasets/train/masked_sentences.txt"
        )

        self.TEST_DATASET = main_path + "/train test datasets/test/final_dataset.txt"
        self.TEST_DATASET_100 = (
            main_path + "/train test datasets/test/final_dataset_100.txt"
        )

        self.datasets = datasets

        self.letters = [
            "ا",
            "ب",
            "پ",
            "ت",
            "ث",
            "ج",
            "چ",
            "ح",
            "خ",
            "د",
            "ذ",
            "ر",
            "ز",
            "ژ",
            "س",
            "ش",
            "ص",
            "ض",
            "ط",
            "ظ",
            "ع",
            "غ",
            "ف",
            "ق",
            "ک",
            "گ",
            "ل",
            "م",
            "ن",
            "و",
            "ه",
            "ی",
            "آ",
        ]
        self.NUMOFLETTERS = len(self.letters)
        self.polymorph = {
            "ز": ["ذ", "ض", "ظ"],
            "ذ": ["ز", "ض", "ظ"],
            "ض": ["ز", "ذ", "ظ"],
            "ظ": ["ز", "د", "ض"],
            "س": ["ث", "ص"],
            "ث": ["س", "ص"],
            "ص": ["س", "ث"],
            "ط": ["ت"],
            "ت": ["ط"],
            "ق": ["غ"],
            "غ": ["ق"],
            "ه": ["ح"],
            "ح": ["ه"],
        }
        self.keyboard_neighbor = {
            "آ": "غتلدذع",
            "ا": "غتلدذع",
            "ب": "قفلیزر",
            "إ": "قفلیزر",
            "پ": "ونتد",
            "ت": "اعهنپد",
            "ة": "اعهنپد",
            "ث": "سیقص",
            "ج": "چگکح",
            "چ": "جگ",
            "ح": "مکجخ",
            "خ": "هنمح",
            "د": "اتذپ",
            "ذ": "الدر",
            "ر": "لبذز",
            "ز": "یبرط",
            "ژ": "یبطر",
            "س": "صثشیظط",
            "ش": "ضصسطظ",
            "ؤ": "ضصسطظ",
            "ص": "ثضیشس",
            "ض": "صسش",
            "ط": "ظسیز",
            "ظ": "طشس",
            "ع": "غاته",
            "غ": "فلاع",
            "ف": "قغبل",
            "ق": "ثفبی",
            "ک": "گمحج",
            "گ": "جک",
            "ل": "ابغفذر",
            "أ": "ابغفذر",
            "م": "کنحخو",
            "ن": "متخهوپ",
            "و": "پنم",
            "ه": "عختن",
            "ی": "قثبسطز",
        }

        print("creating dictionary ...")
        self.dictionary = self.__create_dictionary()

        print("generating realword error files ...")
        # self.__generate_realword_errors_per_token_files()

        print("load homophone, keyboard, substitution realword errors ...")
        self.homophone_realword_errors = self.__load_homophone_errors()
        self.keyboard_realword_errors = self.__load_keyboard_errors()
        self.substitution_realword_errors = self.__load_substitution_errors()

        print("merging datasets ...")
        # self.__merge_files(datasets, self.RAW_DATASET_DIR, True)

        print("prune dataset ...")
        # self.__prune_dataset()

        print("generate test and train dataset ...")
        self.__generate_final_dataset_test()
        # self.__generate_final_dataset_test_100()

        final_datasets = []
        for i in range(epochs):
            # self.__generate_final_dataset_train(id_=i)
            # final_datasets.append(self.TRAIN_DATASET + f"final_dataset_{i}.txt")
            pass

        # self.__merge_files(final_datasets, self.TRAIN_FINAL_DATASET)

        print("generate metadata ...")
        # self.__generate_meta_datasets()

    def __create_dictionary(self):
        dictionary = {}

        with open(self.DICTIONARY_DIR, "r", encoding="utf-8") as f:
            for idx, word in enumerate(f):
                dictionary[word.strip()] = idx

        return dictionary

    def __split(self, token):
        return [char for char in token]

    def __manual_replace(self, token, new_char, index):
        return token[:index] + new_char + token[index + 1 :]

    def __generate_keyboard_errors_per_token(self, token):
        keyboard_errors = []

        for index, char in enumerate(token):
            neighbour_string = self.keyboard_neighbor.get(char, "")
            neighbour_list = self.__split(neighbour_string)

            for ch in neighbour_list:
                new_token = self.__manual_replace(token, ch, index)

                if new_token in self.dictionary:
                    keyboard_errors.append(new_token)

        return keyboard_errors

    def __generate_substitution_errors_per_token(self, token):
        substitution_errors = []

        wordLength = len(token)
        for indexOfSub in range(0, wordLength - 1):
            if token[indexOfSub + 1] == token[indexOfSub]:
                continue

            new_token = ""
            idx = 0
            while idx < wordLength:
                if idx == indexOfSub:
                    new_token += token[idx + 1]
                elif idx == indexOfSub + 1:
                    new_token += token[idx - 1]
                else:
                    new_token += token[idx]
                idx += 1

            if new_token in self.dictionary:
                substitution_errors.append(new_token)

        return substitution_errors

    def __generate_realword_errors_per_token_files(self):
        with open(self.KEYBOARD_ERRORS_DIR, "w", encoding="utf-8") as f1, open(
            self.SUBSTITUTION_ERRORS_DIR, "w", encoding="utf-8"
        ) as f2:
            for i, word in enumerate(self.dictionary):
                keyboard_realword_errors = self.__generate_keyboard_errors_per_token(
                    word.strip()
                )
                substitution_realword_errors = (
                    self.__generate_substitution_errors_per_token(word.strip())
                )

                line = word.strip() + " "
                if len(keyboard_realword_errors) > 0:
                    for error in keyboard_realword_errors:
                        line += error + ","
                    f1.write(line.strip(",") + "\n")

                line = word.strip() + " "
                if len(substitution_realword_errors) > 0:
                    for error in substitution_realword_errors:
                        line += error + ","
                    f2.write(line.strip(",") + "\n")

    def __is_line_in_dataset(self, line):
        words = line.strip().split(" ")

        for word in words:
            if word not in self.dictionary:
                return False
        return True

    def __load_homophone_errors(self):
        homophone_realword_errors = {}
        with open(self.HOMOPHONE_ERRORS_DIR, "r", encoding="utf-8") as f:
            for line in f:
                line = f.readline().strip()
                word, errors = line.split(" ")
                listoferrors = errors.split(",")
                listoferrors.append(word)

                for err in listoferrors:
                    temp = [x for x in listoferrors if x != err]

                    if err in homophone_realword_errors:
                        homophone_realword_errors[err] += temp
                    else:
                        homophone_realword_errors[err] = temp

        return homophone_realword_errors

    def __load_keyboard_errors(self):
        keyboard_realword_errors = {}
        with open(self.KEYBOARD_ERRORS_DIR, "r", encoding="utf-8") as f:
            for line in f:
                word, errors = line.strip().split(" ")
                errors = errors.split(",")

                if len(errors) > 0:
                    keyboard_realword_errors[word] = errors

        return keyboard_realword_errors

    def __load_substitution_errors(self):
        substitution_realword_errors = {}
        with open(self.SUBSTITUTION_ERRORS_DIR, "r", encoding="utf-8") as f:
            for line in f:
                word, errors = line.strip().split(" ")
                errors = errors.split(",")

                if len(errors) > 0:
                    substitution_realword_errors[word] = errors

        return substitution_realword_errors

    def __merge_files(self, sources, target, add_nextline=False):
        with open(target, "w", encoding="utf-8") as target:
            for src in sources:
                with open(src, "r", encoding="utf-8") as source:
                    for line in source:
                        if add_nextline:
                            target.write(line + "\n")
                        else:
                            target.write(line)

    def __prune_dataset(self):
        with open(self.RAW_DATASET_DIR, "r", encoding="utf-8") as f1, open(
            self.PRUNE_DATA, "w", encoding="utf-8"
        ) as f2:
            for line in tqdm(f1):
                # if re.match(".*[۱۲۳۴۵۶۷۸۹]", line) or re.match(".*[a-zA-Z0-9]", line):
                #     continue

                line = re.sub(".*[۱۲۳۴۵۶۷۸۹]", "", line)

                line = re.sub(".*[a-zA-Z0-9]", "", line)

                line = re.sub(r"\(.*\)", "", line)

                line = re.sub(r"-.*-", "", line)

                if line.count("(") != line.count(")"):
                    continue

                puctuations = string.punctuation + "»«؟،؛"
                for p in puctuations:
                    if p != "(" and p != ")":
                        line = line.replace(p, "")

                line = line.strip()
                length = len(line.split())

                if length < 5 or length > 256:
                    continue

                if self.__is_line_in_dataset(line):
                    f2.write(line + "\n")

    def __generate_keyboard_nonrealword_errors_per_line(self, li):
        newLine = ""
        token = ""
        newToken = ""
        tokens = li.split(" ")
        indices = list(range(0, len(tokens)))
        random.shuffle(indices)
        for ind in indices:
            t = tokens[ind]
            if len(t) > 2:
                tokenLength = len(t)
                newToken = "خالی"
                indexOfMisspelled = random.randrange(0, tokenLength)
                indexOfAlternative = random.randrange(0, self.NUMOFLETTERS)
                newToken = self.__manual_replace(
                    t, self.letters[indexOfAlternative], indexOfMisspelled
                )
                while (
                    newToken in self.dictionary
                    or t[indexOfMisspelled] == self.letters[indexOfAlternative]
                ):
                    indexOfMisspelled = random.randrange(0, tokenLength)
                    indexOfAlternative = random.randrange(0, self.NUMOFLETTERS)
                    newToken = self.__manual_replace(
                        t, self.letters[indexOfAlternative], indexOfMisspelled
                    )
                token = t
                tokens[ind] = newToken
                newLine = " ".join(tokens)
                break

        return newLine, token, newToken

    def __change_polymorph(self, word):
        possibletokens = []

        for key, value in self.polymorph.items():
            if key in word:
                ind = word.find(key)
                wordlength = len(word)
                random.shuffle(value)
                newtoken = ""

                for z in range(0, ind):
                    newtoken += word[z]
                newtoken += value[0]

                for z in range(ind + 1, wordlength):
                    newtoken += word[z]
                possibletokens.append(newtoken)

        for p in possibletokens:
            if p not in self.dictionary:
                random.shuffle(possibletokens)
                return possibletokens[0]

        return False

    def __has_polymorph(self, word):
        for key, value in self.polymorph.items():
            if key in word:
                return True
        return False

    def __generate_polymorph_nonrealword_per_line(self, li):
        newLine = ""
        newToken = ""
        token = ""
        tokens = li.split(" ")
        indices = list(range(0, len(tokens)))
        random.shuffle(indices)
        for ind in indices:
            t = tokens[ind]
            if len(t) > 2 and self.__has_polymorph(t):
                newToken = self.__change_polymorph(t)
                if not newToken:
                    continue
                token = t
                tokens[ind] = newToken
                newLine = " ".join(tokens)
                break
        return newLine, token, newToken

    def __check_substitution_nonrealword_errors(self, token):
        possibletokens = []
        tokenLength = len(token)
        for indexOfSub in range(0, tokenLength - 1):
            newtoken = ""
            ind = 0

            while ind < tokenLength:
                if ind == indexOfSub:
                    newtoken += token[ind + 1]
                elif ind == indexOfSub + 1:
                    newtoken += token[ind - 1]
                else:
                    newtoken += token[ind]
                ind += 1
            possibletokens.append(newtoken)

        for p in possibletokens:
            if p not in self.dictionary:
                random.shuffle(possibletokens)
                return possibletokens[0]
        return False

    def __generate_substitution_nonrealword_errors_per_line(self, li):
        newLine = ""
        newToken = ""
        token = ""
        tokens = li.split(" ")
        indices = list(range(0, len(tokens)))
        random.shuffle(indices)
        for ind in indices:
            t = tokens[ind]
            if len(t) > 2:
                newToken = self.__check_substitution_nonrealword_errors(t)

                if newToken is False:
                    newToken = ""
                    continue
                token = t
                tokens[ind] = newToken
                newLine = " ".join(tokens)
                break
        return newLine, token, newToken

    def __generate_homophone_realword_errors_per_line(self, li, token):
        possibleErrors = self.homophone_realword_errors[token]
        random.shuffle(possibleErrors)

        for err in possibleErrors:
            if token != err:
                tokens = li.split(" ")
                ind = tokens.index(token)
                tokens[ind] = err
                replaced = " ".join(tokens)
                return replaced, token, err

    def __generate_keyboard_realword_errors_per_line(self, li, token):
        possibleErrors = self.keyboard_realword_errors[token]
        random.shuffle(possibleErrors)

        for err in possibleErrors:
            if token != err:
                tokens = li.split(" ")
                ind = tokens.index(token)
                tokens[ind] = err
                newLine = " ".join(tokens)

                return newLine, token, err

    def __generate_substitution_realword_errors_per_line(self, li, token):
        possibleErrors = self.substitution_realword_errors[token]
        random.shuffle(possibleErrors)

        for err in possibleErrors:
            if token != err:
                tokens = li.split(" ")
                ind = tokens.index(token)
                tokens[ind] = err
                newLine = " ".join(tokens)

                return newLine, token, err

    def __can_be_keyboard_realword_errors(self, li):
        listoftokens = li.split(" ")
        random.shuffle(listoftokens)
        for token in listoftokens:
            if len(token) > 2 and token in self.keyboard_realword_errors:
                return token
        return False

    def __can_be_substitution_realword_errors(self, li):
        listoftokens = li.split(" ")
        random.shuffle(listoftokens)
        for token in listoftokens:
            if len(token) > 2 and token in self.substitution_realword_errors:
                return token
        return False

    def __can_be_homophone_realword_errors(self, li):
        listoftokens = li.strip().split(" ")
        random.shuffle(listoftokens)
        for token in listoftokens:
            if len(token) > 2 and token in self.homophone_realword_errors:
                return token
        return False

    def __generate_final_dataset_test(self):
        with open(self.PRUNE_DATA, "r", encoding="utf-8") as f1, open(
            self.TEST_DATASET, "w", encoding="utf-8"
        ) as f2:
            total = 0

            homophone_realword_error = 0
            homophone_realword_correct = 0

            substitution_realword_error = 0
            substitution_realword_correct = 0

            keyboard_realword_error = 0
            keyboard_realword_correct = 0

            keyboard_nonrealword_error = 0

            polymorph_nonrealword_error = 0

            substitution_nonrealword_error = 0

            for line in f1:
                line = line.strip()

                newLine = ""
                oldToken = ""
                newToken = ""

                type_ = ""
                total += 1

                can1 = self.__can_be_homophone_realword_errors(line)
                if can1 is not False and random.random() < 0.8:
                    if random.random() < 0.5:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_homophone_realword_errors_per_line(
                            line, can1
                        )
                        type_ = "homophone_realword_error"
                        homophone_realword_error += 1
                    else:
                        (
                            _,
                            oldToken,
                            newToken,
                        ) = self.__generate_homophone_realword_errors_per_line(
                            line, can1
                        )
                        newLine = line
                        type_ = "homophone_realword_correct"
                        homophone_realword_correct += 1

                elif random.random() <= 0.5:  # real word error
                    can5 = self.__can_be_substitution_realword_errors(line)
                    can6 = self.__can_be_keyboard_realword_errors(line)

                    i = random.randrange(2, 4)

                    if i == 2 and can5 is not False:
                        if random.random() < 0.5:
                            (
                                newLine,
                                oldToken,
                                newToken,
                            ) = self.__generate_substitution_realword_errors_per_line(
                                line, can5
                            )
                            type_ = "substitution_realword_error"
                            substitution_realword_error += 1
                        else:
                            (
                                _,
                                oldToken,
                                newToken,
                            ) = self.__generate_substitution_realword_errors_per_line(
                                line, can5
                            )
                            newLine = line
                            type_ = "substitution_realword_correct"
                            substitution_realword_correct += 1

                    elif i == 3 and can6 is not False:
                        if random.random() < 0.5:
                            (
                                newLine,
                                oldToken,
                                newToken,
                            ) = self.__generate_keyboard_realword_errors_per_line(
                                line, can6
                            )
                            if newToken is False:
                                continue
                            type_ = "keyboard_realword_error"
                            keyboard_realword_error += 1
                        else:
                            (
                                _,
                                oldToken,
                                newToken,
                            ) = self.__generate_keyboard_realword_errors_per_line(
                                line, can6
                            )
                            if newToken is False:
                                continue
                            newLine = line
                            type_ = "keyboard_realword_correct"
                            keyboard_realword_correct += 1

                else:  # non real word error
                    i = random.randrange(1, 4)
                    if i == 1:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_keyboard_nonrealword_errors_per_line(line)
                        type_ = "keyboard_nonrealword_error"
                        keyboard_nonrealword_error += 1
                    elif i == 2:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_polymorph_nonrealword_per_line(line)
                        type_ = "polymorph_nonrealword_error"
                        polymorph_nonrealword_error += 1
                    elif i == 3:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_substitution_nonrealword_errors_per_line(
                            line
                        )
                        type_ = "substitution_nonrealword_error"
                        substitution_nonrealword_error += 1
                if newLine:
                    f2.write(
                        newLine.strip()
                        + "^"
                        + str(type_)
                        + "^"
                        + str(oldToken)
                        + "^"
                        + str(newToken)
                        + "\n"
                    )

            print("test")
            print("total", total)
            print("homophone_realword_error", homophone_realword_error)
            print("homophone_realword_correct", homophone_realword_correct)
            print("keyboard_realword_error", keyboard_realword_error)
            print("keyboard_realword_correct", keyboard_realword_correct)
            print("substitution_realword_error", substitution_realword_error)
            print("substitution_realword_correct", substitution_realword_correct)
            print("polymorph_nonrealword_error", polymorph_nonrealword_error)
            print("substitution_nonrealword_error", substitution_nonrealword_error)
            print("keyboard_nonrealword_error", keyboard_nonrealword_error)

    def __generate_final_dataset_train(self, id_=1):
        with open(self.PRUNE_DATA, "r", encoding="utf-8") as f1, open(
            self.TRAIN_DATASET + f"final_dataset_{id_}.txt", "w", encoding="utf-8"
        ) as f2:
            c0 = 0
            c1 = 0
            c2 = 0
            c3 = 0
            c4 = 0
            c5 = 0
            c6 = 0
            c7 = 0

            for line in f1:
                line = line.strip()

                newLine = ""
                oldToken = ""
                newToken = ""

                typeOfError = ""
                c0 += 1

                can1 = self.__can_be_homophone_realword_errors(line)
                if can1 is not False and random.random() < 0.8:
                    (
                        newLine,
                        oldToken,
                        newToken,
                    ) = self.__generate_homophone_realword_errors_per_line(line, can1)
                    typeOfError = "homophone_realword_error"
                    c1 += 1

                elif random.random() <= 0.6:  # No error
                    newLine = line
                    oldToken = "-"
                    newToken = "-"
                    typeOfError = "no_error"
                    c7 += 1

                elif random.random() <= 0.5:  # real word error
                    can5 = self.__can_be_substitution_realword_errors(line)
                    can6 = self.__can_be_keyboard_realword_errors(line)

                    i = random.randrange(2, 4)

                    if i == 2 and can5 is not False:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_substitution_realword_errors_per_line(
                            line, can5
                        )
                        typeOfError = "substitution_realword_error"
                        c5 += 1
                    elif i == 3 and can6 is not False:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_keyboard_realword_errors_per_line(
                            line, can6
                        )
                        if newToken is False:
                            continue
                        typeOfError = "keyboard_realword_error"
                        c6 += 1

                else:  # non real word error
                    i = random.randrange(1, 4)
                    if i == 1:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_keyboard_nonrealword_errors_per_line(line)
                        typeOfError = "keyboard_nonrealword_error"
                        c2 += 1
                    elif i == 2:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_polymorph_nonrealword_per_line(line)
                        typeOfError = "polymorph_nonrealword_error"
                        c3 += 1
                    elif i == 3:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_substitution_nonrealword_errors_per_line(
                            line
                        )
                        typeOfError = "substitution_nonrealword_error"
                        c4 += 1

                f2.write(
                    newLine.strip()
                    + "^"
                    + str(typeOfError)
                    + "^"
                    + str(oldToken)
                    + "^"
                    + str(newToken)
                    + "\n"
                )

            print("train@epoch", id_)
            print("total", c0)
            print("no_error", c7)
            print("homophone_realword_error", c1)
            print("keyboard_nonrealword_error", c2)
            print("polymorph_nonrealword_error", c3)
            print("substitution_nonrealword_error", c4)
            print("substitution_realword_error", c5)
            print("keyboard_realword_error", c6)

    def __generate_final_dataset_test_100(self):
        with open(self.PRUNE_DATA, "r", encoding="utf-8") as f1, open(
            self.TEST_DATASET_100, "w", encoding="utf-8"
        ) as f2:
            c0 = 0
            c1 = 0
            c2 = 0
            c3 = 0
            c4 = 0
            c5 = 0
            c6 = 0
            c7 = 0

            for line in f1:
                line = line.strip()

                newLine = ""
                oldToken = ""
                newToken = ""

                typeOfError = ""
                c0 += 1

                can1 = self.__can_be_homophone_realword_errors(line)
                if can1 is not False and random.random() < 1 and c1 < 17:
                    (
                        newLine,
                        oldToken,
                        newToken,
                    ) = self.__generate_homophone_realword_errors_per_line(line, can1)
                    typeOfError = "homophone_realword_error"
                    c1 += 1

                elif random.random() < 0:  # No error
                    newLine = line
                    oldToken = "-"
                    newToken = "-"
                    typeOfError = "no_error"
                    c7 += 1

                elif random.random() <= 0.42:  # real word error
                    can5 = self.__can_be_substitution_realword_errors(line)
                    can6 = self.__can_be_keyboard_realword_errors(line)

                    i = random.randrange(2, 4)

                    if i == 2 and can5 is not False and c5 < 17:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_substitution_realword_errors_per_line(
                            line, can5
                        )
                        typeOfError = "substitution_realword_error"
                        c5 += 1
                    elif i == 3 and can6 is not False and c6 < 17:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_keyboard_realword_errors_per_line(
                            line, can6
                        )
                        if newToken is False:
                            continue
                        typeOfError = "keyboard_realword_error"
                        c6 += 1

                else:  # non real word error
                    i = random.randrange(1, 4)
                    if i == 1 and c2 < 17:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_keyboard_nonrealword_errors_per_line(line)
                        typeOfError = "keyboard_nonrealword_error"
                        c2 += 1
                    elif i == 2 and c3 < 16:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_polymorph_nonrealword_per_line(line)
                        typeOfError = "polymorph_nonrealword_error"
                        c3 += 1
                    elif i == 3 and c4 < 16:
                        (
                            newLine,
                            oldToken,
                            newToken,
                        ) = self.__generate_substitution_nonrealword_errors_per_line(
                            line
                        )
                        typeOfError = "substitution_nonrealword_error"
                        c4 += 1

                if typeOfError != "":
                    f2.write(
                        newLine.strip()
                        + "^"
                        + str(typeOfError)
                        + "^"
                        + str(oldToken)
                        + "^"
                        + str(newToken)
                        + "\n"
                    )

                if (
                    c1 == 17
                    and c6 == 17
                    and c5 == 17
                    and c2 == 17
                    and c3 == 16
                    and c4 == 16
                ):
                    break

            print("test_100")
            print("total", c0)
            print("no_error", c7)
            print("homophone_realword_error", c1)
            print("keyboard_nonrealword_error", c2)
            print("polymorph_nonrealword_error", c3)
            print("substitution_nonrealword_error", c4)
            print("substitution_realword_error", c5)
            print("keyboard_realword_error", c6)

    def __generate_meta_datasets(self):
        with open(self.TRAIN_FINAL_DATASET, "r", encoding="utf-8") as f, open(
            self.CORRECT_SENTNCES, "w+", encoding="utf-8"
        ) as f1, open(self.METADATA, "w+", encoding="utf-8") as f2, open(
            self.CORRUPTED_SENTENCES, "w+", encoding="utf-8"
        ) as f3, open(
            self.MASKED_SENTENCES, "w+", encoding="utf-8"
        ) as f4:
            for line in f:
                tokens = line.strip().split("^")
                # print(tokens[3])
                f1.write(tokens[0].replace(tokens[3], tokens[2], 1) + "\n")
                f2.write(tokens[1] + "," + tokens[2] + "," + tokens[3] + "\n")
                f3.write(tokens[0] + "\n")
                f4.write(tokens[0].replace(tokens[3], "[MASK]", 1) + "\n")


if __name__ == "__main__":
    datasets = []
    main_path = input("main path: ")
    print("add dataset addresses, otherwise type !q")
    while True:
        dataset = input()
        if "!q" in dataset:
            break
        datasets.append(dataset)

    pre = Preprocess(main_path, datasets)
