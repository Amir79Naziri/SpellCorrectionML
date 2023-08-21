from polyleven import levenshtein
from happytransformer import HappyWordPrediction
import pandas as pd
from tqdm import tqdm
import os
import re

"""

    Non Real-word:
        same as version 7
               
    Real-word:
        make bert thereshold dynamic
    
    Procedure:
        same as version3
    
"""


class TestModel:
    def __init__(
        self,
        main_path="/mnt/disk1/users/naziri",
        model_path="HooshvareLab/bert-base-parsbert-uncased",
        output_file_name="result",
        mask_token="[MASK]",
        save_model=None,
        bert_realword_thereshold=1e-3
    ):
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
        self.FINAL_DATASET_DIR = (
            main_path + "/train test datasets/test/final_dataset_v2.txt"
        )
        self.OUTPUT_FILE_DIR = (
            main_path + "/evaluation results/" + output_file_name + ".csv"
        )
        self.MASK = mask_token
        self.bert_realword_thereshold = bert_realword_thereshold
        
        print("creating dictionary ...")
        self.dictionary = self.__create_dictionary()

        print("load homophone, keyboard, substitution realword errors ...")
        total_realword_errors = self.__load_homophone_errors({})
        total_realword_errors = self.__load_keyboard_errors(total_realword_errors)
        self.realword_errors = self.__load_substitution_errors(total_realword_errors)

        print("load model ...")
        self.model = HappyWordPrediction("BERT", load_path=model_path)
        
        if save_model:
            self.model.save(save_model)

        print("evaluation ...")
        self.__evaluate()

    def __create_dictionary(self):
        dictionary = {}

        with open(self.DICTIONARY_DIR, "r", encoding="utf-8") as f:
            for idx, word in enumerate(f):
                dictionary[word.strip()] = idx

        return dictionary

    def __load_homophone_errors(self, total_realword_errors):
        with open(self.HOMOPHONE_ERRORS_DIR, "r", encoding="utf-8") as f:
            for line in f:
                line = f.readline().strip()
                word, errors = line.split(" ")
                listoferrors = errors.split(",")
                listoferrors.append(word)

                for err in listoferrors:
                    temp = [x for x in listoferrors if x != err]

                    if err in total_realword_errors:
                        total_realword_errors[err] += temp
                        total_realword_errors[err] = list(
                            set(total_realword_errors[err])
                        )
                    else:
                        total_realword_errors[err] = temp
                        total_realword_errors[err] = list(
                            set(total_realword_errors[err])
                        )

        return total_realword_errors

    def __load_keyboard_errors(self, total_realword_errors):
        with open(self.KEYBOARD_ERRORS_DIR, "r", encoding="utf-8") as f:
            for line in f:
                word, errors = line.strip().split(" ")
                errors = errors.split(",")

                if len(errors) > 0:
                    if word in total_realword_errors:
                        total_realword_errors[word] += errors
                        total_realword_errors[word] = list(
                            set(total_realword_errors[word])
                        )
                    else:
                        total_realword_errors[word] = errors
                        total_realword_errors[word] = list(
                            set(total_realword_errors[word])
                        )

        return total_realword_errors

    def __load_substitution_errors(self, total_realword_errors):
        with open(self.SUBSTITUTION_ERRORS_DIR, "r", encoding="utf-8") as f:
            for line in f:
                word, errors = line.strip().split(" ")
                errors = errors.split(",")

                if len(errors) > 0:
                    if word in total_realword_errors:
                        total_realword_errors[word] += errors
                        total_realword_errors[word] = list(
                            set(total_realword_errors[word])
                        )
                    else:
                        total_realword_errors[word] = errors
                        total_realword_errors[word] = list(
                            set(total_realword_errors[word])
                        )

        return total_realword_errors

    def __get_most_similar_token_levenshtein(self, target_word, k=300):
        def find(myList):
            for element in myList:
                if element.get("score") > 1:
                    return myList.index(element)
            return len(myList)

        list_of_similars = []

        for word in self.dictionary:
            score = levenshtein(word, target_word)
            # freq = word_frequency(word, 'fa')

            list_of_similars.append({"word": word, "score": score})

        list_of_similars.sort(key=lambda x: x["score"])
        indUntil1 = find(list_of_similars)

        list_of_similars = list_of_similars[0:indUntil1]

        for i in range(len(target_word) - 1):
            j = i + 1
            temp_word = (
                target_word[:i] + target_word[j] + target_word[i] + target_word[j + 1 :]
            )
            if temp_word in self.dictionary:
                list_of_similars.append({"word": temp_word, "score": 2})

        return list_of_similars

    def __get_most_similar_token_mix(
        self, sentence, target_word, top_k=10, targets=None
    ):
        most_levenshtein_score = None
        most_similar_word = ""
        most_bert_score = 0

        if targets:
            results = self.model.predict_mask(
                sentence.strip(), targets=targets, top_k=min(top_k, len(targets))
            )

            for result in results:
                levenshtein_score = levenshtein(result.token, target_word)

                if levenshtein_score < 3 and result.score >= self.bert_realword_thereshold:
                    most_levenshtein_score = levenshtein_score
                    most_bert_score = result.score
                    most_similar_word = result.token

                    return most_similar_word, (most_levenshtein_score, most_bert_score)

            return target_word, (0, 1)  # return original word

        else:
            targets = self.__get_most_similar_token_levenshtein(target_word)

            results = self.model.predict_mask(
                sentence.strip(),
                targets=[i["word"] for i in targets],
                top_k=min(top_k, len(targets)),
            )

            for result in results:
                levenshtein_score = levenshtein(result.token, target_word)

                if most_bert_score < result.score:
                    most_levenshtein_score = levenshtein_score
                    most_bert_score = result.score
                    most_similar_word = result.token

            return most_similar_word, (most_levenshtein_score, most_bert_score)

    def __check_sentence(self, sentence, candidate_word):
        tokens = sentence.split()
        ind = tokens.index(candidate_word)
        tokens[ind] = self.MASK

        detect_is_realword = None

        ### RealWord
        if candidate_word in self.realword_errors:
            possiblewords = self.realword_errors[candidate_word]
            possiblewords.append(candidate_word)
            possiblewords = list(set(possiblewords))

            masked_sentence = " ".join(tokens)

            (
                most_similar_word_mix,
                most_score_mix,
            ) = self.__get_most_similar_token_mix(
                masked_sentence, candidate_word, targets=possiblewords
            )

            detect_is_realword = True

        ### NonRealWord
        elif candidate_word not in self.dictionary:
            masked_sentence = " ".join(tokens)

            (
                most_similar_word_mix,
                most_score_mix,
            ) = self.__get_most_similar_token_mix(masked_sentence, candidate_word)

            detect_is_realword = False

        return pd.DataFrame(
            {
                "sentence": [sentence],
                "is_realword": [detect_is_realword],
                "mix_word": [most_similar_word_mix],
                "mix_levenshtein_score": [most_score_mix[0]],
                "mix_bert_score": [most_score_mix[1]],
            }
        )

    def __evaluate(self):
        final_df = None

        with open(self.FINAL_DATASET_DIR, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                (
                    sentence,
                    type_,
                    correct_word,
                    misspelled_word,
                ) = line.strip().split("^")

                if "correct" in type_:
                    candidate_word = correct_word
                else:
                    candidate_word = misspelled_word
                print(type_)

                df = self.__check_sentence(sentence, candidate_word)

                os.system("clear")

                df["type"] = type_
                if "correct" in type_:
                    df["correct_word"] = correct_word
                    df["candidate_word"] = correct_word
                else:
                    df["correct_word"] = correct_word
                    df["candidate_word"] = misspelled_word

                if final_df is not None:
                    final_df = pd.concat([final_df, df], axis=0).copy()
                    final_df = final_df.reset_index(drop=True)

                else:
                    final_df = df.copy()

        final_df.to_csv(self.OUTPUT_FILE_DIR)


if __name__ == "__main__":
    main_path = input("main path: ")
    model_path = input("model path, otherwise for default click enter: ")
    output_file_name = input("output file name, otherwise for default click enter: ")
    mask_token = input("mask token, otherwise for default ([MASK]) click enter: ")
    save_model = input("save model path (only if load model from url), otherwise click enter: ")
    bert_realword_thereshold = float(input("bert_realword_thereshold, otherwise for default (1e-3) click enter:"))
    
    if not mask_token:
        mask_token = "[MASK]"
    
    if not save_model:
        save_model = None
        
    if model_path and output_file_name:
        tm = TestModel(
            main_path=main_path,
            model_path=model_path,
            output_file_name=output_file_name,
            mask_token=mask_token,
            save_model=save_model,
            bert_realword_thereshold=bert_realword_thereshold
        )
    elif model_path:
        tm = TestModel(main_path=main_path, model_path=model_path, mask_token=mask_token,
            save_model=save_model,
            bert_realword_thereshold=bert_realword_thereshold)
    elif output_file_name:
        tm = TestModel(main_path=main_path, output_file_name=output_file_name, mask_token=mask_token,
            save_model=save_model,
            bert_realword_thereshold=bert_realword_thereshold)
    else:
        tm = TestModel(main_path=main_path, mask_token=mask_token,
            save_model=save_model,
            bert_realword_thereshold=bert_realword_thereshold)
