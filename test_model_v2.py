from polyleven import levenshtein
from happytransformer import HappyWordPrediction
import pandas as pd
from tqdm import tqdm
import os
import re


class TestModel:
    def __init__(
        self,
        main_path="/mnt/disk1/users/naziri",
        model_path="HooshvareLab/bert-base-parsbert-uncased",
        output_file_name="result",
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
            main_path + "/train test datasets/test/final_dataset.txt"
        )
        self.OUTPUT_FILE_DIR = (
            main_path + "/evaluation results/" + output_file_name + ".csv"
        )

        print("creating dictionary ...")
        self.dictionary = self.__create_dictionary()

        print("load homophone, keyboard, substitution realword errors ...")
        total_realword_errors = self.__load_homophone_errors({})
        total_realword_errors = self.__load_keyboard_errors(total_realword_errors)
        self.realword_errors = self.__load_substitution_errors(total_realword_errors)

        print("load model ...")
        self.model = HappyWordPrediction("BERT", load_path=model_path)

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
        list_of_similars = []

        for word in self.dictionary:
            score = levenshtein(word, target_word)
            # freq = word_frequency(word, 'fa')

            list_of_similars.append({"word": word, "score": score})

        list_of_similars.sort(key=lambda x: x["score"])
        list_of_similars = list_of_similars[0 : min(k, len(list_of_similars) - 1)]

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

                if levenshtein_score < 3 and result.score >= 1e-3:
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

                if most_levenshtein_score == None or (
                    levenshtein_score < most_levenshtein_score
                ):
                    most_levenshtein_score = levenshtein_score
                    most_bert_score = result.score
                    most_similar_word = result.token

            return most_similar_word, (most_levenshtein_score, most_bert_score)

    def __check_sentence(self, sentence):
        candidate_words = []
        sentences = []

        mix_output_words = []
        mix_levenshtein_output_scores = []
        mix_bert_output_scores = []

        detect_in_realword = []

        # sentence = re.sub("\\s+", " ", sentence)
        # tokens = re.split(r"[\u200c | \\s]", sentence)
        tokens = sentence.split()

        for idx, token in enumerate(tokens):
            if len(token) < 3:
                continue

            ### NonRealWord
            if token not in self.dictionary:
                temp_sentence = ""

                for i in range(0, len(tokens)):
                    if i == idx:
                        temp_sentence += "[MASK] "
                    else:
                        temp_sentence += tokens[i] + " "

                ### TYPE 3
                (
                    most_similar_word_mix,
                    most_score_mix,
                ) = self.__get_most_similar_token_mix(temp_sentence, token)

                sentences.append(sentence)
                candidate_words.append(token)

                mix_output_words.append(most_similar_word_mix)
                mix_levenshtein_output_scores.append(most_score_mix[0])
                mix_bert_output_scores.append(most_score_mix[1])

                detect_in_realword.append(False)

            ### RealWord
            if token in self.realword_errors:
                possiblewords = self.realword_errors[token]
                possiblewords.append(token)
                possiblewords = list(set(possiblewords))

                temp_sentence = ""

                for i in range(0, len(tokens)):
                    if i == idx:
                        temp_sentence += "[MASK] "
                    else:
                        temp_sentence += tokens[i] + " "

                ### TYPE 2
                (
                    most_similar_word_mix,
                    most_score_mix,
                ) = self.__get_most_similar_token_mix(
                    temp_sentence, token, targets=possiblewords
                )

                sentences.append(sentence)
                candidate_words.append(token)

                mix_output_words.append(most_similar_word_mix)
                mix_levenshtein_output_scores.append(most_score_mix[0])
                mix_bert_output_scores.append(most_score_mix[1])

                detect_in_realword.append(True)

        return pd.DataFrame(
            {
                "sentence": sentences,
                "candidate_word": candidate_words,
                "is_realword": detect_in_realword,
                "mix_word": mix_output_words,
                "mix_levenshtein_score": mix_levenshtein_output_scores,
                "mix_bert_score": mix_bert_output_scores,
            }
        )

    def __evaluate(self):
        final_df = None
        counter = 0

        with open(self.FINAL_DATASET_DIR, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                (
                    sentence,
                    type_of_error,
                    correct_word,
                    misspelled_word,
                ) = line.strip().split("^")

                df = self.__check_sentence(sentence)

                os.system("clear")

                df["type_of_error"] = type_of_error
                df["correct_word"] = correct_word
                df["misspelled_word"] = misspelled_word

                if final_df is not None:
                    final_df = pd.concat([final_df, df], axis=0).copy()
                    final_df = final_df.reset_index(drop=True)
                    counter += 1
                    if counter == 25000:
                        break

                else:
                    final_df = df.copy()

        final_df.to_csv(self.OUTPUT_FILE_DIR)


if __name__ == "__main__":
    main_path = input("main path: ")
    model_path = input("model path, otherwise for default click space: ")
    output_file_name = input("output file name, otherwise for default click space: ")
    if model_path and output_file_name:
        tm = TestModel(
            main_path=main_path,
            model_path=model_path,
            output_file_name=output_file_name,
        )
    elif model_path:
        tm = TestModel(main_path=main_path, model_path=model_path)
    elif output_file_name:
        tm = TestModel(main_path=main_path, output_file_name=output_file_name)
    else:
        tm = TestModel(main_path=main_path)
