from similarity import levenshtein
from wordfreq import word_frequency
from happytransformer import HappyWordPrediction
import pandas as pd


class Test:
    def __init__(self, data_path, model_path, dictionary, homophone_realword_errors, keyboard_realword_errors,
                 substitution_realword_errors):
        
        self.FINAL_DATASET_DIR = data_path + '/final_dataset.txt'
        self.FINAL_DATAFRAME_DIR = data_path + '/results_finetuned_reversed.csv'
    
        self.model = HappyWordPrediction("BERT", load_path=model_path)
        
        self.__get_levenshtein_distance = levenshtein.get_levenshtein_similarity
        
        self.dictionary = dictionary
                
        self.realword_errors = homophone_realword_errors.copy()
        
        for key in substitution_realword_errors:
            if key in self.realword_errors:
                self.realword_errors[key] += substitution_realword_errors[key].copy()
            else:
                self.realword_errors[key] = substitution_realword_errors[key].copy()
                
        for key in keyboard_realword_errors:
            if key in self.realword_errors:
                self.realword_errors[key] += keyboard_realword_errors[key].copy()
            else:
                self.realword_errors[key] = keyboard_realword_errors[key].copy()
        
                
        
    
    
    def __get_most_similar_token_levenshtein(self, target_word):

        list_of_similars = []

        for word in self.dictionary:
            score = self.__get_levenshtein_distance(word, target_word)
            # freq = word_frequency(word, 'fa')
            if score >= 0.25:
                score = score 
            else:
                score = 0

            list_of_similars.append({'word': word, 'score': score})
        list_of_similars.sort(key=lambda x: x['score'], reverse=True)
        list_of_similars = list_of_similars[0:min(300, len(list_of_similars) - 1)]

        return list_of_similars
    
    
    def __get_most_similar_token_mix(self, sentence, target_word, targets=None):

        most_levenshtein_score = 0
        most_similar_word = ""
        most_bert_score = 0

        if targets:
            results = self.model.predict_mask(sentence.strip(), targets=targets)

            for result in results:
        
                levenshtein_score = self.__get_levenshtein_distance(result.token, target_word)

                if levenshtein_score > most_levenshtein_score:
                    most_levenshtein_score = levenshtein_score
                    most_bert_score = result.score
                    most_similar_word = result.token

        else:
            targets = self.__get_most_similar_token_levenshtein(target_word)
        
            results = self.model.predict_mask(sentence.strip(), targets=[i['word'] for i in targets])

            for result in results:
        
                levenshtein_score = self.__get_levenshtein_distance(result.token, target_word)

                if result.score > most_bert_score:
                    most_levenshtein_score = levenshtein_score
                    most_bert_score = result.score
                    most_similar_word = result.token

        # most_similar_word_bert = results[0].token
        # most_score_bert = results[0].score
        return most_similar_word, (most_levenshtein_score, most_bert_score)
    
    
    def __check_sentence(self, sentence):

        candidate_words = []
        sentences = []
        
        mix_output_words = []
        mix_levenshtein_output_scores = []
        mix_bert_output_scores = []

        detect_in_realword = []

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
                most_similar_word_mix, most_score_mix = self.__get_most_similar_token_mix(temp_sentence, token)
            
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

                temp_sentence = ""

                for i in range(0, len(tokens)):
                    if i == idx:
                        temp_sentence += "[MASK] "
                    else:
                        temp_sentence += tokens[i] + " "


                ### TYPE 2
                most_similar_word_mix, most_score_mix = self.__get_most_similar_token_mix(temp_sentence, token, targets=possiblewords)

                sentences.append(sentence)
                candidate_words.append(token)

                mix_output_words.append(most_similar_word_mix)
                mix_levenshtein_output_scores.append(most_score_mix[0])
                mix_bert_output_scores.append(most_score_mix[1])
                
                detect_in_realword.append(True)

        return pd.DataFrame({'sentence': sentences, 'candidate_word': candidate_words, 'is_realword': detect_in_realword,
                            'mix_word': mix_output_words, 'mix_levenshtein_score': mix_levenshtein_output_scores,
                            'mix_bert_score': mix_bert_output_scores})
        
        
    def __evaluate(self):

        final_df = None
        counter = 0
  
        with open(self.FINAL_DATASET_DIR, 'r', encoding='utf-8') as f:
            for line in f:    

                sentence, type_of_error, correct_word, misspelled_word = line.strip().split("^")
      
                df = self.__check_sentence(sentence)

      
                df['type_of_error'] = type_of_error
                df['correct_word'] = correct_word
                df['misspelled_word'] = misspelled_word

                if final_df is not None:
                    final_df = pd.concat([final_df, df], axis=0).copy()
                    final_df = final_df.reset_index(drop=True)
                    counter += 1
                    # if counter == 4000:
                    #     break
                    
                else:
                    final_df = df.copy()
                    
        final_df.to_csv(self.FINAL_DATASET_DIR)