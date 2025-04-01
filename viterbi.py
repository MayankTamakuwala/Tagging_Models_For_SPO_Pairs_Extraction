import csv
import random
from nltk.tag import hmm

def evaluate_hmm(tagger, test_data):
    
    total = 0
    correct = 0

    for sentence in test_data:
        words = [w for w, _ in sentence]
        true_tags = [t for _, t in sentence]

        predicted_tags = [t for _, t in tagger.tag(words)]

        for pred, true in zip(predicted_tags, true_tags):
            total += 1
            if pred == true:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

if __name__ == "__main__":

    with open('finance_articles_triplets.csv', mode='r', newline='', encoding='utf-8') as article:

        reader = csv.reader(article)
    
        next(reader)
            
        data = []
        for row in reader:
            
            last_index = len(row)
            while last_index > 0 and row[last_index - 1] == '': last_index -= 1

            selected_columns = row[1:last_index]
            
            curr = []
            for s in selected_columns:

                words = s.split(",") 
                w_label = tuple([words[0].strip().lower(), words[1].strip()])    
                curr.append(w_label)

            data.append(curr)

        random.shuffle(data)
        split_index = int(0.8 * len(data))
        
        train_data = data[:split_index]
        test_data = data[split_index:]
        trainer = hmm.HiddenMarkovModelTrainer()
        tagger = trainer.train_supervised(train_data)

        accuracy = evaluate_hmm(tagger, test_data)
        print(f"Viterbi Tagging Accuracy: {accuracy:.2%}")