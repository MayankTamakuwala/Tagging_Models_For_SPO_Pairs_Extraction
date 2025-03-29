import csv
from collections import defaultdict

if __name__ == "__main__":

    label = defaultdict(lambda: "OTHER")
    with open('finance_articles_triplets.csv', mode='r', newline='', encoding='utf-8') as triplets_file:

            reader = csv.reader(triplets_file)
            next(reader)

            labels = ["SUBJECT", "PREDICATE", "OBJECT"]
            for row in reader:

                last_index = len(row)
                while last_index > 0 and row[last_index - 1] == '': last_index -= 1
                
                selected_columns = row[1:last_index]

                for triplet in selected_columns:
                     
                    triplet = triplet[1:-1]
                    words = triplet.split(",")

                    for i in range(3):
                        
                        curr = words[i]
                        curr.strip()

                        for word in curr.split(" "): label[curr] = labels[i]

                break

    