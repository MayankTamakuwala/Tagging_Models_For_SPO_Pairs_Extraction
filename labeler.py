import csv

if __name__ == "__main__":

    url_labels = dict()
    with open('finance_articles_triplets.csv', mode='r', newline='', encoding='utf-8') as triplets_file:

            reader = csv.reader(triplets_file)
            next(reader)

            labels = ["SUBJECT", "PREDICATE", "OBJECT"]
            for row in reader:

                label = dict()
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

                # Store results for each url
                url_labels[row[0]] = label
                
                ###
                break

    with open('globenewswire_articles_finance.csv', mode='r', newline='', encoding='utf-8') as article_file, \
        open('data.csv', mode='w', newline='', encoding='utf-8') as data_file:
        
        reader = csv.reader(article_file)
        next(reader)

        for row in reader:
             
            content = row[2]
            url = row[3]

            curr_labels = url_labels[url]


            ###
            break

        # url_labels