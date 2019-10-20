# README

run the Word_Embedding.py file to generate word vectors

To evaluate result, run this command in code-fall2019-a3/NLP_class/classes

java -classpath ../lib/commons-math3-3.5.jar::nlp/util/.*.java \
nlp.assignments.WordSimTester -embeddings ../../../output/word_vecs.txt \
-wordsim ../data5/wordsim353/combined.csv

