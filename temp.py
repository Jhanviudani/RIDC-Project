import pandas as pd

from mainapp import load_scraped_texts, find_best_text_for_program
texts = load_scraped_texts("scraped")
print(len(texts))
for _, row in df.iterrows():
    print(row["Program Name"], bool(find_best_text_for_program(texts, row["Program Name"], row["Organization"])))
