from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(texts):
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

courses_df = pd.read_csv('courses.csv')

courses_df['combined_text'] = courses_df.apply(
    lambda row: f"Price: {row['Price']}, Course Name: {row['Course Name']}, "
                f"Description: {row['Description']}, Lessons: {row['Lessons']}", axis=1)

combined_texts = courses_df['combined_text'].tolist()
embeddings = create_embeddings(combined_texts)

print(embeddings.shape)

import numpy as np
np.save('course_embeddings.npy', embeddings)
