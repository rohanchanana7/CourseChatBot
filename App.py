import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

course_df = pd.read_csv('courses.csv')
index = faiss.read_index('course_embeddings.index')
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title('Course Chatbot')
user_query = st.text_input("Ask a question about a course")

if st.button("Submit"):
    def embed_query(query):
        return model.encode(query)

    def extract_intent(query):
        if re.search(r'price|per session price', query, re.IGNORECASE):
            return 'price'
        elif re.search(r'lessons|how many lessons', query, re.IGNORECASE):
            return 'lessons'
        elif re.search(r'details|description|about', query, re.IGNORECASE):
            return 'description'
        else:
            return 'topic_search'

    def get_course_details(course_title, detail_type):
        course = course_df[course_df['Course Name'].str.contains(course_title, case=False)]
        
        if course.empty:
            return None
        
        if detail_type == 'price':
            return course['Price'].values[0]
        elif detail_type == 'lessons':
            return course['Lessons'].values[0]
        elif detail_type == 'description':
            return course['Description'].values[0]
        else:
            return None

    if user_query:
        intent = extract_intent(user_query)

        if intent == 'topic_search':
            st.subheader("Searching for courses related to your topic...")
            query_vector = embed_query(user_query)
            D, I = index.search(np.array([query_vector]), k=5)
            
            if len(I[0]) == 0:
                st.error("No similar courses found.")
            else:
                st.write(f"Here are some courses related to '{user_query}':")
                for idx in I[0]:
                    course_data = course_df.iloc[idx]
                    st.write(f"- **{course_data['Course Name']}**: ${course_data['Price']} per session, {course_data['Lessons']} lessons, Description: {course_data['Description']}")
        
        else:
            matching_title = None
            
            for title in course_df['Course Name']:
                if title.lower() in user_query.lower():
                    matching_title = title
                    break
            
            if matching_title:
                course_detail = get_course_details(matching_title, intent)
                
                if course_detail is not None:
                    if intent == 'price':
                        st.write(f"The price for **{matching_title}** is: ${course_detail}.")
                    elif intent == 'lessons':
                        st.write(f"The number of lessons for **{matching_title}** is: {course_detail}.")
                    elif intent == 'description':
                        st.write(f"Description of **{matching_title}**: {course_detail}.")
                else:
                    st.error("Course detail not found.")
            else:
                st.error("Could not find a matching course. Please mention a valid course title.")
