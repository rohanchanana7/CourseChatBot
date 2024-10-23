from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pandas as pd
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
api = Api(app)

course_df = pd.read_csv('courses.csv')
index = faiss.read_index('course_embeddings.index')
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Course Chatbot API! Use the /chatbot endpoint to interact."})

def find_courses_by_topic(topic):
    topic_lower = topic.lower()
    matching_courses = course_df[course_df['Course Name'].str.contains(topic_lower, case=False) |
                                  course_df['Description'].str.contains(topic_lower, case=False)]
    return matching_courses[['Course Name', 'Price', 'Description', 'Lessons']].to_dict(orient='records')

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

def extract_intent(query):
    if re.search(r'price|per session price', query, re.IGNORECASE):
        return 'price'
    elif re.search(r'lessons|how many lessons', query, re.IGNORECASE):
        return 'lessons'
    elif re.search(r'details|description|about', query, re.IGNORECASE):
        return 'description'
    else:
        return 'topic_search'

def embed_query(query):
    return model.encode(query)

class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        user_query = data.get('query', None)
        
        if not user_query:
            return jsonify({"error": "Please provide a valid query."}), 400
        
        intent = extract_intent(user_query)
        
        if intent == 'topic_search':
            matching_courses = find_courses_by_topic(user_query)
            if matching_courses:
                return jsonify({
                    "query": user_query,
                    "message": f"Here are some courses related to '{user_query}':",
                    "courses": matching_courses
                })
            else:
                return jsonify({"error": "No courses found related to that topic."}), 404
        
        else:
            query_vector = embed_query(user_query)
            D, I = index.search(np.array([query_vector]), k=5)
            
            if len(I[0]) == 0:
                return jsonify({"error": "No similar courses found."}), 404

            matching_courses = []
            for idx in I[0]:
                course_data = course_df.iloc[idx]
                matching_courses.append({
                    'Course Name': course_data['Course Name'],
                    'Price': course_data['Price'],
                    'Description': course_data['Description'],
                    'Lessons': course_data['Lessons']
                })
            
            return jsonify({
                "query": user_query,
                "courses": matching_courses
            })

api.add_resource(Chatbot, '/chatbot')

if __name__ == '__main__':
    app.run(debug=True)
