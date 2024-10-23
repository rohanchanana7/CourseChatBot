import csv
from langchain_community.document_loaders import UnstructuredURLLoader

url = "https://brainlox.com/courses/category/technical"
loader = UnstructuredURLLoader(urls=[url])
documents = loader.load()
courses = []

for doc in documents:
    content = doc.page_content
    lines = content.splitlines()
    course_data = []
    capturing = False

    for line in lines:
        if '$' in line:
            capturing = True
            course_data = []
        
        if capturing:
            course_data.append(line.strip())
        
        if "View Details" in line:
            if capturing:
                if len(course_data) >= 7:
                    price = course_data[0]
                    course_name = course_data[2]
                    description = course_data[4]
                    lessons = course_data[6]
                    courses.append({
                        'Price': price,
                        'Course Name': course_name,
                        'Description': description,
                        'Lessons': lessons
                    })
                course_data = []
            capturing = False

with open('courses.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['Price', 'Course Name', 'Description', 'Lessons'])
    writer.writeheader()
    for course in courses:
        writer.writerow(course)

print("Data has been successfully saved to courses.csv.")
