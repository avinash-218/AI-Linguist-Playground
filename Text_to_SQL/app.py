import streamlit as st
import os
import sqlite3
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

DB_NAME = 'student.db'

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Prompt
prompt = """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS, 
    SECTION and MARKS\n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM STUDENT ;
    \nExample 2 - Tell me all the students studying in Data Science class?, 
    the SQL command will be something like this SELECT * FROM STUDENT 
    where CLASS="Data Science"; 
    also the sql code should not have ``` in beginning or end and sql word in output
"""

# text to SQL query
def get_gemini_response(prompt, question):
    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content([prompt, question])
    return response.text

# SQL query to result from DB
def read_sql_query(sql_cmd, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(sql_cmd)
    rows = cursor.fetchall()
    conn.commit()
    conn.close()

    for row in rows:
        print(row)

    return rows

# streamlit app
st.set_page_config(page_title='SQL Query Retriever')
st.header('SQL Q&A')

question = st.text_input('Enter your query:', key='input')
submit = st.button('Query the LLM')

if submit:
    res = get_gemini_response(prompt, question)
    print(res)
    data = read_sql_query(res)
    st.subheader('The response is')
    for row in data:
        print(row[0])
        st.header(row[0])

