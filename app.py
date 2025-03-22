import streamlit as st
import pickle
import re
import nltk

#loading models

clf=pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))


def regex(txt):
    cleanTxt=re.sub(r'http\S+',' ',txt)
    cleanTxt=re.sub(r'RT|CC',' ',cleanTxt)
    cleanTxt=re.sub(r'@\S+',' ',cleanTxt)
    cleanTxt=re.sub(r'#\S+',' ',cleanTxt)
    cleanTxt = re.sub(r'[%s]' % re.escape(r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"), ' ', cleanTxt)
    cleanTxt=re.sub(r'[^\x00-\x7f]',' ',cleanTxt)
    cleanTxt=re.sub(r'\s',' ',cleanTxt)


    return cleanTxt

#creating website

def main():
    st.title("Resume Screening App")
    upload= st.file_uploader('upload resume',type=['txt','pdf'])
     
    if upload is not None:
        try:
            resume_bytes= upload.read()
            resume_text=resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text=resume_bytes.decode('latin-1')

         
        cleaned_resume=regex(resume_text)
        cleaned_resume2=tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(cleaned_resume2)[0]
        

        Categories = {
        0: "Advocate",
        1: "Arts",
        2: "Automation Testing",
        3: "Blockchain",
        4: "Business Analyst",
        5: "Civil Engineer",
        6: "Data Science",
        7: "Database",
        8: "DevOps Engineer",
        9: "DotNet Developer",
        10: "ETL Developer",
        11: "Electrical Engineer",
        12: "HR",
        13: "Hadoop",
        14: "Health and Fitness",
        15: "Java Developer",
        16: "Mechanical Engineer",
        17: "Network Security Engineer",
        18: "Operations Manager",
        19: "PMO",
        20: "Software Developer",
        21: "SAP Developer",
        22: "Sales",
        23: "Testing"
        }
        category_name=Categories.get(prediction_id,"Unknown")
        st.write("Predicted Category", category_name)

if __name__=="__main__":
    main()