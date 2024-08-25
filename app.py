import streamlit as st


#################
## Page Config ##
#################
st.set_page_config(
    page_title="PSO-OOP",
    #page_icon="static/dollar.png",
    layout="centered",
    initial_sidebar_state="auto",
)

## Page Title ##
st.title("PSO Algorithm")
st.divider()


## Navigation ##
pg = st.navigation([
    st.Page("presentation/index.py", title="Presentation", icon=":material/chat:"), 
    st.Page("presentation/algorithm.py", title="Algorithm", icon=":material/insights:")
    ])
pg.run()
