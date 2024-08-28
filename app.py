import streamlit as st


#################
## Page Config ##
#################
st.set_page_config(
    page_title="PSO-OOP",
    #page_icon="static/dollar.png",
    layout="wide",
    initial_sidebar_state="auto",
)

## Page Title ##
st.title("PSO Algorithm")
#st.latex("POO = {a, j, d : (a $\in$ II) $\land$ (d, j $\in$ CC)}")
st.divider()


## Navigation ##
pg = st.navigation([
    st.Page("presentation/index.py", title="Presentation", icon=":material/chat:"), 
    st.Page("presentation/algorithm.py", title="Algorithm", icon=":material/insights:"),
    st.Page("presentation/credits.py", title="Credits", icon=":material/thumb_up:")
    ])
pg.run()
