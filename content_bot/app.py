import os 
from apikey import apikey, serper, hugging_face
import tiktoken
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey
os.environ["SERPER_API_KEY"] = serper
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_face

st.title("Messing with LangChain and GPT")
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'duck_duck_go_research', 'wikipedia_research'], 
    template='write me a youtube video script based on this title TITLE: {title} while leveraging this duckduckgo research:{duck_duck_go_research} and this wikipedia article: {wikipedia_research}'
)


# Llms
llm = OpenAI(model_name="text-davinci-003", temperature=0.9) 


# Memory 
title_memory = ConversationBufferMemory( input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory( input_key='title', memory_key='chat_history')
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()
search = DuckDuckGoSearchRun()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    duck_duck_go_search = search.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, duck_duck_go_research=duck_duck_go_search, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)
    
    with st.expander('DuckDuckGo Research'):
      st.info(duck_duck_go_search)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)

