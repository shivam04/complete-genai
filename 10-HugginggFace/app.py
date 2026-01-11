import validators, streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint
import os

## streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    hf_api_key=st.text_input("Huggingface API Token",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

repo_id="google/gemma-2-9b"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="conversational",   # 🔥 THIS is required
    max_new_tokens=150,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

def combine_docs(docs):
    return {"text": "\n\n".join(d.page_content for d in docs)}

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}
"""

prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()
                chain = RunnableLambda(combine_docs) | prompt | llm | StrOutputParser()
                output_summary = chain.invoke(docs)

                st.success(output_summary)
        except Exception as e:
            st.error(f"Exception:{e}")


