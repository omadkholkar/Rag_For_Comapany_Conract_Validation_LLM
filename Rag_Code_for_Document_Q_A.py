def load_data(file_path):
     from langchain_community.document_loaders import PyPDFLoader
     loader = PyPDFLoader(file_path)
     data = loader.load()
     return data
#Function to create a chunk of the data
def chunk_data(data, chunk_size=2000):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunk = text_splitter.split_documents(data)
    return chunk

# Function to calculate the cost of embedding
def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens=sum([len(enc.encode(page.page_content)) for page in texts])
    #print(f'Total Tokens: {total_tokens}')
    #print(f'Embedding Cost in USD{total_tokens/1000*0.0004:.6f}')
    return total_tokens, total_tokens/1000*0.0004

# Function to crete vectors of the chunks
#'''def create_embeddings(chunk):
#   from langchain_openai import OpenAIEmbeddings
#    from langchain.vectorstores import Chroma
#    embedding = OpenAIEmbeddings(model='text-embedding-3-small')
 #   vector_store = Chroma.from_documents(chunk, embedding)
 #   return vector_store'''

def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone()
    
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        st.write('Deleting all indexes ... ')
        for index in indexes:
            pc.delete_index(index)
        st.write('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        st.write('Ok')

def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec
    from pinecone import ServerlessSpec
    
    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings...',end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index 
        print(f'Creating index {index_name} and embeddings ...', end='')

        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
        ) 
        )

        # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
        # inserting the embeddings into the index and returning a new Pinecone vector store object. 
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store

#Function to create question annd answer to google
def ask_question_answer_google(q,Vectors,data,k=3):
    from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0)
    retriever = Vectors.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    tmplate='''You are an helpful assistant that evaluates whether contract document met the required criterias or not according to the below details. 
In the below you will see the Criteria and then it's description.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
{question}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
You have to provide the output in below format for each criteria.
Criteria : Met or Not met\n
Explanation : Provide me the reason why it met the criteria or why it not met the criteria\n
{context}


'''

    QA_CHAIN_PROMPT=PromptTemplate(input_variables=["context","question"],template=tmplate)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    Criterias=[
        "Initial Contract Term",
        "Contract Parties",
        "Renewal Terms",
        "Scope of Services",
        "Pricing Structure",
        "General Liability Insurance",
        "Document Management",
        "Data Security",
        "Termination Cause"
    ]
    discription=['The initial contract term should be equal or more than 3 years',
    'Identify the parties involved in the agreement, atleast two parties should be present',
    'Verify the conditions and duration for possible contract renewals should be based on mutual agreement',
    'The services included under the contract should have guarantees for atleast 1 year',
    'Pricing structure can be modified during the contract based on quality of work delivered',
    'Contractor should have general liability insurance include all the uneven accidents',
    'Contractor should keep all the bills for claiming and he should store it for 7 years',
    'The data should be confidential and there should not be any breach. If it happens contractor should pay 50K USD as penalty',
    'The contractor can be terminated immediately if the performance of the contract is default']


    for i in range(len(Criterias)):
        one_question=f'Criteria {Criterias[i]} , and its discription {discription[i]}'
        answer = chain.invoke(one_question)
        print("Answer:",answer)
        simillar_text=retriever.vectorstore.similarity_search(one_question)
        print(simillar_text)
        ans=answer['result']
        st.write(f"{i+1}){Criterias[i]}")
        st.write(ans)
        st.write("Source document text:")
        st.write(f"{simillar_text[0].page_content}")
        st.write(f"From page No. {simillar_text[0].metadata['page']}")
        st.write('-'*100)
#Function to retrive answer from openAI
def ask_question_answer(q,Vectors,data,k,file_path):
    import streamlit as st
    import io
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.)
    retriever = Vectors.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    tmplate='''You are an helpful assistant that evaluates whether contract document met the required criterias or not according to the below details. 
In the below you will see the Criteria and then it's description.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
{question}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
You have to provide the output in below format for each criteria.
Criteria : Met or Not met\n
Explanation : Provide me the reason why it met the criteria or why it not met the criteria\n
{context}


'''

    QA_CHAIN_PROMPT=PromptTemplate(input_variables=["context","question"],template=tmplate)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    Criterias=[
        "Initial Contract Term",
        "Contract Parties",
        "Renewal Terms",
        "Scope of Services",
        "Pricing Structure",
        "General Liability Insurance",
        "Document Management",
        "Data Security",
        "Termination Cause"
    ]
    discription=['The initial contract term should be equal or more than 3 years',
    'Identify the parties involved in the agreement, atleast two parties should be present',
    'Verify the conditions and duration for possible contract renewals should be based on mutual agreement',
    'The services included under the contract should have guarantees for atleast 1 year',
    'Pricing structure can be modified during the contract based on quality of work delivered',
    'Contractor should have general liability insurance include all the uneven accidents',
    'Contractor should keep all the bills for claiming and he should store it for 7 years',
    'The data should be confidential and there should not be any breach. If it happens contractor should pay 50K USD as penalty',
    'The contractor can be terminated immediately if the performance of the contract is default']
  
        

    for i in range(len(Criterias)):
        one_question=f'Criteria {Criterias[i]} , and its discription {discription[i]}'
        answer = chain.invoke(one_question)
        print("Answer:",answer)
        simillar_text=retriever.vectorstore.similarity_search(one_question)
        print(simillar_text)
        ans=answer['result']
        st.write(f"{i+1}){Criterias[i]}")
        st.write(ans)
        st.write("Source document text:")
        st.write(f"{simillar_text[0].page_content}")
        st.write(f"From page No. {simillar_text[0].metadata['page']}")
        
        
# Function to Clear history session          
def clear_history():
   if 'history' in st.session_state:
       del st.session_state['history']


if __name__ == "__main__":
    import os
    import streamlit as st
    st.subheader('LLM Question-Answering Application for company Documments')
    with st.sidebar:
        LLm_model=st.radio('Select LLM model to answer',['Gemini','OpenAI'])
        if LLm_model=='OpenAI':
            api_key=st.text_input('OpenAI API Key :', type='password')
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
        else:
            api_key=st.text_input('Gemini API Key :', type='password')
            if api_key:
                os.environ['GOOGLE_API_KEY'] = api_key
        pinecone_api_key=st.text_input('Pinecone API Key :', type='password')
        os.environ['PINECONE_API_KEY'] = pinecone_api_key

        uploaded_file = st.file_uploader('Uploa a file :', type=['pdf','txt','docx'])
        chunk_size =st.number_input('Chunk size:', min_value=100,max_value=2048,value=2000,disabled=True)
        k = st.number_input('k',min_value=1,max_value=20,value=3,on_change=clear_history)
        add_data = st.button('ADD DATA',on_click=clear_history)

        if uploaded_file and add_data:
             with st.spinner('Reading, chunking and embedding file ....'):
                bytes_data=uploaded_file.read()
                file_name = os.path.join('./',uploaded_file.name)
                with open(file_name,'wb')as f:
                     f.write(bytes_data)

                
                data=load_data(file_name)
                chunks=chunk_data(data,chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunk length: {len(chunks)}')
                delete_pinecone_index()
                tokens,embedding_cost=print_embedding_cost(chunks)
                st.write(f'Embedding cost: $ {embedding_cost: .4f}')
                index_name='askadocument'
                vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)
                st.session_state.vs = vector_store
                st.session_state.dt = data
                st.session_state.fp=file_name
                st.success('File uploaded and chunked and embedded successuflly')
                
    q=st.button('Check all Criteria')
    if q:
        if 'vs' and 'dt' in st.session_state:
            vector_store=st.session_state.vs
            data =st.session_state.dt
            path= st.session_state.fp
            #st.write(f'k: {k}')
            if LLm_model=='OpenAI':
                answer=ask_question_answer(q,vector_store,data,k=k,file_path=path)
            else:
                answer=ask_question_answer_google(q,vector_store,data,k=k)
            #st.text_area('LLM Answer :',value= answer,height=200)
            #simillar_data=page_and_content(answer[1],answer[2],data)
           