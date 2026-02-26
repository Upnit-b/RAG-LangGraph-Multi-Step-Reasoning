from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

docs = [
    Document(
        page_content="Peak Performance Gym was founded in 2015 by former Olympic athlete Marcus Chen. With over 15 years of experience in professional athletics, Marcus established the gym to provide personalized fitness solutions for people of all levels. The gym spans 10,000 square feet and features state-of-the-art equipment.",
        metadata={"source": "about.txt"}
    ),
    Document(
        page_content="Peak Performance Gym is open Monday through Friday from 5:00 AM to 11:00 PM. On weekends, our hours are 7:00 AM to 9:00 PM. We remain closed on major national holidays. Members with Premium access can enter using their key cards 24/7, including holidays.",
        metadata={"source": "hours.txt"}
    ),
    Document(
        page_content="Our membership plans include: Basic (₹1,500/month) with access to gym floor and basic equipment; Standard (₹2,500/month) adds group classes and locker facilities; Premium (₹4,000/month) includes 24/7 access, personal training sessions, and spa facilities. We offer student and senior citizen discounts of 15% on all plans. Corporate partnerships are available for companies with 10+ employees joining.",
        metadata={"source": "membership.txt"}
    ),
    Document(
        page_content="Group fitness classes at Peak Performance Gym include Yoga (beginner, intermediate, advanced), HIIT, Zumba, Spin Cycling, CrossFit, and Pilates. Beginner classes are held every Monday and Wednesday at 6:00 PM. Intermediate and advanced classes are scheduled throughout the week. The full schedule is available on our mobile app or at the reception desk.",
        metadata={"source": "classes.txt"}
    ),
    Document(
        page_content="Personal trainers at Peak Performance Gym are all certified professionals with minimum 5 years of experience. Each new member receives a complimentary fitness assessment and one free session with a trainer. Our head trainer, Neha Kapoor, specializes in rehabilitation fitness and sports-specific training. Personal training sessions can be booked individually (₹800/session) or in packages of 10 (₹7,000) or 20 (₹13,000).",
        metadata={"source": "trainers.txt"}
    ),
    Document(
        page_content="Peak Performance Gym's facilities include a cardio zone with 30+ machines, strength training area, functional fitness space, dedicated yoga studio, spin class room, swimming pool (25m), sauna and steam rooms, juice bar, and locker rooms with shower facilities. Our equipment is replaced or upgraded every 3 years to ensure members have access to the latest fitness technology.",
        metadata={"source": "facilities.txt"}
    )
]


def main():
    # Create the embedding function for RAG embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-miniLM-L6-v2")

    # Create the Database
    db = Chroma.from_documents(documents=docs, embedding=embedding_function)

    # Create retriever
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    # Create LLM model
    llm = ChatGroq(model="llama-3.1-8b-instant")

    # Create main template prompt
    template = """Answer the question based on the following context and the Chathistory. Especially take the latest question into consideration:

    Chathistory: {history}

    Context: {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # create the RAG chain
    rag_chain = prompt | llm

    # Create LangGraph Nodes and Functions

    class AgentState(TypedDict):
        messages: List[BaseMessage]
        documents: List[Document]
        on_topic: str
        rephrased_question: str
        proceed_to_generate: bool
        rephrase_count: int
        question: HumanMessage

    class GradeQuestion(BaseModel):
        score: str = Field(
            description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'")

    def question_rewriter(state: AgentState):
        """This function rewrites the function if the question is not clear or ambigious"""
        print(f"Entering question_rewriter with following state: {state}")

        # Reset state variables except for 'question' and 'messages'
        state["documents"] = []
        state["on_topic"] = ""
        state["rephrased_question"] = ""
        state["proceed_to_generate"] = False
        state["rephrase_count"] = 0

        if "messages" not in state or state["messages"] is None:
            state["messages"] = []

        if state["question"] not in state["messages"]:
            state["messages"].append(state["question"])

        if len(state["messages"]) > 1:
            # get all the conversation except the last message to get context for rephrasing the question
            conversation = state["messages"][:-1]

            # get the main user question
            current_question = state["question"].content

            messages = [SystemMessage(
                content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval.")]

            messages.extend(conversation)
            messages.append(HumanMessage(content=current_question))
            rephrase_prompt = ChatPromptTemplate.from_messages(messages).format()
            response = llm.invoke(rephrase_prompt)
            better_question = response.content.strip()
            print(f"question_rewriter: Rephrased question: {better_question}")
            state["rephrased_question"] = better_question
        else:
            state["rephrased_question"] = state["question"].content
        return state



    def question_classifier(state: AgentState):
        pass







if __name__ == "__main__":
    main()
