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

load_dotenv()

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

    # Graph Memory
    checkpointer = MemorySaver()

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

    class GradeDocument(BaseModel):
        score: str = Field(
            description="Document is relevant to the question? If yes -> 'Yes' if not -> 'No'"
        )

    def question_rewriter(state: AgentState):
        """Rewrites the user's question to be standalone for retrieval.
        Returns partial state updates only (messages + rephrased_question)."""

        print(f"Entering question_rewriter with following state: {state}")

        # Normalize inputs
        messages = state.get("messages") or []
        question = state["question"]

        # Ensure the user's latest question is recorded
        if not messages or messages[-1] is not question:
            messages = messages + question

        # If only one message and no rephrase is needed
        if len(messages) == 1:
            return {
                "messages": messages,
                "rephrased_question": question.content
            }

        # Use prior conversation (except last) to rephrase
        conversation = messages[:-1]
        current_question = question.content

        prompt_messages = [
            SystemMessage(
                content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval."),
            *conversation,
            HumanMessage(content=current_question)
        ]

        rephrase_prompt = ChatPromptTemplate.from_messages(prompt_messages)
        response = llm.invoke(rephrase_prompt.invoke({}))
        better_question = response.content.strip()

        print(f"question_rewriter: Rephrased Question: {better_question}")

        return {
            "messages": messages,
            "rephrased_question": better_question
        }

    def question_classifier(state: AgentState):
        """Classify whether the rephrased question is on-topic"""
        print("Entering question classifier")

        system_message = SystemMessage(content=""" You are a classifier that determines whether a user's question is about one of the following topics
        1. Gym History & Founder
        2. Operating Hours
        3. Membership Plans
        4. Fitness Classes
        5. Personal Trainers
        6. Facilities & Equipment
        7. Anything else about Peak Performance Gym
        If the question IS about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'.
        """)

        human_message = HumanMessage(
            content=f"User question: {state.get('rephrased_question', '')}")

        grade_prompt = ChatPromptTemplate.from_messages(
            [system_message, human_message])
        structured_llm = llm.with_structured_output(GradeQuestion)

        grader_llm_chain = grade_prompt | structured_llm

        # invoke -> returns pydantic GradeQuestion
        result = grader_llm_chain.invoke()
        on_topic_score = result.score.strip()
        print(f"question_classifier: on_topic = {on_topic_score}")
        return {"on_topic": on_topic_score}

    def on_topic_router(state: AgentState):
        """This function routes the graph depending on whether the question is on topic or not"""
        print("Entering on_topic_router")
        on_topic = state.get("on_topic", "").strip().lower()
        if on_topic == "yes":
            print("Routing to retrieve")
            return "retrieve"
        else:
            print("Routing to off_topic_response")
            return "off_topic_response"

    def retrieve(state: AgentState):
        """This function retrieves the relevatn documents"""
        print("Entering retrieve")
        documents = retriever.invoke(state["rephrased_question"])
        print(f"retrieve: Retrieved {len(documents)} documents")
        return {"documents": documents}

    def retrieval_grader(state: AgentState):
        """Grade retrieved documents for relevance and return the filtered set"""

        print("Entering retrieval_grader")

        system_message = SystemMessage(
            content="""You are a grader assessing the relevance of a retrieved document to a user question.
            Only answer with 'Yes' or 'No'.
            If the document contains information relevant to the user's question, respond with 'Yes'.
            Otherwise, respond with 'No'."""
        )

        structured_llm = llm.with_structured_output(GradeDocument)

        relevant_docs = []
        for doc in state.get("documents", []):
            human_message = HumanMessage(
                content=f"User question: {state.get('rephrased_question', '')}\n\nRetrieved document:\n{doc.page_content}"
            )
            grade_prompt = ChatPromptTemplate.from_messages(
                [system_message, human_message])

            grader_llm = grade_prompt | structured_llm
            result = grader_llm.invoke({})
            score = (result.score or "").strip().lower()
            print(
                f"Grading document: {doc.page_content[:30]}... Result: {score}"
            )
            proceed = len(relevant_docs) > 0
            print(
                f"retrieval_grader: {len(relevant_docs)} relevant docs, proceed_to_generate = {proceed}")
            return {"documents": relevant_docs, "proceed_to_generate": proceed}

    def proceed_router(state: AgentState):
        """This function decides whether the graph should proceed with generating content and invoking LLM or just say cannot answer"""

        print("Entering proceed_router")
        rephrase_count = state.get("rephrase_count", 0)
        if state.get("proceed_to_generate", False):
            print("Routing to generate_answer")
            return "generate_answer"
        elif rephrase_count >= 2:
            print("Maximum rephrase attempts reached. Cannot find relevant documents.")
            return "cannot_answer"
        else:
            print("Routing to refine_question")
            return "refine_question"

    def refine_question(state: AgentState):
        """This function refines the question if its not clear"""

        print("Entering refine_question")
        rephrase_count = state.get("rephrase_count", 0)
        if rephrase_count >= 2:
            print("Maximum rephrase attempts reached")
            return state

        question_to_refine = state.get("rephrased_question", "")

        system_message = SystemMessage(
            content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
            Provide a slightly adjusted version of the question."""
        )

        human_message = HumanMessage(
            content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question."
        )

        refine_prompt = ChatPromptTemplate.from_messages(
            [system_message, human_message])

        response = llm.invoke(refine_prompt.invoke({}))
        refined_question = response.content.strip()

        print(f"refine_question: Refined question: {refined_question}")
        return {"rephrased_question": refined_question, "rephrase_count": rephrase_count + 1}

    def generate_answer(state: AgentState):
        """This function calls the LLM and generates the answer"""

        print("Entering generate_answer")

        if "messages" not in state or state["messages"] is None:
            raise ValueError(
                "State must include 'messages' before generating an answer.")

        history = state["messages"]
        history_text = "\n".join(
            f"{m.type.upper()}: {m.content}" for m in history
        )
        documents = state["documents"]
        context = "\n\n".join(doc.page_content for doc in documents)

        rephrased_question = state["rephrased_question"]

        response = rag_chain.invoke(
            {"history": history_text, "context": context,
                "question": rephrased_question}
        )

        generation = response.content.strip()

        print(f"generate_answer: Generated response: {generation}")

        return {"messages": [generation]}


    def cannot_answer(state: AgentState):
        """This function is to trigger when the question is not relevant to RAG documents"""

        print("Entering cannot_answer")

        return {"messages": [AIMessage(content="I'm sorry, but I cannot find the information you're looking for.")]}


    def off_topic_response(state: AgentState):
        """This question is to trigger when the question is not as per RAG"""

        print("Entering off_topic_response")

        return {"messages": [AIMessage(
                content="I'm sorry, but I cannot find the information you're looking for."
                )]}

    # Workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("off_topic_response", off_topic_response)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("retrieval_grader", retrieval_grader)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("refine_question", refine_question)
    workflow.add_node("cannot_answer", cannot_answer)

    workflow.add_edge(START, "question_rewriter")
    workflow.add_edge("question_rewriter", "question_classifier")
    workflow.add_conditional_edges("question_classifier", on_topic_router, {
        "retrieve": "retrieve",
        "off_topic_response": "off_topic_response"
    })
    workflow.add_edge("retrieve", "retrieval_grader")
    workflow.add_conditional_edges("retrieval_grader", proceed_router, {
        "generate_answer": "generate_answer",
        "refine_question": "refine_question",
        "cannot_answer": "cannot_answer"
    })
    workflow.add_edge("refine_question", "retrieve")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("cannot_answer", END)
    workflow.add_edge("off_topic_response", END)

    graph = workflow.compile(checkpointer=checkpointer)

    # Test Examples

    # 1. Off Topic
    input_data = {"question": HumanMessage(
        content="What does the company Apple do?")}
    response = graph.invoke(input=input_data, config={
                            "configurable": {"thread_id": 1}})
    print(response)


if __name__ == "__main__":
    main()
