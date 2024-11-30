import logging
import json
import time

from pymilvus import connections
from pymilvus.model.dense import VoyageEmbeddingFunction
# from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import CohereRerankFunction

from typing import List, Dict, Any
from typing import Callable
from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)
from tqdm import tqdm
import anthropic

from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler
from telegram import Update

from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

memorize_tool = {
    "name": "memorize",
    "description": "Do not use this tool unless you are explicitly asked by user to memorize something. This tool is used to memorize messages from user not assistant. It saves the content to the database for future use.",
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to memorize. It is a text that is saved to the database."
            },
            "project_name": {
                "type": "string",
                "description": "The name of the project to which the content belongs. If there is no project name, write 'общее'"
            }
        },
        "required": ["content", "project_name"]
    }
}

class MilvusContextualRetriever:
    def __init__(
        self,
        uri="milvus.db",
        collection_name="contextual_gomunkul",
        dense_embedding_function=None,
        use_sparse=False,
        sparse_embedding_function=None,
        use_contextualize_embedding=False,
        anthropic_client=None,
        use_reranker=False,
        rerank_function=None,
    ):
        self.collection_name = collection_name

        # For Milvus-lite, uri is a local path like "./milvus.db"
        # For Milvus standalone service, uri is like "http://localhost:19530"
        # For Zilliz Clond, please set `uri` and `token`, which correspond to the [Public Endpoint and API key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#cluster-details) in Zilliz Cloud.
        self.client = MilvusClient(
            uri=uri,
            user=MILVUS_USER,
            password=MILVUS_PASSWORD,
        )

        self.embedding_function = dense_embedding_function

        self.use_sparse = use_sparse
        self.sparse_embedding_function = None

        self.use_contextualize_embedding = use_contextualize_embedding
        self.anthropic_client = anthropic_client

        self.use_reranker = use_reranker
        self.rerank_function = rerank_function

        if use_sparse is True and sparse_embedding_function:
            self.sparse_embedding_function = sparse_embedding_function
        elif sparse_embedding_function is False:
            raise ValueError(
                "Sparse embedding function cannot be None if use_sparse is False"
            )
        else:
            pass

    def build_collection(self):
        # Check if collection exists
        if self.client.list_collections().get(self.collection_name) is not None:
            print(f"Collection {self.collection_name} already exists")
            return

        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_function.dim,
        )
        if self.use_sparse is True:
            schema.add_field(
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
            )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )
        if self.use_sparse is True:
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True,
        )

    def insert_data(self, chunk, metadata):
        dense_vec = self.embedding_function([chunk])[0]
        if self.use_sparse is True:
            sparse_result = self.sparse_embedding_function.encode_documents([chunk])
            if type(sparse_result) == dict:
                sparse_vec = sparse_result["sparse"][[0]]
            else:
                sparse_vec = sparse_result[[0]]
            self.client.insert(
                collection_name=self.collection_name,
                data={
                    "dense_vector": dense_vec,
                    "sparse_vector": sparse_vec,
                    **metadata,
                },
            )
        else:
            self.client.insert(
                collection_name=self.collection_name,
                data={"dense_vector": dense_vec, **metadata},
            )

    def insert_contextualized_data(self, doc, chunk, metadata):
        contextualized_text, usage = self.situate_context(doc, chunk)
        metadata["context"] = contextualized_text
        text_to_embed = f"{chunk}\n\n{contextualized_text}"
        dense_vec = self.embedding_function([text_to_embed])[0]
        if self.use_sparse is True:
            sparse_vec = self.sparse_embedding_function.encode_documents(
                [text_to_embed]
            )["sparse"][[0]]
            self.client.insert(
                collection_name=self.collection_name,
                data={
                    "dense_vector": dense_vec,
                    "sparse_vector": sparse_vec,
                    **metadata,
                },
            )
        else:
            self.client.insert(
                collection_name=self.collection_name,
                data={"dense_vector": dense_vec, **metadata},
            )

    def situate_context(self, doc: str, chunk: str):
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        response = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20241022",
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                        },
                    ],
                },
            ],
        )
        return response.content[0].text, response.usage

    def search(self, query: str, k: int = 20, tags: List[str] = None) -> List[Dict[str, Any]]:
        filter = ""
        if tags != None and len(tags) > 0:
            filter = f"project_name in {tags}"
            print(f"Filter: {filter}")
        dense_vec = self.embedding_function([query])[0]
        if self.use_sparse is True:
            sparse_vec = self.sparse_embedding_function.encode_queries([query])[
                "sparse"
            ][[0]]

        req_list = []
        if self.use_reranker:
            k = k * 10
        if self.use_sparse is True:
            req_list = []
            dense_search_param = {
                "data": [dense_vec],
                "anns_field": "dense_vector",
                "param": {"metric_type": "IP"},
                "limit": k * 2,
            }
            dense_req = AnnSearchRequest(**dense_search_param)
            req_list.append(dense_req)

            sparse_search_param = {
                "data": [sparse_vec],
                "anns_field": "sparse_vector",
                "param": {"metric_type": "IP"},
                "limit": k * 2,
            }
            sparse_req = AnnSearchRequest(**sparse_search_param)

            req_list.append(sparse_req)

            docs = self.client.hybrid_search(
                self.collection_name,
                req_list,
                RRFRanker(),
                k,
                output_fields=[
                    "content",
                    "project_name",
                ],
            )
        else:
            docs = self.client.search(
                self.collection_name,
                data=[dense_vec],
                anns_field="dense_vector",
                filter=filter,
                limit=k,
                output_fields=[
                    "content",
                    "project_name",
                ],
            )
        if self.use_reranker and self.use_contextualize_embedding:
            reranked_texts = []
            reranked_docs = []
            for i in range(k):
                if self.use_contextualize_embedding:
                    reranked_texts.append(
                        f"{docs[0][i]['entity']['content']}\n\n{docs[0][i]['entity']['project_name']}"
                    )
                else:
                    reranked_texts.append(f"{docs[0][i]['entity']['content']}")
            results = self.rerank_function(query, reranked_texts)
            for result in results:
                reranked_docs.append(docs[0][result.index])
            docs[0] = reranked_docs
        return docs

def memorize_context(content: str, project_name: str):
    metadata = {
        "content": content,
        "project_name": project_name.lower(),
    }
    contextual_retriever.insert_data(content, metadata)

anthropic_client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
)
dense_ef = VoyageEmbeddingFunction(
    api_key=VOYAGE_API_KEY, model_name="voyage-3"
)
# sparse_ef = BGEM3EmbeddingFunction()
cohere_rf = CohereRerankFunction(api_key=COHERE_API_KEY)
contextual_retriever = MilvusContextualRetriever(
    uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
    collection_name="contextual_gomunkul",
    dense_embedding_function=dense_ef,
    use_sparse=False,
    sparse_embedding_function=None,
    use_contextualize_embedding=True,
    anthropic_client=anthropic_client,
)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    force_tool = anthropic.NOT_GIVEN
    tools = anthropic.NOT_GIVEN
    if update.message.text.lower().startswith("запомни"):
        force_tool = memorize_tool["name"]
        tools = [memorize_tool]
    # Extract hashtags from the message text
    hashtags = [word[1:].lower() for word in update.message.text.split() if word.startswith('#')]
    message_text = ' '.join(word for word in update.message.text.split() if not word.startswith('#'))
    docs = contextual_retriever.search(message_text, tags=hashtags)
    print(docs)
    if tools != anthropic.NOT_GIVEN:
        prompt = f"У тебя есть доступ к инструментам, но используй их только когда это необходимо. В своём ответе не упоминай о доступных инструментах и их использовании. Ответь на запрос: {message_text}"
    else:
        prompt = f"Ответь на запрос: {message_text}"
    context_content = ""
    for doc in docs[0]:
        if doc["distance"] > 0.3:
            project_name = "" if doc["entity"]["project_name"] in ["общее", ""] else f"проект {doc['entity']['project_name']}: "
            context_content += f"<context>{project_name}{doc['entity']['content']}</context>\n"
    
    if context_content:
        prompt = f"Вот некоторый контекст:\n{context_content}\n\nИспользуй этот контекст только если он релевантен к запросу. Не используй его если он не релевантен. В ответе не упоминай факт использования контекста.\n\n{prompt}"
    print(f"Prompt: {prompt}")

    await claude_request([
            {
                "role": "user", "content": prompt,
            }
        ],
        update,
        tools=tools,
        force_tool=force_tool,
    )
    # await update.message.reply_text(response)

async def handle_memo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Extract project name if provided after /memo command
    message_parts = update.message.text.split(maxsplit=1)
    if len(message_parts) < 2:
        await update.message.reply_text("Please provide text to memorize after /memo command")
        return
        
    text = message_parts[1]
    project_name = "общее"
    
    # Check if text starts with project name in square brackets
    if text.startswith("[") and "]" in text:
        project_end = text.find("]")
        project_name = text[1:project_end].strip()
        text = text[project_end + 1:].strip()

    if not text:
        await update.message.reply_text("Текст для запоминания не может быть пустым")
        return

    memorize_context(text, project_name)
    await update.message.reply_text(f"Для проекта [{project_name}] запомнил:\n{text}")    

bot = Application.builder().token(TELEGRAM_API_KEY).build()
bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
bot.add_handler(CommandHandler("mem", handle_memo))

async def handle_memo_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Remove "запомни" from the beginning and trim whitespace
    text = update.message.text[7:].strip()
    
    project_name = "общее"
    
    # Check if text starts with project name in square brackets
    if text.startswith("[") and "]" in text:
        project_end = text.find("]")
        project_name = text[1:project_end].strip()
        text = text[project_end + 1:].strip()

    if not text:
        await update.message.reply_text("Текст для запоминания не может быть пустым")
        return

    memorize_context(text, project_name)
    await update.message.reply_text(f"Для проекта [{project_name}] запомнил:\n{text}")


async def process_tool_call(tool_name, tool_input):
    print("Tool name:", tool_name)
    print("Tool input:", tool_input)
    if tool_name == "memorize":
        return memorize_context(tool_input["content"], tool_input["project_name"])
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

async def claude_request(msgs: List[Dict[str, str]], update: Update, tools: List[Dict[str, Any]] = anthropic.NOT_GIVEN, force_tool: str = None):
    tool_choice = anthropic.NOT_GIVEN
    if force_tool and force_tool != "":
        tool_choice = {"type": "tool", "name": force_tool}
    
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0.8,
        messages=msgs,
        tools=tools,
        tool_choice=tool_choice,
    )
    print("Response:", response)

    msgs.append(
        {"role": "assistant", "content": response.content}
    )
    if response.stop_reason == "tool_use":
        tool_use = response.content[-1]
        tool_name = tool_use.name
        tool_input = tool_use.input

        #Actually run the underlying tool functionality on our db
        tool_result = await process_tool_call(tool_name, tool_input)

        #Add our tool_result message:
        msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(tool_result)
                    }
                ],
            },
        )
        for ent in response.content:
            if ent.type == "text" and ent.text != "":
                print(f"[type=text]: {ent.text}")
                await update.message.reply_text(text=f"{ent.text}")
            elif ent.type == "tool_use":
                print(f"[type=tool_use]: {tool_name} {tool_input}")
                formatted_input = json.dumps(tool_input, indent=2, ensure_ascii=False)
                await update.message.reply_markdown_v2(text=f"Используем инструмент `{tool_name}()`:\n```json\n{formatted_input}```")
        await claude_request(msgs, update, tools=tools)
    elif response.stop_reason == "end_turn" and len(response.content) > 0 and response.content[0].text != "":
        print(f"[end_turn]: {response.content[0].text}")
        await update.message.reply_text(response.content[0].text)

def wait_for_milvus():
    max_retries = 30
    for i in range(max_retries):
        try:
            connections.connect(host='milvus', port='19530')
            print("Successfully connected to Milvus")
            return
        except Exception as e:
            print(f"Failed to connect to Milvus, attempt {i+1}/{max_retries}")
            time.sleep(2)
    raise Exception("Could not connect to Milvus after maximum retries")


if __name__ == "__main__":
    wait_for_milvus()
    contextual_retriever.build_collection()
    bot.run_polling()
