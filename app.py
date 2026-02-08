# ===================================================================
# Libraries
# ===================================================================

from typing import Iterable
import re, os
from pyairtable import Api
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

# ===================================================================
# CONFIGURATION
# ===================================================================



class Config:
    AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
    BASE_ID = os.getenv("BASE_ID")
    TABLE_NAME = os.getenv("TABLE_NAME")
    URI = os.getenv("MONGODB_URI")
    CERTIFICATE_PATHWAY = "mad_cert.pem"
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    ENDPOINT = "https://models.github.ai/inference"
    MODEL_NAME = "gpt-4o"
    EMBEDDINGS_MODEL = "text-embedding-3-small"



# ===================================================================
# AI MANAGER (LLM & Embeddings)
# ===================================================================

class AIMANAGER:
    """ AI MANAGER FOR THE OPENAI MODELS """
    @staticmethod
    def initialize_models(model_name: str = Config.MODEL_NAME, embeddings_model_name : str = Config.EMBEDDINGS_MODEL, temperature: float = 0.7) -> tuple:
        """
        Initialize Both LLM and the Embedding model

        Returns:
            Tuple: (ChatOpenAI, OpenAIEmbeddings)
        """
        try:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=Config.GITHUB_TOKEN,
                openai_api_base=Config.ENDPOINT,
                temperature=temperature,
                top_p=0.95,
                streaming=True
                )
            
            embeddings = OpenAIEmbeddings(
                model=embeddings_model_name,
                openai_api_key=Config.GITHUB_TOKEN,
                openai_api_base=Config.ENDPOINT
                )
            return llm, embeddings
        
        except Exception as e:
            raise RuntimeError(f"AI Manager Error: {str(e)}")
            
        
    
    @staticmethod
    def get_prompt() -> ChatPromptTemplate:
        """
        Creates and caches the ChatPromptTemplate for the Real Estate advisor.

        Returns:
            ChatPromptTemplate: A LangChain prompt template object.

        """

        try: 
            system_message = """أنت مستشار عقاري خبير في مدينتي. هدفك الأول هو مساعدة العميل بلهجة مصرية بيعية محترفة.
            التعليمات البيعية :
            1. لو طلب العميل متوفر اعرض تفاصيل الشقة بالكامل ( المساحة , السعر الاجمالي, الاوفر, المدفوع, مده التقسيط, الاستلام)
            2. البديل الأقرب: إذا لم تجد طلب العميل، اقترح أقرب وحدات من السياق (2-3 وحدات) واعرض تفاصيلها بالكامل.
            3. قارن بين الوحدات المختارة من حيث المميز في كل واحدة فيهم.
            4. الأمانة: شجع العميل على المتاح حالياً لأن "الفرص بتخلص بسرعة".
            5. ممنوع استعمال bullet points.
            6. الختام: اختم دائماً برقم التواصل: 0106."""
            
            human_message = """إليك الوحدات المتاحة حالياً (السياق):
            {context}
            سؤال العميل: {query}"""
            
            return ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", human_message)
            ])
        except Exception as e:
            raise RuntimeError(f"Prompt Initialization Error: {str(e)}")
            


# ===================================================================
# VECTOR DATABASE MANAGER (Airtable & MongoDB)
# ===================================================================


class DatabaseManager:
    """ Handles all data operations between Airtable and MongoDB Atlas """

    @staticmethod
    def _get_client() -> MongoClient:
        """
        Internal helper for MongoDB connection.
        
        Returns:
            MongoClient: Initialized MongoDB client

        """
        try:
            return MongoClient(
                Config.URI,
                tls=True,
                tlsCertificateKeyFile=Config.CERTIFICATE_PATHWAY, 
                server_api=ServerApi('1')
                )
        
        except Exception as e:
            raise RuntimeError(f"MongoDB Connection Error: {str(e)}")
            
    
    
    @staticmethod
    def clean_text(name: str) -> str:
        """ 
        Removes English characters and parentheses from text to maintain Arabic-only formatting for cleaner RAG processing.
        
        Args:
            name : The raw text string from Airtable.
            
        Returns:
            str: The cleaned Arabic text string.

        """
        return re.sub(r'[a-zA-Z\(\)]', '', name).strip()



    @staticmethod
    def str_to_numeric(value) -> int:
        """
        Extracts digits from a string or mixed-type value and converts to integer.

        Args:
            value: The raw input (string, float, or None).

        Returns:
            int: The extracted numeric value, or 0 if no digits found.
        """
        if value is None:
            return 0
        # Remove everything except digits
        cleaned = re.sub(r'[^\d]', '', str(value))
        # Check if the result is empty before converting to int
        if cleaned == '':
            return 0
        return int(cleaned)


    @staticmethod
    def get_vector_store() -> MongoDBAtlasVectorSearch:
        """
        Establishes a persistent connection to the MongoDB Atlas Vector Store.

        Returns:
            MongoDBAtlasVectorSearch: The LangChain vector store interface.

        """
        try:
            # Retrieve cached embeddings from the AI Manager
            _, embeddings = AIMANAGER.initialize_models()

            if not embeddings:
                raise RuntimeError("Embeddings model failed to initialize.")
                
            
            # Establish connection via the internal client helper
            client = DatabaseManager._get_client()

            # Define Database and Collection
            db = client["RealEstate_RAG_OpenAI"]
            collection = db["available_units"]

            return MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embeddings,
                index_name="vector_index"
            )
        
        except Exception as e:
            raise RuntimeError(f"Vector Store Connection Error: {str(e)}")
            

        


    def sync_airtable_to_mongodb() -> bool:
        """
        Synchronizes Airtable records with MongoDB Atlas.
        
        Returns:
            bool: True if sync succeeded, False otherwise.
        """
        
        try:

            client = DatabaseManager._get_client()
        
            _, embeddings = AIMANAGER.initialize_models()

            # Airtable Setup
            api = Api(Config.AIRTABLE_TOKEN)
            formula = "{حالة البيع} = 'متاحة'"
            table = api.table(Config.BASE_ID, Config.TABLE_NAME)

        
            # Fetch all records matching the formula
            records = table.all(formula=formula)

            # MongoDB Setup
            db = client["RealEstate_RAG_OpenAI"]
            collection = db["available_units"]

            # Clear current database before sync
            collection.delete_many({})
            
            # Data Processing
            documents = []
            for record in records:
                preprocessed_fields = record['fields']
                # Clean keys and values
                fields = {k.strip(): v for k, v in preprocessed_fields.items()}
                clean_project_name = DatabaseManager.clean_text(fields.get('اسم المشروع',''))
                attachments = preprocessed_fields.get('Attachments', [])
                image_url = ''
                if attachments and len(attachments) > 0:
                    image_url = attachments[0].get('url', '')
                
                meta = {
                    "اسم المشروع" : clean_project_name,
                    "المرحلة" : fields.get('المرحلة') or 'غير محدد',
                    "رقم المجموعة" : DatabaseManager.str_to_numeric(fields.get('رقم المجموعة')),
                    "رقم العمارة" : DatabaseManager.str_to_numeric(fields.get('رقم العمارة')),
                    "رقم الوحدة" : DatabaseManager.str_to_numeric(fields.get('رقم الوحدة')),
                    "نوع الوحدة" : fields.get('نوع الوحدة'),
                    "المساحة" : DatabaseManager.str_to_numeric(fields.get('المساحة')),
                    "السعر" : DatabaseManager.str_to_numeric(fields.get('اجمالى الوحده')),
                    "الاستلام" : fields.get('الاستلام'),
                    "مدة التقسيط" : DatabaseManager.str_to_numeric(fields.get('مدة التقسيط')),
                    "تاريخ الحجز": fields.get('تاريخ الحجز'),
                    "الأوفر المطلوب" : DatabaseManager.str_to_numeric(fields.get('الأوفر المطلوب')),
                    "مسدد على الوحده" : DatabaseManager.str_to_numeric(fields.get('مسدد على الوحده')),
                    "اجمالى الوحده" : DatabaseManager.str_to_numeric(fields.get('اجمالى الوحده')),
                    "رابط الصورة" : image_url if image_url else 'لا يوجد صورة حالياً'
                }
                    
            
                content = (
                    f"مشروع {clean_project_name}، {fields.get('نوع الوحدة')} في {fields.get('المرحلة') or 'غير محدد'}. "
                    f"الموقع: مجموعة {fields.get('رقم المجموعة')}، عمارة {fields.get('رقم العمارة')}، وحدة رقم {fields.get('رقم الوحدة')}. "
                    f"المساحة {fields.get('المساحة')} متر. "
                    f"التفاصيل المالية: السعر الإجمالي {fields.get('اجمالى الوحده')} جنيه، "
                    f"المبلغ المسدد {fields.get('مسدد على الوحده')} جنيه، "
                    f"والأوفر المطلوب {fields.get('الأوفر المطلوب')} جنيه. "
                    f"نظام التقسيط على {fields.get('مدة التقسيط')} ، والاستلام بعد {fields.get('الاستلام')}."
                    
                )
                # add processed data 
                documents.append({"text": content, "metadata":meta})


            if documents:
                MongoDBAtlasVectorSearch.from_texts(
                    texts=[doc["text"] for doc in documents],
                    embedding=embeddings,
                    metadatas=[doc["metadata"] for doc in documents],
                    collection=collection,
                    index_name="vector_index"
                )
                return True
            
            return False
            
            
        except Exception as e:
            raise RuntimeError(f"Sync Error: {str(e)}")

    


# ===================================================================
# CHAT LOGIC
# ===================================================================
    

def ask_real_estate_bot(user_query: str) -> Iterable[str]:
    """
    Retrieves relevant real estate units from MongoDB and returns an AI-generated response.
    
    Args:
        user_query: The question or requirement provided by the user.

    Returns:
        Iterable[str]: A stream of text chunks representing the assistant's response.

    """
    try:
        # Initialize components from managers
        model, _ = AIMANAGER.initialize_models()
        prompt_template = AIMANAGER.get_prompt()
        vector_db = DatabaseManager.get_vector_store()

        # Validation check
        if not all([model, prompt_template, vector_db]):
            yield "لم أتمكن من العثور على وحدات تطابق طلبك حالياً، ولكن يمكنني مساعدتك في خيارات أخرى"
            return
        
        docs = vector_db.similarity_search(user_query, k=5)
        if not docs:
            yield "لم أتمكن من العثور على وحدات تطابق طلبك حالياً، ولكن يمكنني مساعدتك في خيارات أخرى."
            return
        
        full_context = "\n\n".join([doc.page_content for doc in docs])
        
        # Starting the chain
        chain = prompt_template | model
        
        # Returns the stream reponse
        for chunk in chain.stream({"context": full_context, "query": user_query}):
            if chunk.content:  
                yield chunk.content
    
    except Exception as e:
        print(f"Error in returning a response {str(e)}")
        yield "حدث خطأ غير متوقع أثناء معالجة طلبك. من فضلك حاول مرة أخرى بعد قليل."











