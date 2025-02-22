from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pandasai import SmartDataframe
from pandasai.llm.google_gemini import GoogleGemini
import pandas as pd
from io import StringIO
import os
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = GoogleGemini(GOOGLE_API_KEY)

df = SmartDataframe("C:/ragapi/employee.csv", config={"llm": llm})
response = df.chat("create a bar chart to show who got the highest salary")
#print(response)

app = FastAPI()

# Enable CORS for frontend interaction
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


smart_df = None  


@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Endpoint to upload a CSV file and initialize the SmartDataframe.
    """
    global smart_df

    try:
        # Read the uploaded file into a Pandas DataFrame
        file_content = await file.read()
        df = pd.read_csv(StringIO(file_content.decode("utf-8")))

        # Initialize SmartDataframe with the uploaded data and LLM configuration
        smart_df = SmartDataframe(df, config={"llm": llm})
        return {"message": "CSV file uploaded successfully and SmartDataframe initialized."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


class QueryRequest(BaseModel):
    query: str


@app.post("/chat/")
async def chat_with_csv(request: QueryRequest):
    """
    Endpoint to query the SmartDataframe using natural language.
    """
    global smart_df

    if smart_df is None:
        raise HTTPException(status_code=400, detail="No CSV file uploaded yet.")

    try:
        # Use the SmartDataframe to process the query
        response = smart_df.chat(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")




   
        
      
