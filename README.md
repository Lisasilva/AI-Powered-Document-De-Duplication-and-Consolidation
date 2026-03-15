# AI-Powered Deduplication Software

This application is designed to deduplicate `.docx` files by clustering them based on content similarity, removing duplicates within each cluster, and consolidating each cluster into a single document. The deduplication process is powered by OpenAI's GPT-3.5-turbo model.

## Prerequisites

Make sure you have Python installed on your system. This application requires Python 3.6 or higher.

## Installation

1. **Navigate to the Required Path:**

   Open your terminal and navigate to the directory where the application is located:
   ```sh
   cd path/to/your/application

2. **Install Dependencies:**

    Run the following command to install all the required libraries:
    ```
    pip install -r requirements.txt

## Configuration

1. **Set Up Environment Variables:**

    Create a .env file in the root directory of your application with the following content:

        INPUT_FOLDER_PATH='path/to/your/input/folder'
        OUTPUT_FOLDER_PATH='path/to/your/output/folder'
        OPENAI_API_KEY='your-openai-api-key'
        
    Replace path/to/your/input/folder, path/to/your/output/folder, and your-openai-api-key with your actual input folder path, output folder path, and OpenAI API key respectively.

## Input File Requirements

All input files should be in .docx format.
All input files should be kept in one folder.
The folder path should be specified in the .env file under INPUT_FOLDER_PATH.

## Running the Application

To run the application, execute the following command in your terminal:
```
python Team56.py
```
After running the program, you can see the consolidated output files in the output folder you specified in the .env file.



