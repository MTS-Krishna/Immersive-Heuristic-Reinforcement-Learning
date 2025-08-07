# ollama_chat.py
import subprocess
import json

def ask_ollama(prompt, model='mistral'):
    command = ['ollama', 'run', model]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate(prompt)
        return output.strip().split("\n")[-1]  # return last line as response
    except Exception as e:
        return f"Error: {e}"
