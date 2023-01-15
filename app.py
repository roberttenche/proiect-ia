# MADE WITH ChatGPT - https://chat.openai.com/chat
import tkinter as tk
from tkinter import filedialog

from predict import predict

def get_file_and_predict():
  filepath = filedialog.askopenfilename()
  prediction = predict(filepath)
  print(prediction)

app = tk.Tk()

frame = tk.Frame(app)
frame.grid()

### ROW 0 ###
tk.Label(frame, text="Proiect Inteligenta Artificiala").grid(row=0, column=0)

### ROW 1 ###
tk.Button(frame, text="Predict", command=get_file_and_predict).grid(row=1, column=0)

### ROW 2 ###
tk.Button(frame, text="Quit", command=app.destroy).grid(row=2, column=0)



app.mainloop()