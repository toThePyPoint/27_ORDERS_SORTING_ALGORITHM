import tkinter as tk
from tkinter import messagebox


# Function to display message box
def show_message(msg_text):
    root = tk.Tk()  # Create a root window
    root.withdraw()  # Hide the root window
    messagebox.showinfo("Message", msg_text)  # Display the message box
    root.destroy()  # Close the root window when the message box is closed