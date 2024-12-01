import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plot import create_plot, run_rt_freq, stop_recording

# main window
root = tk.Tk()
root.title("Audio Recognizer")
root.geometry("900x900")  # window size
root.configure(bg='black')

# title
label = tk.Label(root, text="recognize", font=("Helvetica", 32, "bold"), fg="white", bg="black")
label.pack(pady=40)

# buttons
play_button = tk.Button(root, text="Play", font=("Helvetica", 20, "bold"), command=lambda: run_rt_freq(label, play_button, stop_button, root, line, canvas), bg="white", fg="black")
play_button.pack(pady=40)

stop_button = tk.Button(root, text="Stop", font=("Helvetica", 20, "bold"), command=lambda: stop_recording(label, play_button, stop_button, line, canvas), bg="white", fg="black")
stop_button.pack_forget()

# plot
fig, ax, line, fill = create_plot()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=20)

# Run the application
root.mainloop()