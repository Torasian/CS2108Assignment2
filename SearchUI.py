from Tkinter import *

class SearchUI(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)

        self.grid()
        self.create_widgets()


    def create_widgets(self):
        self.choose_button = Button(self, text="Choose a video")
        self.choose_button.grid(row=1, column=1)
        self.estimate_venue = Button(self, text="Estimate its venue")
        self.estimate_venue.grid(row=1, column=2)

# create the window
root = Tk()
# modify the window
root.title("Search UI")
root.geometry("1200x800")

app = SearchUI(root)
root.mainloop()





